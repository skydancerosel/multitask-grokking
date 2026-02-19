#!/usr/bin/env python3
"""
PCA threshold scan for tri-task grokking.

For each WD × seed, test k = 1..20 PCA components to find the exact
threshold where accuracy transitions from chance to near-perfect.

Also: Test grok time delay when deleting orthogonal directions (one seed).
"""

import sys, random, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))

from train_tritask import (
    TriTaskConfig, TriTaskTransformer, build_dataset, sample_batch,
    get_device, eval_accuracy,
)
from commutator_analysis import flatten_model_params, _write_params

RESULTS_DIR = Path(__file__).parent / "results"
PLOT_DIR = Path(__file__).parent / "plots"
SEEDS = [42, 137, 2024]

# Tri-task conditions (excluding wd0 which never grokked)
WD_CONDITIONS = [
    {"wd": 1.0,  "tag": "tritask_wd1"},
    {"wd": 0.5,  "tag": "tritask_wd0.5"},
    {"wd": 0.3,  "tag": "tritask_wd0.3"},
    {"wd": 0.2,  "tag": "tritask_wd0.2"},
    {"wd": 0.1,  "tag": "tritask_wd0.1"},
]

K_VALUES = list(range(1, 21))
ACC_THRESHOLD = 0.90


def trajectory_pca(checkpoints, model, top_k=30):
    """PCA on full parameter trajectory (uncentered)."""
    thetas = []
    for step, sd in checkpoints:
        model.load_state_dict(sd)
        theta = flatten_model_params(model).cpu().numpy()
        thetas.append(theta)

    X = np.array(thetas)
    X = X - X[0:1, :]  # delta from init
    X = X[1:, :]        # remove zero row

    # Uncentered PCA — preserves drift direction essential for reconstruction
    U, S, Vt = np.linalg.svd(X, full_matrices=False)

    eigenvalues = S ** 2 / (len(X) - 1)
    explained = eigenvalues / eigenvalues.sum()

    k = min(top_k, len(S))
    return {
        "components": Vt[:k],
        "explained": explained[:k],
    }


def measure_accuracy(model, theta, cfg, device):
    _write_params(model, theta.to(device))
    model.to(device)
    model.eval()
    train_pairs, test_pairs = build_dataset(cfg.P, cfg.TRAIN_FRACTION, cfg.SEED)
    acc_add, acc_mul, acc_sq = eval_accuracy(model, test_pairs, cfg, device)
    return {"test_add": acc_add, "test_mul": acc_mul, "test_sq": acc_sq}


def run_single(wd_cond, seed):
    """Run PCA reconstruction sweep for one WD × seed."""
    tag = f"{wd_cond['tag']}_s{seed}"
    path = RESULTS_DIR / f"{tag}.pt"
    if not path.exists():
        print(f"    [{tag}] not found — skipping")
        return None

    data = torch.load(path, map_location="cpu", weights_only=False)
    cfg_dict = data["cfg"]
    cfg = TriTaskConfig(**cfg_dict)
    device = get_device()

    model = TriTaskTransformer(cfg)
    checkpoints = data["checkpoints"]

    # Get init and final
    model.load_state_dict(data["init_state"])
    theta_init = flatten_model_params(model).cpu()

    model.load_state_dict(checkpoints[-1][1])
    theta_final = flatten_model_params(model).cpu()

    delta = theta_final - theta_init
    P = theta_init.numel()

    grok_steps = data.get("grok_step", {})
    grokked = data.get("grokked", {})

    # Reference accuracy
    acc_final = measure_accuracy(model, theta_final, cfg, device)

    # Subsample checkpoints for SVD tractability
    if len(checkpoints) > 100:
        idx = np.linspace(0, len(checkpoints)-1, 100, dtype=int)
        ckpts = [checkpoints[i] for i in idx]
    else:
        ckpts = list(checkpoints)

    # PCA
    print(f"    PCA on {len(ckpts)} checkpoints, P={P}...")
    pca = trajectory_pca(ckpts, model, top_k=max(K_VALUES) + 1)
    B = torch.from_numpy(pca["components"].T).float()

    # Sweep k = 1..20
    results = {}
    for k in K_VALUES:
        if k > B.shape[1]:
            break
        Bk = B[:, :k]
        coeffs = Bk.T @ delta
        delta_recon = Bk @ coeffs
        theta_recon = theta_init + delta_recon
        var_cap = (delta_recon.norm()**2 / delta.norm()**2).item()

        acc = measure_accuracy(model, theta_recon, cfg, device)
        results[k] = {"var": var_cap, **acc}

    return {
        "wd": wd_cond["wd"],
        "seed": seed,
        "grok_steps": grok_steps,
        "grokked": grokked,
        "ref_final": acc_final,
        "explained": pca["explained"].tolist(),
        "results": results,
        "n_checkpoints": len(checkpoints),
        "P": P,
        "delta_norm": delta.norm().item(),
    }


# ═══════════════════════════════════════════════════════════════════
# Part 2: Orthogonal deletion grok delay test
# ═══════════════════════════════════════════════════════════════════

def project_gradient(model, B, strength=1.0):
    """Remove (strength fraction of) gradient component orthogonal to basis B."""
    grads = []
    params_with_grad = []
    for p in model.parameters():
        if not p.requires_grad or p.grad is None:
            continue
        grads.append(p.grad.flatten())
        params_with_grad.append(p)

    if not grads:
        return

    grad_flat = torch.cat(grads)
    device = grad_flat.device
    B_dev = B.to(device)

    grad_parallel = B_dev @ (B_dev.T @ grad_flat)
    grad_perp = grad_flat - grad_parallel
    grad_new = grad_flat - strength * grad_perp

    offset = 0
    for p in params_with_grad:
        n = p.grad.numel()
        p.grad.copy_(grad_new[offset:offset + n].view_as(p.grad))
        offset += n


def train_with_ortho_deletion(cfg, B, delete_frac, seed, max_steps=50000, t_start=500):
    """
    Train tri-task with partial deletion of orthogonal gradient component.
    delete_frac: fraction of orthogonal gradient to remove (0=baseline, 1=fully constrained)
    """
    device = get_device()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    train_pairs, test_pairs = build_dataset(cfg.P, cfg.TRAIN_FRACTION, seed)
    model = TriTaskTransformer(cfg).to(device)
    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY,
        betas=(cfg.ADAM_BETA1, cfg.ADAM_BETA2)
    )
    loss_fn = nn.CrossEntropyLoss()
    B_dev = B.to(device) if B is not None else None

    grokked = {"add": False, "mul": False, "sq": False}
    grok_step = {"add": None, "mul": None, "sq": None}
    patience = {"add": 0, "mul": 0, "sq": 0}
    records = []
    t0 = time.time()

    for step in range(1, max_steps + 1):
        model.train()
        a, b, y_add, y_mul, y_sq = sample_batch(train_pairs, cfg.BATCH_SIZE, cfg.P, device)
        logits_add, logits_mul, logits_sq = model(a, b)
        loss = loss_fn(logits_add, y_add) + loss_fn(logits_mul, y_mul) + loss_fn(logits_sq, y_sq)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)

        # Gradient projection intervention
        if step >= t_start and B_dev is not None and delete_frac > 0:
            project_gradient(model, B_dev, strength=delete_frac)

        opt.step()

        if step % 200 == 0 or step == 1:
            model.eval()
            test_add, test_mul, test_sq = eval_accuracy(model, test_pairs, cfg, device)

            rec = {"step": step, "test_add": test_add, "test_mul": test_mul, "test_sq": test_sq}
            records.append(rec)

            for task, acc in [("add", test_add), ("mul", test_mul), ("sq", test_sq)]:
                if not grokked[task] and acc >= cfg.STOP_ACC:
                    patience[task] += 1
                    if patience[task] >= cfg.STOP_PATIENCE:
                        grokked[task] = True
                        grok_step[task] = step
                else:
                    if not grokked[task]:
                        patience[task] = 0

            if all(grokked.values()):
                break

            if step % 2000 == 0:
                elapsed = (time.time() - t0) / 60
                print(f"        step {step:6d} | add={test_add:.3f} mul={test_mul:.3f} sq={test_sq:.3f} | {elapsed:.1f}m")

    elapsed = (time.time() - t0) / 60
    return {
        "grokked": grokked,
        "grok_step": grok_step,
        "records": records,
        "elapsed_min": elapsed,
        "delete_frac": delete_frac,
        "seed": seed,
    }


def run_ortho_deletion_test(seed=42, wd=1.0):
    """Run grok delay test for different deletion fractions."""
    tag_map = {1.0: "tritask_wd1", 0.5: "tritask_wd0.5", 0.1: "tritask_wd0.1"}
    tag = f"{tag_map[wd]}_s{seed}"
    path = RESULTS_DIR / f"{tag}.pt"
    if not path.exists():
        print(f"  [{tag}] not found!")
        return None

    data = torch.load(path, map_location="cpu", weights_only=False)
    cfg = TriTaskConfig(**data["cfg"])
    device = get_device()

    model = TriTaskTransformer(cfg)
    checkpoints = data["checkpoints"]

    # Subsample and do PCA
    if len(checkpoints) > 100:
        idx = np.linspace(0, len(checkpoints)-1, 100, dtype=int)
        ckpts = [checkpoints[i] for i in idx]
    else:
        ckpts = list(checkpoints)

    pca = trajectory_pca(ckpts, model, top_k=20)
    B = torch.from_numpy(pca["components"].T).float()

    print(f"\n  Ortho deletion test: WD={wd}, seed={seed}")
    print(f"  PCA basis: {B.shape}, top-5 var: {pca['explained'][:5].sum()*100:.1f}%")

    # Test deletion fractions
    fracs = [0.0, 0.10, 0.25, 0.50, 0.75, 1.0]
    results = {}

    for frac in fracs:
        print(f"\n    delete_frac={frac:.2f}:")
        r = train_with_ortho_deletion(cfg, B, frac, seed, max_steps=60000)
        results[frac] = r

        gs = r["grok_step"]
        print(f"      → add={gs['add']} mul={gs['mul']} sq={gs['sq']} ({r['elapsed_min']:.1f}m)")

    return results


# ═══════════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════════

def plot_threshold(all_data):
    wd_colors = {1.0: "#2c3e50", 0.5: "#2980b9", 0.3: "#8e44ad", 0.2: "#e67e22", 0.1: "#e74c3c"}

    # Fig 1: Accuracy vs k, colored by WD (3 tasks)
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    for task_idx, (task, label) in enumerate([
        ("test_add", "Mod-Add"), ("test_mul", "Mod-Mul"), ("test_sq", "Mod-Sq")
    ]):
        ax = axes[task_idx]

        # Individual seeds
        for d in all_data:
            wd = d["wd"]
            color = wd_colors.get(wd, "gray")
            ks = sorted(d["results"].keys())
            accs = [d["results"][k][task] for k in ks]
            ax.plot(ks, accs, "o-", color=color, alpha=0.25, lw=1, ms=3)

        # Mean per WD
        for wd in sorted(wd_colors.keys(), reverse=True):
            subset = [d for d in all_data if d["wd"] == wd]
            if not subset:
                continue
            color = wd_colors[wd]
            by_k = {}
            for d in subset:
                for k, v in d["results"].items():
                    by_k.setdefault(k, []).append(v[task])
            ks = sorted(by_k.keys())
            means = [np.mean(by_k[k]) for k in ks]
            stds = [np.std(by_k[k]) for k in ks]
            ax.plot(ks, means, "D-", color=color, lw=2.5, ms=7,
                    label=f"WD={wd}", zorder=10)
            ax.fill_between(ks,
                           [m-s for m, s in zip(means, stds)],
                           [m+s for m, s in zip(means, stds)],
                           color=color, alpha=0.1)

        ax.axhline(ACC_THRESHOLD, color="green", ls="--", alpha=0.5, lw=1.5)
        ax.axhline(1/97, color="gray", ls=":", alpha=0.3)
        ax.set_xlabel("Number of PCA components (k)", fontsize=12)
        ax.set_ylabel(f"Test accuracy ({label})", fontsize=12)
        ax.set_title(label, fontsize=13)
        ax.set_ylim(-0.02, 1.05)
        ax.set_xlim(0.5, 20.5)
        ax.set_xticks(range(1, 21))
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    fig.suptitle("Tri-Task PCA Reconstruction: Components Needed for Grokking\n"
                 "θ_recon = θ_init + Σ(top-k PCA components of Δθ)",
                 fontsize=13, y=1.03)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "figTRI_THR_A_accuracy_vs_k.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved figTRI_THR_A_accuracy_vs_k.png")

    # Fig 2: Heatmap
    fig, axes = plt.subplots(1, 3, figsize=(22, 5))
    for task_idx, (task, label) in enumerate([
        ("test_add", "Add"), ("test_mul", "Mul"), ("test_sq", "Sq")
    ]):
        ax = axes[task_idx]
        conditions = []
        matrix = []
        for d in sorted(all_data, key=lambda x: (-x["wd"], x["seed"])):
            row = []
            for k in K_VALUES:
                if k in d["results"]:
                    row.append(d["results"][k][task])
                else:
                    row.append(np.nan)
            matrix.append(row)
            conditions.append(f"WD={d['wd']:.1f} s{d['seed']}")

        matrix = np.array(matrix)
        im = ax.imshow(matrix, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1,
                       interpolation='nearest')
        ax.set_xticks(range(len(K_VALUES)))
        ax.set_xticklabels(K_VALUES, fontsize=8)
        ax.set_yticks(range(len(conditions)))
        ax.set_yticklabels(conditions, fontsize=8)
        ax.set_xlabel("k", fontsize=11)
        ax.set_title(f"{label}", fontsize=12)

        for i in range(len(conditions)):
            for j in range(len(K_VALUES)):
                val = matrix[i, j]
                if not np.isnan(val):
                    color = "white" if val < 0.5 else "black"
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                           fontsize=6, color=color)
        plt.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle("Tri-Task PCA Reconstruction Heatmap", fontsize=13, y=1.03)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "figTRI_THR_B_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved figTRI_THR_B_heatmap.png")


def plot_ortho_deletion(ortho_results, wd):
    """Plot grok delay from orthogonal deletion."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    fracs = sorted(ortho_results.keys())
    tasks = ["add", "mul", "sq"]
    colors = {"add": "#2980b9", "mul": "#27ae60", "sq": "#e74c3c"}

    # Left: grok step vs deletion fraction
    ax = axes[0]
    for task in tasks:
        gsteps = []
        for f in fracs:
            gs = ortho_results[f]["grok_step"][task]
            gsteps.append(gs if gs is not None else 60000)
        ax.plot(fracs, gsteps, "D-", color=colors[task], lw=2.5, ms=8,
                label=task.capitalize())
        # Mark failures
        for i, f in enumerate(fracs):
            if ortho_results[f]["grok_step"][task] is None:
                ax.scatter(f, 60000, color=colors[task], marker="X", s=150, zorder=10)

    ax.set_xlabel("Orthogonal deletion fraction", fontsize=13)
    ax.set_ylabel("Grok step", fontsize=13)
    ax.set_title(f"Grok Delay vs Orthogonal Deletion (WD={wd})", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    # Right: accuracy curves overlay
    ax = axes[1]
    frac_subset = [0.0, 0.25, 0.50, 1.0]
    alphas = {0.0: 1.0, 0.25: 0.7, 0.50: 0.5, 1.0: 0.3}

    for frac in frac_subset:
        if frac not in ortho_results:
            continue
        recs = ortho_results[frac]["records"]
        steps = [r["step"] for r in recs]
        for task in tasks:
            accs = [r[f"test_{task}"] for r in recs]
            ls = "-" if task == "add" else ("--" if task == "mul" else ":")
            label = f"frac={frac:.0%} {task}" if task == "add" else None
            ax.plot(steps, accs, ls, color=colors[task], alpha=alphas[frac],
                    lw=1.5, label=label)

    ax.axhline(0.98, color="green", ls=":", alpha=0.3)
    ax.set_xlabel("Training step", fontsize=12)
    ax.set_ylabel("Test accuracy", fontsize=12)
    ax.set_title("Accuracy Curves (solid=add, dash=mul, dot=sq)", fontsize=12)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.suptitle(f"Orthogonal Direction Deletion — Grok Delay (WD={wd}, seed=42)",
                 fontsize=13, y=1.03)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / f"figTRI_ORTHO_grok_delay.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved figTRI_ORTHO_grok_delay.png")


def main():
    PLOT_DIR.mkdir(exist_ok=True)
    device = get_device()
    print(f"Device: {device}")

    # ─── Part 1: PCA Threshold Scan ──────────────────────────────
    print(f"\n{'='*70}")
    print("  PART 1: PCA THRESHOLD SCAN (TRI-TASK)")
    print(f"{'='*70}")

    cache_path = PLOT_DIR / "pca_threshold_scan.pt"
    all_data = []

    if cache_path.exists():
        all_data = torch.load(cache_path, map_location="cpu", weights_only=False)
        print(f"  Loaded {len(all_data)} cached results")
        done = {(d["wd"], d["seed"]) for d in all_data}
    else:
        done = set()

    total = len(WD_CONDITIONS) * len(SEEDS)
    run_i = 0

    for wd_cond in WD_CONDITIONS:
        for seed in SEEDS:
            run_i += 1
            key = (wd_cond["wd"], seed)

            if key in done:
                print(f"  [{run_i}/{total}] WD={wd_cond['wd']} s={seed} — cached")
                continue

            print(f"\n  [{run_i}/{total}] WD={wd_cond['wd']} s={seed}")
            r = run_single(wd_cond, seed)
            if r is not None:
                all_data.append(r)
                for k in K_VALUES:
                    if k in r["results"]:
                        a = r["results"][k]
                        star = " ★" if a["test_add"] > 0.5 or a["test_mul"] > 0.5 else ""
                        print(f"    k={k:2d} ({a['var']*100:5.1f}%): "
                              f"add={a['test_add']:.3f} mul={a['test_mul']:.3f} sq={a['test_sq']:.3f}{star}")
                torch.save(all_data, cache_path)

    # Check for missing WD=0.1 s=2024
    missing_seeds = set()
    for wd_cond in WD_CONDITIONS:
        for seed in SEEDS:
            tag = f"{wd_cond['tag']}_s{seed}"
            path = RESULTS_DIR / f"{tag}.pt"
            if not path.exists():
                missing_seeds.add((wd_cond['wd'], seed))
    if missing_seeds:
        print(f"\n  Note: missing models: {missing_seeds}")

    torch.save(all_data, cache_path)

    # Threshold analysis
    print(f"\n{'='*70}")
    print("  TRI-TASK PCA THRESHOLD ANALYSIS")
    print(f"{'='*70}")

    print(f"\n  Threshold k* (min k where ALL THREE tasks > {ACC_THRESHOLD*100:.0f}%):")
    print(f"  {'WD':>5s}  {'Seed':>5s}  {'k*':>3s}  {'Var%':>7s}  {'Add':>7s}  {'Mul':>7s}  {'Sq':>7s}")
    print(f"  {'-'*5}  {'-'*5}  {'-'*3}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*7}")

    threshold_data = {}

    for d in sorted(all_data, key=lambda x: (-x["wd"], x["seed"])):
        wd = d["wd"]
        seed = d["seed"]
        k_star = None

        for k in sorted(d["results"].keys()):
            r = d["results"][k]
            if (r["test_add"] >= ACC_THRESHOLD and
                r["test_mul"] >= ACC_THRESHOLD and
                r["test_sq"] >= ACC_THRESHOLD):
                k_star = k
                break

        if k_star is not None:
            r = d["results"][k_star]
            print(f"  {wd:>5.1f}  {seed:>5d}  {k_star:>3d}  {r['var']*100:>6.1f}%  "
                  f"{r['test_add']:>6.3f}  {r['test_mul']:>6.3f}  {r['test_sq']:>6.3f}")
        else:
            print(f"  {wd:>5.1f}  {seed:>5d}  {'N/A':>3s}")

        threshold_data.setdefault(wd, []).append(k_star)

    print(f"\n  Summary by WD:")
    print(f"  {'WD':>5s}  {'k* mean':>8s}  {'k* range':>10s}")
    for wd in sorted(threshold_data.keys(), reverse=True):
        ks = [k for k in threshold_data[wd] if k is not None]
        if ks:
            print(f"  {wd:>5.1f}  {np.mean(ks):>8.1f}  {min(ks):>4d}-{max(ks):<4d}")
        else:
            print(f"  {wd:>5.1f}  {'N/A':>8s}")

    # Plot
    plot_threshold(all_data)

    # ─── Part 2: Orthogonal Deletion Grok Delay ──────────────────
    print(f"\n{'='*70}")
    print("  PART 2: ORTHOGONAL DELETION GROK DELAY TEST")
    print(f"{'='*70}")

    ortho_cache = PLOT_DIR / "ortho_deletion_results.pt"
    if ortho_cache.exists():
        ortho_results = torch.load(ortho_cache, map_location="cpu", weights_only=False)
        print(f"  Loaded cached ortho results")
    else:
        ortho_results = run_ortho_deletion_test(seed=42, wd=1.0)
        if ortho_results:
            torch.save(ortho_results, ortho_cache)

    if ortho_results:
        print(f"\n  Grok Delay Summary (WD=1.0, seed=42):")
        print(f"  {'Frac':>6s}  {'Add':>8s}  {'Mul':>8s}  {'Sq':>8s}")
        print(f"  {'-'*6}  {'-'*8}  {'-'*8}  {'-'*8}")
        for frac in sorted(ortho_results.keys()):
            gs = ortho_results[frac]["grok_step"]
            ga = str(gs["add"]) if gs["add"] else "FAIL"
            gm = str(gs["mul"]) if gs["mul"] else "FAIL"
            gsq = str(gs["sq"]) if gs["sq"] else "FAIL"
            print(f"  {frac:>6.2f}  {ga:>8s}  {gm:>8s}  {gsq:>8s}")

        plot_ortho_deletion(ortho_results, wd=1.0)

    print("\nAll done.")


if __name__ == "__main__":
    main()
