#!/usr/bin/env python3
"""
Fine-grained scan of trajectory PCA reconstruction accuracy.

For each WD × seed, test k = 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
to find the exact number of PCA components where accuracy transitions
from chance to near-perfect.

This answers: "How many independent directions in parameter space
does the grokking solution require?"
"""

import sys, random
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

from train_multitask import (
    MultiTaskConfig, MultiTaskTransformer, build_dataset,
    get_device, eval_accuracy,
)
from commutator_analysis import flatten_model_params, _write_params

RESULTS_DIR = Path(__file__).parent / "results"
PLOT_DIR = Path(__file__).parent / "plots"
SEEDS = [42, 137, 2024]

# All WD conditions with their file tags and config overrides
WD_CONDITIONS = [
    {"wd": 1.0,  "tag": "multitask",       "cfg_overrides": {}},
    {"wd": 0.5,  "tag": "multitask_wd05",  "cfg_overrides": {"WEIGHT_DECAY": 0.5}},
    {"wd": 0.3,  "tag": "multitask_wd03",  "cfg_overrides": {"WEIGHT_DECAY": 0.3}},
    {"wd": 0.2,  "tag": "multitask_wd02",  "cfg_overrides": {"WEIGHT_DECAY": 0.2}},
    {"wd": 0.1,  "tag": "multitask_wd01",  "cfg_overrides": {"WEIGHT_DECAY": 0.1}},
]

K_VALUES = list(range(1, 21))  # k = 1 to 20


def trajectory_pca(checkpoints, model, top_k=30):
    """PCA on full parameter trajectory."""
    thetas = []
    for step, sd in checkpoints:
        model.load_state_dict(sd)
        theta = flatten_model_params(model).cpu().numpy()
        thetas.append(theta)

    X = np.array(thetas)  # [T, P]
    # Delta from init
    X = X - X[0:1, :]
    X = X[1:, :]  # remove zero row

    # NOTE: Do NOT center — uncentered PCA preserves the drift direction
    # which is essential for reconstruction. Centering removes the dominant
    # direction that theta_final lies along, making reconstruction fail.
    # X -= X.mean(axis=0, keepdims=True)  # INTENTIONALLY DISABLED

    U, S, Vt = np.linalg.svd(X, full_matrices=False)

    eigenvalues = S ** 2 / (len(X) - 1)
    explained = eigenvalues / eigenvalues.sum()

    k = min(top_k, len(S))
    return {
        "components": Vt[:k],  # [k, P]
        "explained": explained[:k],
    }


def measure_accuracy(model, theta, cfg, device):
    _write_params(model, theta.to(device))
    model.to(device)
    model.eval()
    train_pairs, test_pairs = build_dataset(cfg.P, cfg.TRAIN_FRACTION, cfg.SEED)
    train_add, train_mul = eval_accuracy(model, train_pairs, cfg, device)
    test_add, test_mul = eval_accuracy(model, test_pairs, cfg, device)
    return {
        "train_add": train_add, "train_mul": train_mul,
        "test_add": test_add, "test_mul": test_mul,
    }


def run_single(wd_cond, seed):
    """Run PCA reconstruction sweep for one WD × seed."""
    tag = f"{wd_cond['tag']}_s{seed}"
    path = RESULTS_DIR / f"{tag}.pt"
    if not path.exists():
        print(f"    [{tag}] not found — skipping")
        return None

    data = torch.load(path, map_location="cpu", weights_only=False)
    cfg_dict = data["cfg"]
    cfg_dict.update(wd_cond["cfg_overrides"])
    cfg = MultiTaskConfig(**cfg_dict)
    device = get_device()

    model = MultiTaskTransformer(cfg)
    checkpoints = data["checkpoints"]

    # Get init and final
    model.load_state_dict(data["init_state"])
    theta_init = flatten_model_params(model).cpu()

    model.load_state_dict(checkpoints[-1][1])
    theta_final = flatten_model_params(model).cpu()

    delta = theta_final - theta_init
    P = theta_init.numel()

    grok_add = data.get("grok_step_add")
    grok_mul = data.get("grok_step_mul")

    # Reference accuracy
    acc_final = measure_accuracy(model, theta_final, cfg, device)

    # Subsample checkpoints for SVD
    if len(checkpoints) > 100:
        idx = np.linspace(0, len(checkpoints)-1, 100, dtype=int)
        ckpts = [checkpoints[i] for i in idx]
    else:
        ckpts = list(checkpoints)

    # PCA
    pca = trajectory_pca(ckpts, model, top_k=max(K_VALUES) + 1)
    B = torch.from_numpy(pca["components"].T).float()  # [P, k_max]

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
        "grok_add": grok_add,
        "grok_mul": grok_mul,
        "ref_final": acc_final,
        "explained": pca["explained"].tolist(),
        "results": results,
        "n_checkpoints": len(checkpoints),
        "P": P,
        "delta_norm": delta.norm().item(),
    }


def main():
    PLOT_DIR.mkdir(exist_ok=True)
    device = get_device()
    print(f"Device: {device}")

    cache_path = PLOT_DIR / "pca_threshold_scan.pt"
    all_data = []

    if cache_path.exists():
        all_data = torch.load(cache_path, map_location="cpu", weights_only=False)
        print(f"  Loaded {len(all_data)} cached results")
        # Check what's already done
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
                # Print summary line
                for k in K_VALUES:
                    if k in r["results"]:
                        acc_a = r["results"][k]["test_add"]
                        acc_m = r["results"][k]["test_mul"]
                        var = r["results"][k]["var"]
                        star = " ★" if acc_a > 0.5 or acc_m > 0.5 else ""
                        print(f"    k={k:2d} ({var*100:5.1f}%): add={acc_a:.3f} mul={acc_m:.3f}{star}")

                # Save after each condition
                torch.save(all_data, cache_path)

    print(f"\n  All done. {len(all_data)} results total.")
    torch.save(all_data, cache_path)

    # ═══════════════════════════════════════════════════════════════
    # ANALYSIS: Find threshold k* for each condition
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*75}")
    print("  PCA THRESHOLD ANALYSIS")
    print(f"{'='*75}")

    ACC_THRESHOLD = 0.90  # define "grokked" as >90% test accuracy

    print(f"\n  Threshold k* (min k where BOTH tasks > {ACC_THRESHOLD*100:.0f}%):")
    print(f"  {'WD':>5s}  {'Seed':>5s}  {'k*':>3s}  {'Var%':>7s}  {'Add':>7s}  {'Mul':>7s}  {'Grok':>8s}")
    print(f"  {'-'*5}  {'-'*5}  {'-'*3}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*8}")

    threshold_data = {}  # wd → [k* values]

    for d in sorted(all_data, key=lambda x: (x["wd"], x["seed"]), reverse=True):
        wd = d["wd"]
        seed = d["seed"]
        k_star = None

        for k in sorted(d["results"].keys()):
            r = d["results"][k]
            if r["test_add"] >= ACC_THRESHOLD and r["test_mul"] >= ACC_THRESHOLD:
                k_star = k
                break

        if k_star is not None:
            r = d["results"][k_star]
            grok = max(d["grok_add"] or 0, d["grok_mul"] or 0)
            print(f"  {wd:>5.1f}  {seed:>5d}  {k_star:>3d}  {r['var']*100:>6.1f}%  "
                  f"{r['test_add']:>6.3f}  {r['test_mul']:>6.3f}  {grok:>8d}")
        else:
            print(f"  {wd:>5.1f}  {seed:>5d}  {'N/A':>3s}  {'—':>7s}  {'—':>7s}  {'—':>7s}")

        threshold_data.setdefault(wd, []).append(k_star)

    # Summary by WD
    print(f"\n  Summary by WD:")
    print(f"  {'WD':>5s}  {'k* mean':>8s}  {'k* range':>10s}  {'seeds':>6s}")
    print(f"  {'-'*5}  {'-'*8}  {'-'*10}  {'-'*6}")
    for wd in sorted(threshold_data.keys(), reverse=True):
        ks = [k for k in threshold_data[wd] if k is not None]
        if ks:
            print(f"  {wd:>5.1f}  {np.mean(ks):>8.1f}  {min(ks):>4d}-{max(ks):<4d}  {len(ks):>6d}")
        else:
            print(f"  {wd:>5.1f}  {'N/A':>8s}  {'—':>10s}  {len(threshold_data[wd]):>6d}")

    # ═══════════════════════════════════════════════════════════════
    # PLOTTING
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*75}")
    print("  PLOTTING")
    print(f"{'='*75}")

    wd_colors = {
        1.0: "#2c3e50",
        0.5: "#2980b9",
        0.3: "#27ae60",
        0.2: "#f39c12",
        0.1: "#e74c3c",
    }

    # ─── Fig 1: Accuracy vs k, colored by WD ─────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for task_idx, (task, label) in enumerate([("test_add", "Mod-Add"), ("test_mul", "Mod-Mul")]):
        ax = axes[task_idx]

        # Individual seeds (thin lines)
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

        ax.axhline(ACC_THRESHOLD, color="green", ls="--", alpha=0.5,
                   lw=1.5, label=f"{ACC_THRESHOLD*100:.0f}% threshold")
        ax.axhline(1/97, color="gray", ls=":", alpha=0.3, label="Chance")

        ax.set_xlabel("Number of PCA components (k)", fontsize=13)
        ax.set_ylabel(f"Test accuracy ({label})", fontsize=13)
        ax.set_title(label, fontsize=14)
        ax.set_ylim(-0.02, 1.05)
        ax.set_xlim(0.5, 20.5)
        ax.set_xticks(range(1, 21))
        ax.legend(fontsize=9, loc="center right")
        ax.grid(alpha=0.3)

    fig.suptitle("Trajectory PCA Reconstruction: How Many Components for Grokking?\n"
                 "θ_recon = θ_init + Σ(top-k PCA components of Δθ)",
                 fontsize=13, y=1.03)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "figTHR_A_accuracy_vs_k.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved figTHR_A_accuracy_vs_k.png")

    # ─── Fig 2: k* threshold vs WD ──────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Left: k* vs WD (scatter + mean)
    ax = axes[0]
    for wd in sorted(threshold_data.keys()):
        ks = threshold_data[wd]
        color = wd_colors.get(wd, "gray")
        valid = [k for k in ks if k is not None]
        for k in valid:
            ax.scatter(wd, k, color=color, s=80, alpha=0.5, zorder=5)
        if valid:
            ax.scatter(wd, np.mean(valid), color=color, s=200, marker="D",
                      edgecolors="black", lw=1.5, zorder=10)

    # Connect means
    wds = sorted([wd for wd in threshold_data if
                  any(k is not None for k in threshold_data[wd])])
    means = [np.mean([k for k in threshold_data[wd] if k is not None]) for wd in wds]
    ax.plot(wds, means, "k--", lw=1.5, alpha=0.5)

    ax.set_xlabel("Weight Decay", fontsize=13)
    ax.set_ylabel("k* (min PCA components for >90% acc)", fontsize=13)
    ax.set_title("Threshold k* vs Weight Decay", fontsize=13)
    ax.set_yticks(range(1, 21))
    ax.grid(alpha=0.3)

    # Right: variance captured at k* vs WD
    ax = axes[1]
    for d in all_data:
        wd = d["wd"]
        color = wd_colors.get(wd, "gray")
        k_star = None
        for k in sorted(d["results"].keys()):
            r = d["results"][k]
            if r["test_add"] >= ACC_THRESHOLD and r["test_mul"] >= ACC_THRESHOLD:
                k_star = k
                break
        if k_star:
            var = d["results"][k_star]["var"]
            ax.scatter(wd, (1 - var) * 100, color=color, s=80, alpha=0.5, zorder=5)

    # Connect means
    var_means = {}
    for d in all_data:
        wd = d["wd"]
        k_star = None
        for k in sorted(d["results"].keys()):
            r = d["results"][k]
            if r["test_add"] >= ACC_THRESHOLD and r["test_mul"] >= ACC_THRESHOLD:
                k_star = k
                break
        if k_star:
            var_means.setdefault(wd, []).append((1 - d["results"][k_star]["var"]) * 100)

    wds_v = sorted(var_means.keys())
    means_v = [np.mean(var_means[wd]) for wd in wds_v]
    for wd in wds_v:
        color = wd_colors.get(wd, "gray")
        ax.scatter(wd, np.mean(var_means[wd]), color=color, s=200, marker="D",
                  edgecolors="black", lw=1.5, zorder=10)
    ax.plot(wds_v, means_v, "k--", lw=1.5, alpha=0.5)

    ax.set_xlabel("Weight Decay", fontsize=13)
    ax.set_ylabel("% variance in orthogonal complement at k*", fontsize=13)
    ax.set_title("How much variance is 'grokking info'?", fontsize=13)
    ax.grid(alpha=0.3)

    fig.suptitle("PCA Threshold Analysis Across Weight Decay", fontsize=13, y=1.03)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "figTHR_B_threshold_vs_wd.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved figTHR_B_threshold_vs_wd.png")

    # ─── Fig 3: Eigenspectrum comparison across WD ───────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    ax = axes[0]
    for wd in sorted(wd_colors.keys(), reverse=True):
        subset = [d for d in all_data if d["wd"] == wd]
        if not subset:
            continue
        color = wd_colors[wd]
        # Mean eigenspectrum
        max_k = min(len(d["explained"]) for d in subset)
        mean_ev = np.mean([d["explained"][:max_k] for d in subset], axis=0) * 100
        offset = list(wd_colors.keys()).index(wd) * 0.15 - 0.3
        ax.bar(np.arange(min(15, len(mean_ev))) + 1 + offset, mean_ev[:15],
               alpha=0.5, color=color, label=f"WD={wd}", width=0.15,
               align='center')

    ax.set_xlabel("PC index", fontsize=12)
    ax.set_ylabel("Variance explained (%)", fontsize=12)
    ax.set_title("Eigenspectrum by WD", fontsize=12)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis='y')

    # Cumulative
    ax = axes[1]
    for wd in sorted(wd_colors.keys(), reverse=True):
        subset = [d for d in all_data if d["wd"] == wd]
        if not subset:
            continue
        color = wd_colors[wd]
        max_k = min(len(d["explained"]) for d in subset)
        mean_ev = np.mean([d["explained"][:max_k] for d in subset], axis=0)
        cumsum = np.cumsum(mean_ev) * 100
        ax.plot(np.arange(len(cumsum[:20])) + 1, cumsum[:20], "D-",
                color=color, lw=2, ms=5, label=f"WD={wd}")

    ax.axhline(99, color="green", ls=":", alpha=0.3, label="99%")
    ax.axhline(99.9, color="orange", ls=":", alpha=0.3, label="99.9%")
    ax.set_xlabel("Number of PCs", fontsize=12)
    ax.set_ylabel("Cumulative variance (%)", fontsize=12)
    ax.set_title("Cumulative Variance", fontsize=12)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_ylim(80, 100.1)

    fig.suptitle("Eigenspectrum Comparison Across Weight Decay", fontsize=13, y=1.03)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "figTHR_C_eigenspectrum.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved figTHR_C_eigenspectrum.png")

    # ─── Fig 4: Detailed transition heatmap ──────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for task_idx, (task, label) in enumerate([("test_add", "Mod-Add"), ("test_mul", "Mod-Mul")]):
        ax = axes[task_idx]

        # Build matrix [conditions × k]
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
        ax.set_xticklabels(K_VALUES, fontsize=9)
        ax.set_yticks(range(len(conditions)))
        ax.set_yticklabels(conditions, fontsize=8)
        ax.set_xlabel("Number of PCA components (k)", fontsize=12)
        ax.set_title(f"{label} Test Accuracy", fontsize=12)

        # Annotate cells
        for i in range(len(conditions)):
            for j in range(len(K_VALUES)):
                val = matrix[i, j]
                if not np.isnan(val):
                    color = "white" if val < 0.5 else "black"
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                           fontsize=7, color=color, fontweight='bold' if val > 0.9 else 'normal')

        plt.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle("PCA Reconstruction Heatmap\n"
                 "Red = chance, Green = grokked",
                 fontsize=13, y=1.03)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "figTHR_D_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved figTHR_D_heatmap.png")

    print("\nDone.")


if __name__ == "__main__":
    main()
