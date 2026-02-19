#!/usr/bin/env python3
"""
Post-grok model compression via PCA projection.

Question: Once the model has grokked, can we compress Δθ = θ_final - θ_init
by keeping only the top-k PCA directions?

Two PCA approaches:
  1. FULL-trajectory PCA: basis from all checkpoints (same as v4 — expect failure)
  2. POST-grok PCA: basis from checkpoints AFTER grokking only
     (hypothesis: the post-grok manifold might capture the generalizing solution)
  3. WEIGHT-SPACE SVD: direct SVD of Δθ reshaped per-layer (low-rank approximation)

For each: keep PC1, PC1+PC2, PC1-3, PC1-5, PC1-10 and measure test accuracy.
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


def trajectory_pca(checkpoints, model, start_idx=0, top_k=30):
    """PCA on parameter trajectory from start_idx onward."""
    thetas = []
    steps = []
    for i, (step, sd) in enumerate(checkpoints):
        if i < start_idx:
            continue
        model.load_state_dict(sd)
        theta = flatten_model_params(model).cpu().numpy()
        thetas.append(theta)
        steps.append(step)

    X = np.array(thetas)  # [T, P]
    # Center
    mean = X.mean(axis=0, keepdims=True)
    X_centered = X - mean

    print(f"    SVD on [{X_centered.shape[0]}, {X_centered.shape[1]}]...")
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

    eigenvalues = S ** 2 / (len(X_centered) - 1)
    explained = eigenvalues / eigenvalues.sum()

    k = min(top_k, len(S))
    return {
        "components": Vt[:k],  # [k, P]
        "explained": explained[:k],
        "mean": mean.flatten(),
        "steps": np.array(steps),
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


def find_grok_checkpoint_idx(checkpoints, grok_step):
    """Find checkpoint index closest to grok step."""
    if grok_step is None:
        return len(checkpoints) // 2  # fallback: midpoint
    steps = [s for s, _ in checkpoints]
    diffs = [abs(s - grok_step) for s in steps]
    return diffs.index(min(diffs))


def run_compression(seed):
    tag = f"multitask_s{seed}"
    path = RESULTS_DIR / f"{tag}.pt"
    if not path.exists():
        print(f"  [{tag}] not found")
        return None

    print(f"\n{'='*65}")
    print(f"  Seed {seed}: Loading {tag}...")
    data = torch.load(path, map_location="cpu", weights_only=False)
    cfg = MultiTaskConfig(**data["cfg"])
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
    grok_step = max(grok_add or 0, grok_mul or 0) or None
    print(f"  P={P}, ||Δθ||={delta.norm():.2f}, grok@{grok_step}")
    print(f"  {len(checkpoints)} checkpoints, steps {checkpoints[0][0]}..{checkpoints[-1][0]}")

    # Subsample to ~100 for tractability
    if len(checkpoints) > 100:
        idx = np.linspace(0, len(checkpoints)-1, 100, dtype=int)
        ckpts = [checkpoints[i] for i in idx]
    else:
        ckpts = list(checkpoints)

    # Find grok index in subsampled checkpoints
    grok_idx = find_grok_checkpoint_idx(ckpts, grok_step)
    print(f"  Grok checkpoint idx: {grok_idx} (step {ckpts[grok_idx][0]})")

    results = {}

    # ─── Reference measurements ──────────────────────────────────────
    acc = measure_accuracy(model, theta_final, cfg, device)
    results["full_model"] = acc
    print(f"\n  Full model:  add={acc['test_add']:.3f}  mul={acc['test_mul']:.3f}")

    acc = measure_accuracy(model, theta_init, cfg, device)
    results["init_model"] = acc
    print(f"  Init model:  add={acc['test_add']:.3f}  mul={acc['test_mul']:.3f}")

    k_values = [1, 2, 3, 5, 10, 15, 20, 30]

    # ─── Approach 1: Full-trajectory PCA ─────────────────────────────
    print(f"\n  --- Full-trajectory PCA ---")
    pca_full = trajectory_pca(ckpts, model, start_idx=0, top_k=30)
    B_full = torch.from_numpy(pca_full["components"].T).float()  # [P, k]

    print(f"  Eigenspectrum: " + ", ".join(
        [f"PC{i+1}={v*100:.1f}%" for i, v in enumerate(pca_full["explained"][:5])]))

    results["full_traj_pca"] = {}
    for k in k_values:
        if k > B_full.shape[1]:
            continue
        Bk = B_full[:, :k]
        coeffs = Bk.T @ delta
        delta_recon = Bk @ coeffs
        theta_recon = theta_init + delta_recon
        var_cap = (delta_recon.norm()**2 / delta.norm()**2).item()
        acc = measure_accuracy(model, theta_recon, cfg, device)
        results["full_traj_pca"][k] = {"var": var_cap, **acc}
        print(f"    PC1-{k:2d} ({var_cap*100:5.1f}% var): "
              f"add={acc['test_add']:.3f}  mul={acc['test_mul']:.3f}")

    # ─── Approach 2: Post-grok PCA ───────────────────────────────────
    print(f"\n  --- Post-grok PCA ---")
    n_postgrok = len(ckpts) - grok_idx
    if n_postgrok < 5:
        print(f"    Only {n_postgrok} post-grok checkpoints — skipping")
        results["postgrok_pca"] = None
    else:
        pca_post = trajectory_pca(ckpts, model, start_idx=grok_idx, top_k=min(30, n_postgrok-1))
        B_post = torch.from_numpy(pca_post["components"].T).float()

        print(f"  {n_postgrok} post-grok checkpoints")
        print(f"  Eigenspectrum: " + ", ".join(
            [f"PC{i+1}={v*100:.1f}%" for i, v in enumerate(pca_post["explained"][:5])]))

        results["postgrok_pca"] = {}
        for k in k_values:
            if k > B_post.shape[1]:
                continue
            Bk = B_post[:, :k]
            coeffs = Bk.T @ delta
            delta_recon = Bk @ coeffs
            theta_recon = theta_init + delta_recon
            var_cap = (delta_recon.norm()**2 / delta.norm()**2).item()
            acc = measure_accuracy(model, theta_recon, cfg, device)
            results["postgrok_pca"][k] = {"var": var_cap, **acc}
            print(f"    PC1-{k:2d} ({var_cap*100:5.1f}% var): "
                  f"add={acc['test_add']:.3f}  mul={acc['test_mul']:.3f}")

    # ─── Approach 3: Direct low-rank approximation per weight block ──
    print(f"\n  --- Per-layer weight SVD (low-rank approx) ---")
    model.load_state_dict(data["init_state"])
    init_sd = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(checkpoints[-1][1])
    final_sd = model.state_dict()

    # Find 2D weight matrices
    weight_keys = []
    for k, v in final_sd.items():
        if v.dim() == 2 and v.numel() >= 128:
            weight_keys.append(k)
    print(f"  Found {len(weight_keys)} weight matrices for SVD")

    results["layer_svd"] = {}
    for rank in [1, 2, 3, 5, 10]:
        # For each weight matrix: Δ = U S V^T, keep top-rank
        recon_sd = {k: v.clone() for k, v in final_sd.items()}
        total_params_affected = 0
        for wk in weight_keys:
            delta_w = final_sd[wk] - init_sd[wk]
            r = min(rank, min(delta_w.shape))
            U, S, Vh = torch.linalg.svd(delta_w.float(), full_matrices=False)
            delta_recon = (U[:, :r] * S[:r]) @ Vh[:r, :]
            recon_sd[wk] = init_sd[wk] + delta_recon
            total_params_affected += delta_w.numel()

        model.load_state_dict(recon_sd)
        theta_recon = flatten_model_params(model).cpu()
        delta_recon_full = theta_recon - theta_init
        var_cap = (delta_recon_full.norm()**2 / delta.norm()**2).item()
        acc = measure_accuracy(model, theta_recon, cfg, device)
        results["layer_svd"][rank] = {"var": var_cap, **acc}
        print(f"    rank-{rank:2d} ({var_cap*100:5.1f}% var): "
              f"add={acc['test_add']:.3f}  mul={acc['test_mul']:.3f}")

    # ─── Approach 4: Keep FULL Δθ but zero out small components ──────
    print(f"\n  --- Magnitude pruning of Δθ ---")
    results["pruning"] = {}
    for keep_frac in [0.01, 0.02, 0.05, 0.10, 0.20, 0.50]:
        delta_np = delta.numpy()
        threshold = np.percentile(np.abs(delta_np), (1 - keep_frac) * 100)
        mask = np.abs(delta_np) >= threshold
        delta_pruned = torch.from_numpy(delta_np * mask).float()
        theta_pruned = theta_init + delta_pruned
        var_cap = (delta_pruned.norm()**2 / delta.norm()**2).item()
        n_kept = mask.sum()
        acc = measure_accuracy(model, theta_pruned, cfg, device)
        results["pruning"][keep_frac] = {
            "var": var_cap, "n_kept": int(n_kept), "frac": keep_frac, **acc}
        print(f"    keep {keep_frac*100:5.1f}% ({n_kept:6d} params, {var_cap*100:5.1f}% var): "
              f"add={acc['test_add']:.3f}  mul={acc['test_mul']:.3f}")

    return {
        "seed": seed,
        "grok_step": grok_step,
        "P": P,
        "delta_norm": delta.norm().item(),
        "results": results,
        "pca_full_explained": pca_full["explained"].tolist(),
        "pca_post_explained": pca_post["explained"].tolist() if results["postgrok_pca"] else None,
    }


def plot_all(all_data):
    """Generate comparison plots."""

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    for task_idx, (task, label) in enumerate([("test_add", "Mod-Add"), ("test_mul", "Mod-Mul")]):
        # ─── Top row: PCA reconstruction ─────────────
        ax = axes[0, task_idx]

        # Full-traj PCA
        for d in all_data:
            ks = sorted(d["results"]["full_traj_pca"].keys())
            accs = [d["results"]["full_traj_pca"][k][task] for k in ks]
            ax.plot(ks, accs, "o-", color="#2980b9", alpha=0.3, lw=1, ms=3)
        # Mean
        pca_by_k = {}
        for d in all_data:
            for k, v in d["results"]["full_traj_pca"].items():
                pca_by_k.setdefault(k, []).append(v[task])
        ks_sorted = sorted(pca_by_k.keys())
        means = [np.mean(pca_by_k[k]) for k in ks_sorted]
        ax.plot(ks_sorted, means, "D-", color="#2980b9", lw=2.5, ms=7,
                label="Full-trajectory PCA", zorder=10)

        # Post-grok PCA
        for d in all_data:
            if d["results"]["postgrok_pca"] is None:
                continue
            ks = sorted(d["results"]["postgrok_pca"].keys())
            accs = [d["results"]["postgrok_pca"][k][task] for k in ks]
            ax.plot(ks, accs, "s-", color="#e74c3c", alpha=0.3, lw=1, ms=3)
        post_by_k = {}
        for d in all_data:
            if d["results"]["postgrok_pca"] is None:
                continue
            for k, v in d["results"]["postgrok_pca"].items():
                post_by_k.setdefault(k, []).append(v[task])
        if post_by_k:
            ks_sorted_p = sorted(post_by_k.keys())
            means_p = [np.mean(post_by_k[k]) for k in ks_sorted_p]
            ax.plot(ks_sorted_p, means_p, "s-", color="#e74c3c", lw=2.5, ms=7,
                    label="Post-grok PCA", zorder=10)

        # Full model reference
        full_acc = np.mean([d["results"]["full_model"][task] for d in all_data])
        ax.axhline(full_acc, color="green", ls="--", alpha=0.5, lw=1.5,
                   label=f"Full model ({full_acc:.2f})")
        ax.axhline(1/97, color="gray", ls=":", alpha=0.3, label="Chance")

        ax.set_xlabel("Number of PCA components (k)", fontsize=12)
        ax.set_ylabel(f"Test accuracy ({label})", fontsize=12)
        ax.set_title(f"{label}: PCA Compression", fontsize=12)
        ax.set_ylim(-0.02, 1.05)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        # ─── Bottom row: Layer SVD & Pruning ─────────
        ax2 = axes[1, task_idx]

        # Layer SVD
        for d in all_data:
            ranks = sorted(d["results"]["layer_svd"].keys())
            accs = [d["results"]["layer_svd"][r][task] for r in ranks]
            ax2.plot(ranks, accs, "o-", color="#8e44ad", alpha=0.3, lw=1, ms=3)
        svd_by_r = {}
        for d in all_data:
            for r, v in d["results"]["layer_svd"].items():
                svd_by_r.setdefault(r, []).append(v[task])
        rs = sorted(svd_by_r.keys())
        means_svd = [np.mean(svd_by_r[r]) for r in rs]
        ax2.plot(rs, means_svd, "D-", color="#8e44ad", lw=2.5, ms=7,
                label="Per-layer SVD", zorder=10)

        # Pruning (use keep fraction × 100 as x-axis)
        for d in all_data:
            fracs = sorted(d["results"]["pruning"].keys())
            accs = [d["results"]["pruning"][f][task] for f in fracs]
            ax2.plot([f*100 for f in fracs], accs, "^-", color="#f39c12",
                     alpha=0.3, lw=1, ms=3)
        prune_by_f = {}
        for d in all_data:
            for f, v in d["results"]["pruning"].items():
                prune_by_f.setdefault(f, []).append(v[task])
        fs = sorted(prune_by_f.keys())
        means_prune = [np.mean(prune_by_f[f]) for f in fs]
        ax2.plot([f*100 for f in fs], means_prune, "^-", color="#f39c12",
                lw=2.5, ms=7, label="Magnitude pruning (% params)", zorder=10)

        ax2.axhline(full_acc, color="green", ls="--", alpha=0.5, lw=1.5,
                    label=f"Full model ({full_acc:.2f})")
        ax2.axhline(1/97, color="gray", ls=":", alpha=0.3, label="Chance")

        ax2.set_xlabel("Rank / Keep%", fontsize=12)
        ax2.set_ylabel(f"Test accuracy ({label})", fontsize=12)
        ax2.set_title(f"{label}: SVD & Pruning", fontsize=12)
        ax2.set_ylim(-0.02, 1.05)
        ax2.legend(fontsize=8)
        ax2.grid(alpha=0.3)

    fig.suptitle("Post-Grok Model Compression\n"
                 "Can we 'lean down' a grokked model by removing directions?",
                 fontsize=13, y=1.03)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "figCOMP_A_compression.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved figCOMP_A_compression.png")


def main():
    PLOT_DIR.mkdir(exist_ok=True)
    device = get_device()
    print(f"Device: {device}")

    all_data = []
    for seed in SEEDS:
        r = run_compression(seed)
        if r is not None:
            all_data.append(r)

    if not all_data:
        print("No results.")
        sys.exit(1)

    # Save results
    save_path = PLOT_DIR / "postgrok_compression_results.pt"
    torch.save(all_data, save_path)
    print(f"\nSaved results to {save_path}")

    # Plot
    print(f"\n{'='*65}")
    print("  PLOTTING")
    print(f"{'='*65}")
    plot_all(all_data)

    # Summary table
    print(f"\n{'='*65}")
    print("  POST-GROK COMPRESSION SUMMARY")
    print(f"{'='*65}")

    print(f"\n  Full-Trajectory PCA Reconstruction:")
    print(f"  {'k':>5s}  {'Var%':>7s}  {'Add':>12s}  {'Mul':>12s}")
    print(f"  {'-'*5}  {'-'*7}  {'-'*12}  {'-'*12}")
    for k in [1, 2, 3, 5, 10, 20, 30]:
        adds = [d["results"]["full_traj_pca"][k]["test_add"]
                for d in all_data if k in d["results"]["full_traj_pca"]]
        muls = [d["results"]["full_traj_pca"][k]["test_mul"]
                for d in all_data if k in d["results"]["full_traj_pca"]]
        vrs = [d["results"]["full_traj_pca"][k]["var"]
               for d in all_data if k in d["results"]["full_traj_pca"]]
        if adds:
            print(f"  {k:>5d}  {np.mean(vrs)*100:>6.1f}%  "
                  f"{np.mean(adds):.3f}±{np.std(adds):.3f}  "
                  f"{np.mean(muls):.3f}±{np.std(muls):.3f}")

    print(f"\n  Post-Grok PCA Reconstruction:")
    for k in [1, 2, 3, 5, 10, 20]:
        adds, muls, vrs = [], [], []
        for d in all_data:
            if d["results"]["postgrok_pca"] and k in d["results"]["postgrok_pca"]:
                adds.append(d["results"]["postgrok_pca"][k]["test_add"])
                muls.append(d["results"]["postgrok_pca"][k]["test_mul"])
                vrs.append(d["results"]["postgrok_pca"][k]["var"])
        if adds:
            print(f"  {k:>5d}  {np.mean(vrs)*100:>6.1f}%  "
                  f"{np.mean(adds):.3f}±{np.std(adds):.3f}  "
                  f"{np.mean(muls):.3f}±{np.std(muls):.3f}")

    print(f"\n  Per-Layer SVD (low-rank):")
    for rank in [1, 2, 3, 5, 10]:
        adds = [d["results"]["layer_svd"][rank]["test_add"] for d in all_data]
        muls = [d["results"]["layer_svd"][rank]["test_mul"] for d in all_data]
        vrs = [d["results"]["layer_svd"][rank]["var"] for d in all_data]
        print(f"  {rank:>5d}  {np.mean(vrs)*100:>6.1f}%  "
              f"{np.mean(adds):.3f}±{np.std(adds):.3f}  "
              f"{np.mean(muls):.3f}±{np.std(muls):.3f}")

    print(f"\n  Magnitude Pruning:")
    for frac in [0.01, 0.02, 0.05, 0.10, 0.20, 0.50]:
        adds = [d["results"]["pruning"][frac]["test_add"] for d in all_data]
        muls = [d["results"]["pruning"][frac]["test_mul"] for d in all_data]
        vrs = [d["results"]["pruning"][frac]["var"] for d in all_data]
        nk = [d["results"]["pruning"][frac]["n_kept"] for d in all_data]
        print(f"  {frac*100:>4.0f}%  {np.mean(vrs)*100:>6.1f}%  "
              f"{np.mean(adds):.3f}±{np.std(adds):.3f}  "
              f"{np.mean(muls):.3f}±{np.std(muls):.3f}  "
              f"({int(np.mean(nk)):,d} params)")

    print("\nDone.")


if __name__ == "__main__":
    main()
