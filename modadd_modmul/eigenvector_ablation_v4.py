#!/usr/bin/env python3
"""
Eigenvector reconstruction ablation v4: GLOBAL parameter PCA.

Do PCA on the full θ(t) trajectory (all 302k params), not just attention.
Then reconstruct θ_grok = θ_init + proj_k(Δθ) using top-k global PCs.

This is the maximally faithful test: if the execution manifold is truly
rank-1, then keeping just PC1 of the full-parameter trajectory should
reconstruct a functioning grokked model.
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


def full_param_trajectory_pca(checkpoints, model, top_k=30):
    """
    PCA on full parameter trajectory.
    checkpoints: list of (step, state_dict)
    Returns: components [k, P], explained_ratio [k], scores [T, k]
    """
    # Flatten all checkpoints
    thetas = []
    steps = []
    for step, sd in checkpoints:
        model.load_state_dict(sd)
        theta = flatten_model_params(model).cpu().numpy()
        thetas.append(theta)
        steps.append(step)

    X = np.array(thetas)  # [T, P]
    # Delta from init
    X = X - X[0:1, :]  # [T, P], row 0 is all zeros
    # Remove row 0 (zero delta)
    X = X[1:, :]
    steps = steps[1:]

    # Center
    X -= X.mean(axis=0, keepdims=True)

    # Thin SVD
    print(f"    SVD on [{X.shape[0]}, {X.shape[1]}]...")
    U, S, Vt = np.linalg.svd(X, full_matrices=False)

    eigenvalues = S ** 2 / (len(X) - 1)
    explained_ratio = eigenvalues / eigenvalues.sum()

    k = min(top_k, len(S))
    return {
        "components": Vt[:k],          # [k, P]
        "explained_ratio": explained_ratio[:k],
        "singular_values": S[:k],
        "scores": (U[:, :k] * S[:k]),  # [T-1, k]
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


def run_sweep(seed, tag_prefix="multitask"):
    tag = f"{tag_prefix}_s{seed}"
    path = RESULTS_DIR / f"{tag}.pt"
    if not path.exists():
        print(f"  [{tag}] not found")
        return None

    print(f"\n{'='*60}")
    print(f"  Loading {tag}...")
    data = torch.load(path, map_location="cpu", weights_only=False)
    cfg = MultiTaskConfig(**data["cfg"])
    device = get_device()

    model = MultiTaskTransformer(cfg)
    checkpoints = data["checkpoints"]

    # Get init and final theta
    model.load_state_dict(data["init_state"])
    theta_init = flatten_model_params(model).cpu()

    model.load_state_dict(checkpoints[-1][1])
    theta_final = flatten_model_params(model).cpu()

    P = theta_init.numel()
    delta = theta_final - theta_init
    print(f"  P={P}, ||Δθ||={delta.norm():.2f}, {len(checkpoints)} checkpoints")

    # Subsample checkpoints for SVD tractability (keep ~100)
    if len(checkpoints) > 100:
        idx = np.linspace(0, len(checkpoints)-1, 100, dtype=int)
        ckpts_sub = [checkpoints[i] for i in idx]
    else:
        ckpts_sub = checkpoints
    print(f"  Using {len(ckpts_sub)} checkpoints for PCA")

    # Full-parameter PCA
    top_k = 30
    pca = full_param_trajectory_pca(ckpts_sub, model, top_k=top_k)
    K = len(pca["explained_ratio"])
    print(f"  PCA: {K} components")
    for i in range(min(10, K)):
        print(f"    PC{i+1}: {pca['explained_ratio'][i]*100:.1f}%")
    cumsum = np.cumsum(pca["explained_ratio"])
    print(f"  Cumulative: top-1={cumsum[0]*100:.1f}%, top-2={cumsum[1]*100:.1f}%, "
          f"top-3={cumsum[2]*100:.1f}%, top-5={cumsum[4]*100:.1f}%, "
          f"top-10={cumsum[min(9,K-1)]*100:.1f}%")

    # Build basis (orthonormal from PCA components)
    B = torch.from_numpy(pca["components"].T).float()  # [P, K]

    # Verify orthogonality
    gram = B.T @ B
    off_diag = (gram - torch.eye(K)).abs().max().item()
    print(f"  Basis orthogonality check: max off-diag = {off_diag:.2e}")

    # Variance of delta captured by top-k
    for k in [1, 2, 3, 5, 10, 20, K]:
        if k > K: continue
        Bk = B[:, :k]
        proj = Bk @ (Bk.T @ delta)
        v = (proj.norm() ** 2 / (delta.norm() ** 2)).item()
        print(f"    top-{k:2d}: {v:.1%} of ||Δθ||²")

    # ─── Sweep ───
    k_values = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, min(K, 30)]
    k_values = sorted(set(k for k in k_values if k <= K))

    results = []

    # Full model
    acc = measure_accuracy(model, theta_final, cfg, device)
    results.append({"condition": "Full model", "k": K+1, **acc})
    print(f"\n  {'Full model':>25s}: add={acc['test_add']:.3f} mul={acc['test_mul']:.3f}")

    # Init
    acc = measure_accuracy(model, theta_init, cfg, device)
    results.append({"condition": "Init model", "k": 0, **acc})
    print(f"  {'Init model':>25s}: add={acc['test_add']:.3f} mul={acc['test_mul']:.3f}")

    # PCA sweep
    for k in k_values:
        Bk = B[:, :k]
        coeffs = Bk.T @ delta
        delta_recon = Bk @ coeffs
        theta_recon = theta_init + delta_recon
        var_k = (delta_recon.norm() ** 2 / (delta.norm() ** 2)).item()

        acc = measure_accuracy(model, theta_recon, cfg, device)
        results.append({
            "condition": f"PC 1-{k}",
            "k": k, "var_captured": var_k, **acc,
        })
        star = " ★" if acc["test_add"] > 0.5 or acc["test_mul"] > 0.5 else ""
        print(f"  {'PC 1-'+str(k):>25s}: add={acc['test_add']:.3f} mul={acc['test_mul']:.3f}  "
              f"(var={var_k:.1%}){star}")

    # Random controls
    for k in [1, 2, 5, 10, 20]:
        if k > K: continue
        rng = torch.Generator()
        rng.manual_seed(seed + 77777)
        R = torch.randn(P, k, generator=rng)
        Q, _ = torch.linalg.qr(R, mode="reduced")
        coeffs = Q.T @ delta
        delta_rand = Q @ coeffs
        theta_rand = theta_init + delta_rand
        var_k = (delta_rand.norm() ** 2 / (delta.norm() ** 2)).item()

        acc = measure_accuracy(model, theta_rand, cfg, device)
        results.append({
            "condition": f"Random rank-{k}",
            "k": k, "var_captured": var_k, **acc,
        })
        print(f"  {'Random rank-'+str(k):>25s}: add={acc['test_add']:.3f} mul={acc['test_mul']:.3f}  "
              f"(var={var_k:.1%})")

    return {
        "seed": seed,
        "grok_step_add": data["grok_step_add"],
        "grok_step_mul": data["grok_step_mul"],
        "results": results,
        "pca_explained": pca["explained_ratio"].tolist(),
        "K": K, "P": P,
    }


def plot_accuracy_vs_k(all_results):
    """figABL4_A: Test accuracy vs number of global PCs."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for tidx, (task, label) in enumerate([("test_add", "Mod-Add"), ("test_mul", "Mod-Mul")]):
        ax = axes[tidx]

        for r in all_results:
            pca = [res for res in r["results"] if res["condition"].startswith("PC 1-")]
            ks = [res["k"] for res in pca]
            accs = [res[task] for res in pca]
            ax.plot(ks, accs, "o-", alpha=0.4, lw=1.5, ms=4, label=f"PCA s{r['seed']}")

        # Mean curve
        pca_by_k = {}
        for r in all_results:
            for res in r["results"]:
                if res["condition"].startswith("PC 1-"):
                    pca_by_k.setdefault(res["k"], []).append(res[task])
        ks = sorted(pca_by_k.keys())
        means = [np.mean(pca_by_k[k]) for k in ks]
        ax.plot(ks, means, "k-D", lw=3, ms=6, label="PCA mean", zorder=10)

        # Random
        for r in all_results:
            for res in r["results"]:
                if res["condition"].startswith("Random"):
                    ax.scatter(res["k"], res[task], marker="x", color="red",
                               s=80, zorder=8, alpha=0.6)
        ax.scatter([], [], marker="x", color="red", s=80, label="Random control")

        full = np.mean([res[task] for r in all_results for res in r["results"]
                        if res["condition"] == "Full model"])
        ax.axhline(full, color="blue", ls="--", alpha=0.5, label=f"Full model ({full:.2f})")
        ax.axhline(0.98, color="green", ls=":", alpha=0.3)
        ax.axhline(1/97, color="gray", ls=":", alpha=0.2)

        ax.set_xlabel("Number of global PCA components (k)", fontsize=12)
        ax.set_ylabel(f"Test accuracy ({label})", fontsize=12)
        ax.set_title(label, fontsize=13)
        ax.set_ylim(-0.02, 1.05)
        ax.legend(fontsize=8, loc="center right")
        ax.grid(alpha=0.3)

    # Eigenspectrum info
    mean_ev = np.mean([r["pca_explained"][:10] for r in all_results], axis=0) * 100
    ev_str = ", ".join([f"PC{i+1}={v:.0f}%" for i, v in enumerate(mean_ev[:5])])

    fig.suptitle(f"Global Parameter PCA Reconstruction\n"
                 f"θ_recon = θ_init + Σ(top-k PCA components of Δθ)\n"
                 f"Eigenspectrum: {ev_str}",
                 fontsize=12, y=1.06)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / f"figABL4_A_accuracy_vs_k.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved figABL4_A_accuracy_vs_k.png")


def plot_eigenspectrum(all_results):
    """figABL4_B: Eigenspectrum of global parameter PCA."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: individual eigenvalues
    ax = axes[0]
    for r in all_results:
        ev = np.array(r["pca_explained"][:30]) * 100
        ax.bar(np.arange(len(ev)) + 1, ev, alpha=0.3, width=0.8)
    mean_ev = np.mean([r["pca_explained"][:30] for r in all_results], axis=0) * 100
    ax.bar(np.arange(len(mean_ev)) + 1, mean_ev, alpha=0.7, color="steelblue",
           edgecolor="k", width=0.8, label="Mean")
    ax.set_xlabel("PC index", fontsize=12)
    ax.set_ylabel("Variance explained (%)", fontsize=12)
    ax.set_title("Eigenspectrum", fontsize=12)
    ax.legend()
    ax.grid(alpha=0.3, axis="y")

    # Right: cumulative
    ax = axes[1]
    for r in all_results:
        ev = np.array(r["pca_explained"][:30])
        ax.plot(np.arange(len(ev)) + 1, np.cumsum(ev) * 100, "o-", alpha=0.4, ms=3)
    mean_cum = np.cumsum(mean_ev)
    ax.plot(np.arange(len(mean_cum)) + 1, mean_cum, "k-D", lw=2, ms=5, label="Mean")
    ax.axhline(90, color="green", ls=":", alpha=0.3, label="90%")
    ax.set_xlabel("Number of PCs", fontsize=12)
    ax.set_ylabel("Cumulative variance (%)", fontsize=12)
    ax.set_title("Cumulative Variance", fontsize=12)
    ax.legend()
    ax.grid(alpha=0.3)

    fig.suptitle("Global Parameter Trajectory PCA — Eigenspectrum", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / f"figABL4_B_eigenspectrum.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved figABL4_B_eigenspectrum.png")


def main():
    PLOT_DIR.mkdir(exist_ok=True)
    device = get_device()
    print(f"Device: {device}")

    all_results = []
    for seed in SEEDS:
        r = run_sweep(seed)
        if r is not None:
            all_results.append(r)

    if not all_results:
        print("No results.")
        sys.exit(1)

    save_path = PLOT_DIR / "eigenvector_ablation_v4_results.pt"
    torch.save(all_results, save_path)
    print(f"\nSaved to {save_path}")

    print("\n" + "=" * 70)
    print("  PLOTTING")
    print("=" * 70)
    plot_accuracy_vs_k(all_results)
    plot_eigenspectrum(all_results)

    # Summary
    print(f"\n{'='*70}")
    print("  GLOBAL PCA RECONSTRUCTION SUMMARY")
    print(f"{'='*70}")
    conds = ["Init model"] + [f"PC 1-{k}" for k in [1,2,3,5,10,20,30]] + \
            [f"Random rank-{k}" for k in [1,2,5,10,20]] + ["Full model"]
    print(f"{'Condition':>25s}  {'Add':>12s}  {'Mul':>12s}  {'Var%':>6s}")
    print("-" * 65)
    for cond in conds:
        adds, muls, vcs = [], [], []
        for r in all_results:
            for res in r["results"]:
                if res["condition"] == cond:
                    adds.append(res["test_add"])
                    muls.append(res["test_mul"])
                    if "var_captured" in res:
                        vcs.append(res["var_captured"])
        if not adds:
            continue
        vc = f"{np.mean(vcs)*100:.1f}%" if vcs else "—"
        print(f"{cond:>25s}  {np.mean(adds):.3f}±{np.std(adds):.3f}  "
              f"{np.mean(muls):.3f}±{np.std(muls):.3f}  {vc:>6s}")

    # Eigenspectrum
    print(f"\nEigenspectrum (mean across seeds):")
    mean_ev = np.mean([r["pca_explained"][:10] for r in all_results], axis=0) * 100
    for i, v in enumerate(mean_ev):
        print(f"  PC{i+1}: {v:.1f}%")

    print("\nDone.")


if __name__ == "__main__":
    main()
