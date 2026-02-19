#!/usr/bin/env python3
"""
Eigenvector reconstruction ablation v2: FULL model delta onto PCA basis.

The PCA basis lives in the full parameter space (built from attention weight
trajectory, embedded at the correct offsets, QR-orthogonalized). We project
the ENTIRE model delta (θ_final - θ_init) onto the top-k PCA directions
and measure whether the rank-k reconstruction generalizes.

This is the clean test: if keep-PC1 works, the grokking solution lives on
a 1-dimensional manifold. If keep-PC1+PC2 works, it's 2-dimensional. Etc.

Sweep: k = 1, 2, 3, 5, 8, 12, 16 (full basis)
Controls: random rank-k, init model, full model

Also: per-weight-block PCA reconstruction (attention only, non-attention at final).
"""

import sys, random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))
from pca_sweep_analysis import pca_on_trajectory, collect_trajectory

from train_multitask import (
    MultiTaskConfig, MultiTaskTransformer, build_dataset,
    get_device, eval_accuracy,
)
from commutator_analysis import (
    build_pca_basis, flatten_model_params, _write_params, _param_offsets,
    N_PCA_COMPONENTS,
)

RESULTS_DIR = Path(__file__).parent / "results"
PLOT_DIR = Path(__file__).parent / "plots"
SEEDS = [42, 137, 2024]


def build_extended_pca_basis(model, attn_logs, n_components=8, device="cpu"):
    """
    Like build_pca_basis but with more components per weight block.
    Returns basis B of shape [P, K] where K = n_layers * 4_weights * n_components.
    """
    from commutator_analysis import _param_offsets
    offsets, total_params = _param_offsets(model)
    basis_vecs = []

    for layer_idx, layer in enumerate(model.encoder.layers):
        attn_mod = layer.self_attn
        d = attn_mod.embed_dim
        in_proj_id = id(attn_mod.in_proj_weight)
        out_proj_id = id(attn_mod.out_proj.weight)

        if in_proj_id not in offsets or out_proj_id not in offsets:
            continue

        in_proj_offset = offsets[in_proj_id]
        out_proj_offset = offsets[out_proj_id]

        for wkey, local_start in [("WQ", 0), ("WK", d*d), ("WV", 2*d*d)]:
            _, mats = collect_trajectory(attn_logs, layer_idx, wkey)
            pca = pca_on_trajectory(mats, top_k=n_components)
            if pca is None:
                continue
            n_avail = min(n_components, len(pca["components"]))
            for ci in range(n_avail):
                direction = torch.from_numpy(pca["components"][ci]).float()
                gv = torch.zeros(total_params, device=device)
                start = in_proj_offset + local_start
                gv[start:start + d*d] = direction.to(device)
                basis_vecs.append(gv)

        _, mats = collect_trajectory(attn_logs, layer_idx, "WO")
        pca = pca_on_trajectory(mats, top_k=n_components)
        if pca is None:
            continue
        n_avail = min(n_components, len(pca["components"]))
        for ci in range(n_avail):
            direction = torch.from_numpy(pca["components"][ci]).float()
            gv = torch.zeros(total_params, device=device)
            gv[out_proj_offset:out_proj_offset + d*d] = direction.to(device)
            basis_vecs.append(gv)

    if not basis_vecs:
        return None
    B = torch.stack(basis_vecs, dim=1)
    B_ortho, _ = torch.linalg.qr(B.cpu(), mode="reduced")
    return B_ortho.to(device)


def reconstruct_full_model(theta_init, theta_final, B, keep_k):
    """
    Reconstruct full model: θ_recon = θ_init + proj_k(θ_final - θ_init)

    B: [P, K] orthonormal basis
    keep_k: number of basis vectors to use (first k columns)
    """
    delta = theta_final - theta_init
    B_k = B[:, :keep_k]  # [P, k]
    coeffs = B_k.T @ delta  # [k]
    delta_recon = B_k @ coeffs  # [P]
    return theta_init + delta_recon


def random_basis(total_params, n_dirs, seed):
    rng = torch.Generator()
    rng.manual_seed(seed)
    R = torch.randn(total_params, n_dirs, generator=rng)
    Q, _ = torch.linalg.qr(R, mode="reduced")
    return Q


def measure_accuracy_from_theta(model, theta, cfg, device):
    """Write theta into model, measure accuracy."""
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


def run_sweep_for_seed(seed, tag_prefix="multitask"):
    tag = f"{tag_prefix}_s{seed}"
    path = RESULTS_DIR / f"{tag}.pt"
    if not path.exists():
        print(f"  [{tag}] not found")
        return None

    print(f"\n  Loading {tag}...")
    data = torch.load(path, map_location="cpu", weights_only=False)
    cfg = MultiTaskConfig(**data["cfg"])
    device = get_device()

    attn_logs = data["attn_logs"]

    # Build model, get theta_init and theta_final
    model = MultiTaskTransformer(cfg).to(device)
    model.load_state_dict(data["init_state"])
    theta_init = flatten_model_params(model).cpu()

    model.load_state_dict(data["checkpoints"][-1][1])
    theta_final = flatten_model_params(model).cpu()

    total_params = theta_init.numel()
    delta = theta_final - theta_init
    print(f"  P={total_params}, ||delta||={delta.norm():.2f}")

    # Build extended PCA basis (8 PCs per weight block → 64 total before QR)
    max_pc_per_block = 8
    print(f"  Building PCA basis ({max_pc_per_block} PCs per block)...")
    model_cpu = MultiTaskTransformer(cfg)
    B = build_extended_pca_basis(model_cpu, attn_logs,
                                  n_components=max_pc_per_block, device="cpu")
    if B is None:
        print("  No PCA basis!")
        return None
    print(f"  Basis shape: {B.shape} ({B.shape[1]} directions)")

    # Variance captured by basis
    delta_proj = B @ (B.T @ delta)
    var_captured = (delta_proj.norm() ** 2) / (delta.norm() ** 2)
    print(f"  Full basis captures {var_captured:.1%} of ||Δθ||²")

    # Per-k variance
    for k in [1, 2, 3, 5, 8, 16, B.shape[1]]:
        if k > B.shape[1]:
            continue
        Bk = B[:, :k]
        proj = Bk @ (Bk.T @ delta)
        v = (proj.norm() ** 2) / (delta.norm() ** 2)
        print(f"    top-{k:2d}: {v:.1%}")

    # ─── Sweep: keep top-k PCs ───
    k_values = [1, 2, 3, 4, 5, 6, 8, 10, 12, 16]
    k_values = [k for k in k_values if k <= B.shape[1]]

    results = []

    # Full model baseline
    model.to(device)
    acc = measure_accuracy_from_theta(model, theta_final, cfg, device)
    results.append({"condition": "Full model", "k": -1, **acc})
    print(f"    {'Full model':>25s}: add={acc['test_add']:.3f}, mul={acc['test_mul']:.3f}")

    # Init baseline
    acc = measure_accuracy_from_theta(model, theta_init, cfg, device)
    results.append({"condition": "Init model", "k": 0, **acc})
    print(f"    {'Init model':>25s}: add={acc['test_add']:.3f}, mul={acc['test_mul']:.3f}")

    # PCA reconstruction sweep
    for k in k_values:
        theta_recon = reconstruct_full_model(theta_init, theta_final, B, k)
        acc = measure_accuracy_from_theta(model, theta_recon, cfg, device)
        var_k = ((B[:, :k].T @ delta).norm() ** 2 / (delta.norm() ** 2)).item()
        results.append({
            "condition": f"PCA top-{k}",
            "k": k,
            "var_captured": var_k,
            **acc,
        })
        print(f"    {'PCA top-' + str(k):>25s}: add={acc['test_add']:.3f}, mul={acc['test_mul']:.3f}  (var={var_k:.1%})")

    # Random reconstruction controls (match k=1, 2, 16)
    for k in [1, 2, 16]:
        R = random_basis(total_params, k, seed=seed + 77777)
        coeffs = R.T @ delta
        theta_recon = theta_init + R @ coeffs
        acc = measure_accuracy_from_theta(model, theta_recon, cfg, device)
        var_k = (coeffs.norm() ** 2 / (delta.norm() ** 2)).item()
        results.append({
            "condition": f"Random rank-{k}",
            "k": k,
            "var_captured": var_k,
            **acc,
        })
        print(f"    {'Random rank-' + str(k):>25s}: add={acc['test_add']:.3f}, mul={acc['test_mul']:.3f}  (var={var_k:.1%})")

    return {
        "seed": seed,
        "grok_step_add": data["grok_step_add"],
        "grok_step_mul": data["grok_step_mul"],
        "results": results,
        "basis_shape": B.shape,
        "total_params": total_params,
    }


def plot_accuracy_vs_k(all_results):
    """figABL2_A: Test accuracy vs number of PCA components (line plot)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for task_idx, (task, task_label) in enumerate([("test_add", "Mod-Add"),
                                                     ("test_mul", "Mod-Mul")]):
        ax = axes[task_idx]

        # Collect PCA sweep data
        for r in all_results:
            seed = r["seed"]
            pca_results = [res for res in r["results"] if res["condition"].startswith("PCA")]
            ks = [res["k"] for res in pca_results]
            accs = [res[task] for res in pca_results]
            ax.plot(ks, accs, "o-", alpha=0.5, lw=1.5, markersize=5,
                    label=f"PCA s{seed}")

        # Mean PCA curve
        pca_results_all = {}
        for r in all_results:
            for res in r["results"]:
                if res["condition"].startswith("PCA"):
                    k = res["k"]
                    if k not in pca_results_all:
                        pca_results_all[k] = []
                    pca_results_all[k].append(res[task])

        ks_mean = sorted(pca_results_all.keys())
        means = [np.mean(pca_results_all[k]) for k in ks_mean]
        ax.plot(ks_mean, means, "k-", lw=3, markersize=8, marker="D",
                label="PCA mean", zorder=10)

        # Random controls
        for r in all_results:
            for res in r["results"]:
                if res["condition"].startswith("Random"):
                    k = res["k"]
                    ax.scatter(k, res[task], marker="x", color="red", s=80,
                               zorder=8, alpha=0.7)

        # Add a dummy for legend
        ax.scatter([], [], marker="x", color="red", s=80, label="Random control")

        # Full model and init
        full_accs = [res[task] for r in all_results
                     for res in r["results"] if res["condition"] == "Full model"]
        init_accs = [res[task] for r in all_results
                     for res in r["results"] if res["condition"] == "Init model"]
        ax.axhline(np.mean(full_accs), color="blue", ls="--", alpha=0.5,
                   label=f"Full model ({np.mean(full_accs):.2f})")
        ax.axhline(np.mean(init_accs), color="gray", ls=":", alpha=0.5,
                   label=f"Init model")
        ax.axhline(0.98, color="green", ls=":", alpha=0.3)

        ax.set_xlabel("Number of PCA components kept", fontsize=12)
        ax.set_ylabel(f"Test accuracy ({task_label})", fontsize=12)
        ax.set_title(task_label, fontsize=13)
        ax.set_ylim(-0.02, 1.05)
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(alpha=0.3)

    fig.suptitle("Reconstruction Ablation: How Many Eigenvectors Needed?",
                 fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / f"figABL2_A_accuracy_vs_k.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved figABL2_A_accuracy_vs_k.png")


def plot_combined_bar(all_results):
    """figABL2_B: Bar chart for key conditions."""
    key_conditions = ["Init model", "Random rank-1", "PCA top-1", "PCA top-2",
                      "PCA top-3", "PCA top-5", "PCA top-8", "PCA top-16",
                      "Random rank-16", "Full model"]

    # Collect
    add_data, mul_data = {}, {}
    for cond in key_conditions:
        adds, muls = [], []
        for r in all_results:
            for res in r["results"]:
                if res["condition"] == cond:
                    adds.append(res["test_add"])
                    muls.append(res["test_mul"])
        if adds:
            add_data[cond] = (np.mean(adds), np.std(adds))
            mul_data[cond] = (np.mean(muls), np.std(muls))

    conds = [c for c in key_conditions if c in add_data]
    x = np.arange(len(conds))
    width = 0.35

    fig, ax = plt.subplots(figsize=(15, 6))
    add_means = [add_data[c][0] for c in conds]
    add_stds = [add_data[c][1] for c in conds]
    mul_means = [mul_data[c][0] for c in conds]
    mul_stds = [mul_data[c][1] for c in conds]

    bars1 = ax.bar(x - width/2, add_means, width, yerr=add_stds, capsize=3,
                   label="Test add", color="#3498db", edgecolor="k", alpha=0.85)
    bars2 = ax.bar(x + width/2, mul_means, width, yerr=mul_stds, capsize=3,
                   label="Test mul", color="#e74c3c", edgecolor="k", alpha=0.85)

    ax.axhline(0.98, color="green", ls="--", alpha=0.4, label="Grok threshold")
    ax.axhline(1/97, color="gray", ls=":", alpha=0.3)

    # Color-code bars
    for i, c in enumerate(conds):
        if c.startswith("Random") or c == "Init model":
            bars1[i].set_facecolor("#bdc3c7")
            bars2[i].set_facecolor("#bdc3c7")
            bars1[i].set_alpha(0.6)
            bars2[i].set_alpha(0.6)

    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        if h > 0.03:
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.015,
                    f"{h:.2f}", ha="center", fontsize=7, fontweight="bold")

    ax.set_xlabel("Reconstruction condition", fontsize=12)
    ax.set_ylabel("Test accuracy (mean ± std, 3 seeds)", fontsize=12)
    ax.set_title("Eigenvector Reconstruction: Full-Model Δθ Projection\n"
                 "(How many PCA directions needed to reconstruct the grokked solution?)",
                 fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(conds, rotation=25, ha="right", fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / f"figABL2_B_key_bars.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved figABL2_B_key_bars.png")


def plot_variance_vs_accuracy(all_results):
    """figABL2_C: Variance captured vs accuracy (are they correlated?)."""
    fig, ax = plt.subplots(figsize=(8, 6))

    for r in all_results:
        seed = r["seed"]
        for res in r["results"]:
            if "var_captured" not in res:
                continue
            vc = res["var_captured"]
            avg_acc = (res["test_add"] + res["test_mul"]) / 2
            is_random = res["condition"].startswith("Random")
            color = "red" if is_random else "#3498db"
            marker = "x" if is_random else "o"
            ax.scatter(vc * 100, avg_acc, color=color, marker=marker, s=60,
                       alpha=0.7, zorder=5)
            if not is_random and res["k"] in [1, 2, 3, 8, 16]:
                ax.annotate(f"k={res['k']}", (vc*100, avg_acc),
                            textcoords="offset points", xytext=(5, 5),
                            fontsize=7)

    ax.scatter([], [], color="#3498db", marker="o", label="PCA top-k")
    ax.scatter([], [], color="red", marker="x", label="Random rank-k")
    ax.axhline(0.98, color="green", ls="--", alpha=0.3, label="Grok threshold")
    ax.set_xlabel("Variance captured (%)", fontsize=12)
    ax.set_ylabel("Mean test accuracy", fontsize=12)
    ax.set_title("Variance Captured vs Accuracy\n"
                 "(PCA captures far more accuracy per unit variance than random)",
                 fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xlim(-1, 50)
    ax.set_ylim(-0.02, 1.05)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / f"figABL2_C_variance_vs_accuracy.png", dpi=150)
    plt.close(fig)
    print(f"  saved figABL2_C_variance_vs_accuracy.png")


def main():
    PLOT_DIR.mkdir(exist_ok=True)
    device = get_device()
    print(f"Device: {device}")

    all_results = []
    for seed in SEEDS:
        r = run_sweep_for_seed(seed)
        if r is not None:
            all_results.append(r)

    if not all_results:
        print("No results.")
        sys.exit(1)

    save_path = PLOT_DIR / "eigenvector_ablation_v2_results.pt"
    torch.save(all_results, save_path)
    print(f"\nSaved to {save_path}")

    print("\n" + "=" * 70)
    print("  PLOTTING")
    print("=" * 70)
    plot_accuracy_vs_k(all_results)
    plot_combined_bar(all_results)
    plot_variance_vs_accuracy(all_results)

    # Summary table
    print(f"\n{'='*70}")
    print("  RECONSTRUCTION ABLATION SUMMARY")
    print(f"{'='*70}")

    cond_order = ["Init model", "Random rank-1", "PCA top-1", "PCA top-2",
                  "PCA top-3", "Random rank-2", "PCA top-5", "PCA top-8",
                  "PCA top-16", "Random rank-16", "Full model"]

    print(f"{'Condition':>25s}  {'Add':>12s}  {'Mul':>12s}  {'Var%':>6s}")
    print("-" * 65)
    for cond in cond_order:
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
        vc_str = f"{np.mean(vcs)*100:.1f}%" if vcs else "—"
        print(f"{cond:>25s}  {np.mean(adds):.3f}±{np.std(adds):.3f}  "
              f"{np.mean(muls):.3f}±{np.std(muls):.3f}  {vc_str:>6s}")

    print("\nDone.")


if __name__ == "__main__":
    main()
