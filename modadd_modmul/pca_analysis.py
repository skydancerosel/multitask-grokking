#!/usr/bin/env python3
"""
PCA / subspace analysis for multi-task grokking (add + mul jointly).

Loads results from train_multitask.py and produces:
  figMT_A — Per-task accuracy curves (train/test for add & mul)
  figMT_B — Loss curves (add, mul, total)
  figMT_C — PC1% over training (expanding window) per weight matrix
  figMT_D — Eigenspectrum (top-10) per weight matrix
  figMT_E — SVD of weight deltas over training (top-5 singular values)
  figMT_F — Head alignment: cosine similarity between head_add and head_mul
  figMT_G — PC1% heatmap (layer × weight matrix)
  figMT_H — Integrability ratio: resid/full over training
"""

import sys, json
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── imports from parent ────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))
from pca_sweep_analysis import pca_on_trajectory, collect_trajectory, expanding_window_pca

RESULTS_DIR = Path(__file__).parent / "results"
PLOT_DIR = Path(__file__).parent / "plots"
WEIGHT_KEYS = ["WQ", "WK", "WV", "WO"]
TOP_K = 10


def load_result(seed=42):
    path = RESULTS_DIR / f"multitask_s{seed}.pt"
    if not path.exists():
        print(f"ERROR: {path} not found. Run train_multitask.py first.")
        sys.exit(1)
    return torch.load(path, map_location="cpu", weights_only=False)


# ═══════════════════════════════════════════════════════════════════════════
# Figure MT_A: Per-task accuracy curves
# ═══════════════════════════════════════════════════════════════════════════

def fig_a_accuracy(data, seed):
    metrics = data["metrics"]
    steps = [m["step"] for m in metrics]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(steps, [m["train_add"] for m in metrics], label="train add", color="#1f77b4", lw=2)
    ax1.plot(steps, [m["test_add"] for m in metrics], label="test add", color="#1f77b4", ls="--", lw=2)
    ax1.plot(steps, [m["train_mul"] for m in metrics], label="train mul", color="#d62728", lw=2)
    ax1.plot(steps, [m["test_mul"] for m in metrics], label="test mul", color="#d62728", ls="--", lw=2)
    ax1.axhline(0.98, color="gray", ls=":", alpha=0.5, label="grok threshold")
    if data["grok_step_add"]:
        ax1.axvline(data["grok_step_add"], color="#1f77b4", ls=":", alpha=0.7, label=f"add groks @{data['grok_step_add']}")
    if data["grok_step_mul"]:
        ax1.axvline(data["grok_step_mul"], color="#d62728", ls=":", alpha=0.7, label=f"mul groks @{data['grok_step_mul']}")
    ax1.set_xlabel("Training step")
    ax1.set_ylabel("Accuracy")
    ax1.set_title(f"Multi-Task Grokking: Accuracy (seed={seed})")
    ax1.legend(fontsize=8, ncol=2)
    ax1.grid(alpha=0.3)

    ax2.plot(steps, [m["loss"] for m in metrics], label="total loss", color="black", lw=2)
    ax2.plot(steps, [m["loss_add"] for m in metrics], label="loss (add)", color="#1f77b4", lw=1.5, ls="--")
    ax2.plot(steps, [m["loss_mul"] for m in metrics], label="loss (mul)", color="#d62728", lw=1.5, ls="--")
    ax2.set_xlabel("Training step")
    ax2.set_ylabel("Cross-entropy loss")
    ax2.set_title(f"Multi-Task Loss (seed={seed})")
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)
    ax2.set_yscale("log")

    fig.tight_layout()
    fig.savefig(PLOT_DIR / f"figMT_A_accuracy_s{seed}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved figMT_A_accuracy_s{seed}.png")


# ═══════════════════════════════════════════════════════════════════════════
# Figure MT_C: Expanding-window PC1% per weight matrix
# ═══════════════════════════════════════════════════════════════════════════

def fig_c_expanding_pc1(data, seed):
    attn_logs = data["attn_logs"]
    n_layers = data["cfg"]["N_LAYERS"]

    fig, axes = plt.subplots(n_layers, len(WEIGHT_KEYS), figsize=(5*len(WEIGHT_KEYS), 4*n_layers))
    if n_layers == 1:
        axes = axes.reshape(1, -1)

    for li in range(n_layers):
        for wi, wkey in enumerate(WEIGHT_KEYS):
            ax = axes[li, wi]
            _, mats = collect_trajectory(attn_logs, li, wkey)
            recs = expanding_window_pca(mats, TOP_K, n_checkpoints=20)
            if not recs:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
                continue

            fracs = [r["n_snaps"] / len(mats) * 100 for r in recs]
            pc1s = [r["pc1_pct"] for r in recs]
            top3 = [r["top3_pct"] for r in recs]

            ax.plot(fracs, pc1s, "o-", label="PC1", color="#2ca02c", lw=2, ms=3)
            ax.plot(fracs, top3, "s--", label="Top-3", color="#ff7f0e", lw=1.5, ms=2)

            # Mark grok steps
            total_steps = attn_logs[-1]["step"]
            if data["grok_step_add"] and total_steps > 0:
                frac_add = data["grok_step_add"] / total_steps * 100
                ax.axvline(frac_add, color="#1f77b4", ls=":", alpha=0.6, label="add groks")
            if data["grok_step_mul"] and total_steps > 0:
                frac_mul = data["grok_step_mul"] / total_steps * 100
                ax.axvline(frac_mul, color="#d62728", ls=":", alpha=0.6, label="mul groks")

            ax.set_ylim(0, 100)
            ax.set_xlabel("% trajectory")
            ax.set_ylabel("Var. explained (%)")
            ax.set_title(f"L{li} {wkey}")
            ax.legend(fontsize=6)
            ax.grid(alpha=0.3)

    fig.suptitle(f"Expanding-Window PCA: PC1% Evolution (seed={seed})", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / f"figMT_C_expanding_pc1_s{seed}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved figMT_C_expanding_pc1_s{seed}.png")


# ═══════════════════════════════════════════════════════════════════════════
# Figure MT_D: Eigenspectrum (final)
# ═══════════════════════════════════════════════════════════════════════════

def fig_d_eigenspectrum(data, seed):
    attn_logs = data["attn_logs"]
    n_layers = data["cfg"]["N_LAYERS"]

    fig, axes = plt.subplots(n_layers, len(WEIGHT_KEYS), figsize=(5*len(WEIGHT_KEYS), 4*n_layers))
    if n_layers == 1:
        axes = axes.reshape(1, -1)

    colors = {"WQ": "#1f77b4", "WK": "#ff7f0e", "WV": "#2ca02c", "WO": "#d62728"}

    for li in range(n_layers):
        for wi, wkey in enumerate(WEIGHT_KEYS):
            ax = axes[li, wi]
            _, mats = collect_trajectory(attn_logs, li, wkey)
            pca = pca_on_trajectory(mats, TOP_K)
            if pca is None:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
                continue

            explained = pca["explained_ratio"][:TOP_K] * 100
            ax.bar(range(len(explained)), explained, color=colors[wkey], alpha=0.85)
            ax.set_xlabel("PC index")
            ax.set_ylabel("Var. explained (%)")
            ax.set_title(f"L{li} {wkey} — PC1={explained[0]:.1f}%")
            ax.grid(axis="y", alpha=0.3)

    fig.suptitle(f"Eigenspectrum (top-{TOP_K}) — Multi-Task Model (seed={seed})", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / f"figMT_D_eigenspectrum_s{seed}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved figMT_D_eigenspectrum_s{seed}.png")


# ═══════════════════════════════════════════════════════════════════════════
# Figure MT_E: SVD of weight deltas over training
# ═══════════════════════════════════════════════════════════════════════════

def fig_e_svd_deltas(data, seed):
    svd_logs = data.get("svd_logs", [])
    if not svd_logs:
        print("  [skip] no SVD logs found")
        return

    steps = [s["step"] for s in svd_logs]
    # Plot for layer 0 attention weights
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    weight_names = ["WQ", "WK", "WV", "WO", "head_add", "head_mul"]
    colors_sv = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6"]

    for idx, wname in enumerate(weight_names):
        row, col = idx // 3, idx % 3
        ax = axes[row, col]

        for si in range(5):
            vals = []
            for s in svd_logs:
                svd_data = s["svd"][0]  # layer 0
                if wname in svd_data and si < len(svd_data[wname]):
                    vals.append(svd_data[wname][si])
                else:
                    vals.append(0)
            ax.plot(steps, vals, label=f"SV{si+1}", color=colors_sv[si], lw=1.5)

        # Mark grok steps
        if data["grok_step_add"]:
            ax.axvline(data["grok_step_add"], color="#1f77b4", ls=":", alpha=0.6)
        if data["grok_step_mul"]:
            ax.axvline(data["grok_step_mul"], color="#d62728", ls=":", alpha=0.6)

        ax.set_xlabel("Step")
        ax.set_ylabel("Singular value")
        ax.set_title(f"{wname} (Layer 0)")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    fig.suptitle(f"SVD of Weight Deltas from Init (seed={seed})\n"
                 f"Blue line = add groks, Red line = mul groks", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / f"figMT_E_svd_deltas_s{seed}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved figMT_E_svd_deltas_s{seed}.png")


# ═══════════════════════════════════════════════════════════════════════════
# Figure MT_F: Head alignment (cosine similarity between two heads)
# ═══════════════════════════════════════════════════════════════════════════

def fig_f_head_alignment(data, seed):
    checkpoints = data["checkpoints"]
    steps = [s for s, _ in checkpoints]

    cos_sims = []
    for step, sd in checkpoints:
        w_add = sd["head_add.weight"].float().reshape(-1)
        w_mul = sd["head_mul.weight"].float().reshape(-1)
        cos = (w_add @ w_mul) / (w_add.norm() * w_mul.norm() + 1e-12)
        cos_sims.append(cos.item())

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(steps, cos_sims, color="#8e44ad", lw=2)
    if data["grok_step_add"]:
        ax.axvline(data["grok_step_add"], color="#1f77b4", ls=":", alpha=0.7, label=f"add groks @{data['grok_step_add']}")
    if data["grok_step_mul"]:
        ax.axvline(data["grok_step_mul"], color="#d62728", ls=":", alpha=0.7, label=f"mul groks @{data['grok_step_mul']}")
    ax.axhline(0, color="gray", ls="--", alpha=0.3)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Cosine similarity")
    ax.set_title(f"Head Alignment: cos(head_add, head_mul) (seed={seed})")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(PLOT_DIR / f"figMT_F_head_alignment_s{seed}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved figMT_F_head_alignment_s{seed}.png")


# ═══════════════════════════════════════════════════════════════════════════
# Figure MT_G: PC1% heatmap (layer × weight)
# ═══════════════════════════════════════════════════════════════════════════

def fig_g_heatmap(data, seed):
    attn_logs = data["attn_logs"]
    n_layers = data["cfg"]["N_LAYERS"]

    pc1_grid = np.zeros((n_layers, len(WEIGHT_KEYS)))
    for li in range(n_layers):
        for wi, wkey in enumerate(WEIGHT_KEYS):
            _, mats = collect_trajectory(attn_logs, li, wkey)
            pca = pca_on_trajectory(mats, TOP_K)
            if pca is not None:
                pc1_grid[li, wi] = pca["explained_ratio"][0] * 100

    fig, ax = plt.subplots(figsize=(6, 3))
    im = ax.imshow(pc1_grid, aspect="auto", cmap="YlGnBu", vmin=0, vmax=100)
    ax.set_yticks(range(n_layers))
    ax.set_yticklabels([f"Layer {i}" for i in range(n_layers)])
    ax.set_xticks(range(len(WEIGHT_KEYS)))
    ax.set_xticklabels(WEIGHT_KEYS)
    for i in range(n_layers):
        for j in range(len(WEIGHT_KEYS)):
            color = "white" if pc1_grid[i, j] > 60 else "black"
            ax.text(j, i, f"{pc1_grid[i, j]:.1f}", ha="center", va="center",
                    fontsize=11, color=color, fontweight="bold")
    fig.colorbar(im, ax=ax, pad=0.02).set_label("PC1 %")
    ax.set_title(f"PC1% Heatmap — Multi-Task Model (seed={seed})")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / f"figMT_G_heatmap_s{seed}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved figMT_G_heatmap_s{seed}.png")


# ═══════════════════════════════════════════════════════════════════════════
# Summary across seeds
# ═══════════════════════════════════════════════════════════════════════════

def summarize_all_seeds():
    seeds = [42, 137, 2024]
    results = {}
    for seed in seeds:
        path = RESULTS_DIR / f"multitask_s{seed}.pt"
        if path.exists():
            results[seed] = torch.load(path, map_location="cpu", weights_only=False)

    if not results:
        return

    print(f"\n{'='*70}")
    print("  MULTI-TASK GROKKING SUMMARY")
    print(f"{'='*70}")
    print(f"  {'seed':>5s}  {'add_grok':>10s}  {'mul_grok':>10s}  {'add_step':>10s}  {'mul_step':>10s}  {'final':>8s}")
    for seed, data in sorted(results.items()):
        print(f"  {seed:5d}  {'YES' if data['grokked_add'] else 'no':>10s}  "
              f"{'YES' if data['grokked_mul'] else 'no':>10s}  "
              f"{str(data['grok_step_add']):>10s}  {str(data['grok_step_mul']):>10s}  "
              f"{data['final_step']:8d}")

    # Cross-seed PCA summary
    n_layers = list(results.values())[0]["cfg"]["N_LAYERS"]
    print(f"\n  PC1% across seeds (final trajectory):")
    print(f"  {'Layer':>6s}  {'Weight':>6s}  ", end="")
    for seed in sorted(results.keys()):
        print(f"{'s'+str(seed):>8s}", end="  ")
    print(f"  {'mean':>8s}")

    for li in range(n_layers):
        for wkey in WEIGHT_KEYS:
            vals = []
            print(f"  {li:6d}  {wkey:>6s}  ", end="")
            for seed in sorted(results.keys()):
                _, mats = collect_trajectory(results[seed]["attn_logs"], li, wkey)
                pca = pca_on_trajectory(mats, TOP_K)
                v = pca["explained_ratio"][0] * 100 if pca else 0
                vals.append(v)
                print(f"{v:7.1f}%", end="  ")
            print(f"  {np.mean(vals):7.1f}%")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    PLOT_DIR.mkdir(exist_ok=True)

    seeds = [42, 137, 2024]
    available = [s for s in seeds if (RESULTS_DIR / f"multitask_s{s}.pt").exists()]
    if not available:
        print("No results found. Run train_multitask.py first.")
        sys.exit(1)

    print(f"Found results for seeds: {available}")

    for seed in available:
        print(f"\n{'='*70}")
        print(f"  Analyzing seed={seed}")
        print(f"{'='*70}")
        data = load_result(seed)
        fig_a_accuracy(data, seed)
        fig_c_expanding_pc1(data, seed)
        fig_d_eigenspectrum(data, seed)
        fig_e_svd_deltas(data, seed)
        fig_f_head_alignment(data, seed)
        fig_g_heatmap(data, seed)

    summarize_all_seeds()
    print(f"\nAll figures saved to {PLOT_DIR}/")
    print("Done.")


if __name__ == "__main__":
    main()
