#!/usr/bin/env python3
"""
PCA / eigenspace analysis for tri-task grokking (add + mul + x²+y²).

Loads results from train_tritask.py and produces:
  figTT_A  — Per-task accuracy curves (train/test for add, mul, sq)
  figTT_B  — Loss curves (add, mul, sq, total)
  figTT_C  — PC1% over training (expanding window) per weight matrix
  figTT_D  — Eigenspectrum (top-10) per weight matrix
  figTT_E  — SVD of weight deltas over training (top-5 singular values)
  figTT_F  — Head alignment: pairwise cosine similarity (add↔mul, add↔sq, mul↔sq)
  figTT_G  — PC1% heatmap (layer × weight matrix)
  figTT_H  — Comparison: grok vs no-WD eigenspectra
  figTT_I  — Cross-seed summary table / bar chart
"""

import sys
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
SEEDS = [42, 137, 2024]
TASK_NAMES = ["add", "mul", "sq"]
TASK_COLORS = {"add": "#1f77b4", "mul": "#d62728", "sq": "#2ca02c"}


def load_result(tag):
    path = RESULTS_DIR / f"{tag}.pt"
    if not path.exists():
        print(f"ERROR: {path} not found. Run train_tritask.py first.")
        sys.exit(1)
    return torch.load(path, map_location="cpu", weights_only=False)


# ═══════════════════════════════════════════════════════════════════════════
# Figure TT_A: Per-task accuracy curves
# ═══════════════════════════════════════════════════════════════════════════

def fig_a_accuracy(data, tag):
    metrics = data["metrics"]
    steps = [m["step"] for m in metrics]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

    for t in TASK_NAMES:
        c = TASK_COLORS[t]
        ax1.plot(steps, [m[f"train_{t}"] for m in metrics], label=f"train {t}", color=c, lw=2)
        ax1.plot(steps, [m[f"test_{t}"] for m in metrics], label=f"test {t}", color=c, ls="--", lw=2)
    ax1.axhline(0.98, color="gray", ls=":", alpha=0.5, label="grok threshold")

    grok_step = data.get("grok_step", {})
    for t in TASK_NAMES:
        gs = grok_step.get(t)
        if gs:
            ax1.axvline(gs, color=TASK_COLORS[t], ls=":", alpha=0.7,
                         label=f"{t} groks @{gs}")

    ax1.set_xlabel("Training step"); ax1.set_ylabel("Accuracy")
    ax1.set_title(f"Tri-Task Accuracy ({tag})")
    ax1.legend(fontsize=7, ncol=3); ax1.grid(alpha=0.3)

    # Loss
    for t in TASK_NAMES:
        c = TASK_COLORS[t]
        ax2.plot(steps, [m[f"loss_{t}"] for m in metrics], label=f"loss_{t}", color=c, lw=1.5)
    ax2.plot(steps, [m["loss"] for m in metrics], label="total", color="black", lw=2)
    ax2.set_xlabel("Training step"); ax2.set_ylabel("Loss")
    ax2.set_title(f"Loss ({tag})"); ax2.set_yscale("log")
    ax2.legend(fontsize=8); ax2.grid(alpha=0.3)

    fig.tight_layout()
    fname = f"figTT_A_accuracy_{tag}.png"
    fig.savefig(PLOT_DIR / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {fname}")


# ═══════════════════════════════════════════════════════════════════════════
# Figure TT_C: PC1% expanding window
# ═══════════════════════════════════════════════════════════════════════════

def fig_c_expanding_pc1(data, tag):
    attn_logs = data["attn_logs"]
    n_layers = len(attn_logs[0]["layers"])
    grok_step = data.get("grok_step", {})

    fig, axes = plt.subplots(n_layers, len(WEIGHT_KEYS), figsize=(20, 5*n_layers))
    if n_layers == 1:
        axes = [axes]

    for li in range(n_layers):
        for ki, key in enumerate(WEIGHT_KEYS):
            ax = axes[li][ki]
            steps, mats = collect_trajectory(attn_logs, li, key)
            if len(mats) < 5:
                ax.text(0.5, 0.5, "Too few snapshots", ha="center", va="center",
                         transform=ax.transAxes)
                continue

            records = expanding_window_pca(mats, TOP_K)
            ew_snaps = [r["n_snaps"] for r in records]
            pc1_pcts = [r["pc1_pct"] for r in records]
            # Convert n_snaps to approximate steps
            step_per_snap = steps[-1] / len(steps) if len(steps) > 1 else 1
            ew_steps_approx = [n * step_per_snap for n in ew_snaps]
            ax.plot(ew_steps_approx, pc1_pcts, "b-", lw=2)
            ax.set_ylabel("PC1 %")
            ax.set_title(f"L{li} {key}")
            ax.set_ylim(0, 100)
            ax.grid(alpha=0.3)

            # Mark grok steps
            for t in TASK_NAMES:
                gs = grok_step.get(t)
                if gs:
                    ax.axvline(gs, color=TASK_COLORS[t], ls=":", alpha=0.5)

    fig.suptitle(f"PC1% Expanding Window — {tag}", fontsize=14, y=1.01)
    fig.tight_layout()
    fname = f"figTT_C_pc1_expanding_{tag}.png"
    fig.savefig(PLOT_DIR / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {fname}")


# ═══════════════════════════════════════════════════════════════════════════
# Figure TT_D: Eigenspectrum
# ═══════════════════════════════════════════════════════════════════════════

def fig_d_eigenspectrum(data, tag):
    attn_logs = data["attn_logs"]
    n_layers = len(attn_logs[0]["layers"])

    fig, axes = plt.subplots(n_layers, len(WEIGHT_KEYS), figsize=(20, 5*n_layers))
    if n_layers == 1:
        axes = [axes]

    pc1_summary = {}

    for li in range(n_layers):
        for ki, key in enumerate(WEIGHT_KEYS):
            ax = axes[li][ki]
            _, mats = collect_trajectory(attn_logs, li, key)
            if len(mats) < 3:
                ax.text(0.5, 0.5, "Too few snapshots", ha="center", va="center",
                         transform=ax.transAxes)
                continue

            result = pca_on_trajectory(mats, TOP_K)
            eigs = result["eigenvalues"][:TOP_K]
            var_pct = eigs / eigs.sum() * 100

            ax.bar(range(len(var_pct)), var_pct, color="#3498db", edgecolor="black")
            ax.set_xlabel("PC index")
            ax.set_ylabel("Variance %")
            ax.set_title(f"L{li} {key} — PC1={var_pct[0]:.1f}%")
            ax.grid(alpha=0.3, axis="y")

            pc1_summary[f"L{li}_{key}"] = var_pct[0]

    fig.suptitle(f"Eigenspectrum (top-{TOP_K}) — {tag}", fontsize=14, y=1.01)
    fig.tight_layout()
    fname = f"figTT_D_eigenspectrum_{tag}.png"
    fig.savefig(PLOT_DIR / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {fname}")
    return pc1_summary


# ═══════════════════════════════════════════════════════════════════════════
# Figure TT_E: SVD of weight deltas
# ═══════════════════════════════════════════════════════════════════════════

def fig_e_svd_deltas(data, tag):
    svd_logs = data.get("svd_logs", [])
    if not svd_logs:
        print(f"  No SVD logs for {tag}, skipping figTT_E")
        return

    steps = [s["step"] for s in svd_logs]
    fig, axes = plt.subplots(2, 4, figsize=(20, 8))

    # Attention weight SVDs (first layer)
    for ki, key in enumerate(WEIGHT_KEYS):
        ax = axes[0][ki]
        for rank in range(5):
            vals = [s["svd"][0].get(key, [0]*5)[rank] for s in svd_logs]
            ax.plot(steps, vals, lw=1.5, label=f"σ{rank+1}")
        ax.set_title(f"L0 {key}")
        ax.set_xlabel("Step"); ax.set_ylabel("Singular value")
        ax.legend(fontsize=7); ax.grid(alpha=0.3)

    # Head SVDs
    for hi, head in enumerate(["head_add", "head_mul", "head_sq"]):
        ax = axes[1][hi]
        for rank in range(5):
            vals = [s["svd"][0].get(head, [0]*5)[rank] for s in svd_logs]
            ax.plot(steps, vals, lw=1.5, label=f"σ{rank+1}")
        ax.set_title(f"{head}")
        ax.set_xlabel("Step"); ax.set_ylabel("Singular value")
        ax.legend(fontsize=7); ax.grid(alpha=0.3)
    axes[1][3].set_visible(False)

    fig.suptitle(f"SVD of Weight Deltas — {tag}", fontsize=14, y=1.01)
    fig.tight_layout()
    fname = f"figTT_E_svd_{tag}.png"
    fig.savefig(PLOT_DIR / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {fname}")


# ═══════════════════════════════════════════════════════════════════════════
# Figure TT_F: Head alignment (pairwise cosine)
# ═══════════════════════════════════════════════════════════════════════════

def fig_f_head_alignment(data, tag):
    checkpoints = data["checkpoints"]
    steps_ckpt = [c[0] for c in checkpoints]

    pairs_heads = [("head_add", "head_mul"), ("head_add", "head_sq"), ("head_mul", "head_sq")]
    pair_labels = ["add↔mul", "add↔sq", "mul↔sq"]
    pair_colors = ["#9467bd", "#ff7f0e", "#8c564b"]

    cos_traces = {label: [] for label in pair_labels}

    for step, sd in checkpoints:
        for (h1, h2), label in zip(pairs_heads, pair_labels):
            w1 = sd[f"{h1}.weight"].float().reshape(-1)
            w2 = sd[f"{h2}.weight"].float().reshape(-1)
            cos = (w1 @ w2) / (w1.norm() * w2.norm() + 1e-12)
            cos_traces[label].append(cos.item())

    fig, ax = plt.subplots(figsize=(12, 6))
    for label, color in zip(pair_labels, pair_colors):
        ax.plot(steps_ckpt, cos_traces[label], lw=2, color=color, label=label)

    grok_step = data.get("grok_step", {})
    for t in TASK_NAMES:
        gs = grok_step.get(t)
        if gs:
            ax.axvline(gs, color=TASK_COLORS[t], ls=":", alpha=0.5, label=f"{t} groks")

    ax.axhline(0, color="gray", ls="--", alpha=0.5)
    ax.set_xlabel("Training step", fontsize=12)
    ax.set_ylabel("Cosine similarity", fontsize=12)
    ax.set_title(f"Head Weight Alignment — {tag}", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fname = f"figTT_F_head_alignment_{tag}.png"
    fig.savefig(PLOT_DIR / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {fname}")


# ═══════════════════════════════════════════════════════════════════════════
# Figure TT_G: PC1% heatmap
# ═══════════════════════════════════════════════════════════════════════════

def fig_g_heatmap(pc1_dict, tag):
    n_layers = max(int(k.split("_")[0][1:]) for k in pc1_dict.keys()) + 1
    mat = np.zeros((n_layers, len(WEIGHT_KEYS)))
    for li in range(n_layers):
        for ki, key in enumerate(WEIGHT_KEYS):
            mat[li, ki] = pc1_dict.get(f"L{li}_{key}", 0)

    fig, ax = plt.subplots(figsize=(8, 3 + n_layers))
    im = ax.imshow(mat, cmap="YlOrRd", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(len(WEIGHT_KEYS)))
    ax.set_xticklabels(WEIGHT_KEYS)
    ax.set_yticks(range(n_layers))
    ax.set_yticklabels([f"Layer {i}" for i in range(n_layers)])
    for li in range(n_layers):
        for ki in range(len(WEIGHT_KEYS)):
            ax.text(ki, li, f"{mat[li,ki]:.1f}%", ha="center", va="center", fontsize=11)
    plt.colorbar(im, ax=ax, label="PC1 %")
    ax.set_title(f"PC1% Heatmap — {tag}", fontsize=14)
    fig.tight_layout()
    fname = f"figTT_G_heatmap_{tag}.png"
    fig.savefig(PLOT_DIR / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {fname}")


# ═══════════════════════════════════════════════════════════════════════════
# Figure TT_H: Grok vs no-WD eigenspectrum comparison
# ═══════════════════════════════════════════════════════════════════════════

def fig_h_comparison(data_grok, data_ctrl, seed):
    attn_grok = data_grok["attn_logs"]
    attn_ctrl = data_ctrl["attn_logs"]
    n_layers = len(attn_grok[0]["layers"])

    fig, axes = plt.subplots(n_layers, len(WEIGHT_KEYS), figsize=(20, 5*n_layers))
    if n_layers == 1:
        axes = [axes]

    for li in range(n_layers):
        for ki, key in enumerate(WEIGHT_KEYS):
            ax = axes[li][ki]
            for attn, label, color, alpha in [
                (attn_grok, "WD=1.0", "#2ca02c", 0.8),
                (attn_ctrl, "WD=0.0", "#d62728", 0.6),
            ]:
                _, mats = collect_trajectory(attn, li, key)
                if len(mats) < 3:
                    continue
                result = pca_on_trajectory(mats, TOP_K)
                eigs = result["eigenvalues"][:TOP_K]
                var_pct = eigs / eigs.sum() * 100
                ax.bar(np.arange(TOP_K) + (0.2 if label == "WD=0.0" else -0.2),
                       var_pct, width=0.35, color=color, alpha=alpha, label=label,
                       edgecolor="black", linewidth=0.5)

            ax.set_xlabel("PC index")
            ax.set_ylabel("Variance %")
            ax.set_title(f"L{li} {key}")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3, axis="y")

    fig.suptitle(f"Grok vs No-WD Eigenspectrum — seed={seed}", fontsize=14, y=1.01)
    fig.tight_layout()
    fname = f"figTT_H_comparison_s{seed}.png"
    fig.savefig(PLOT_DIR / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {fname}")


# ═══════════════════════════════════════════════════════════════════════════
# Figure TT_I: Cross-seed summary
# ═══════════════════════════════════════════════════════════════════════════

def fig_i_summary(all_pc1, all_grok):
    """Bar chart: PC1% by seed, plus grok timing table."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left: PC1% across seeds for each weight matrix
    ax = axes[0]
    keys = sorted(set(k for d in all_pc1.values() for k in d.keys()))
    x = np.arange(len(keys))
    width = 0.25
    for si, seed in enumerate(SEEDS):
        vals = [all_pc1.get(seed, {}).get(k, 0) for k in keys]
        ax.bar(x + si * width, vals, width, label=f"seed={seed}",
               edgecolor="black", linewidth=0.5)
    ax.set_xticks(x + width)
    ax.set_xticklabels(keys, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("PC1 %")
    ax.set_title("PC1% by Weight Matrix (WD=1.0)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis="y")

    # Right: Grok timing table
    ax = axes[1]
    ax.axis("off")
    cell_text = []
    for seed in SEEDS:
        gs = all_grok.get(seed, {})
        row = [str(seed)]
        for t in TASK_NAMES:
            v = gs.get(t)
            row.append(str(v) if v else "—")
        cell_text.append(row)
    # Mean row
    mean_row = ["Mean"]
    for t in TASK_NAMES:
        vals = [all_grok[s][t] for s in SEEDS if all_grok.get(s, {}).get(t) is not None]
        if vals:
            mean_row.append(f"{np.mean(vals):.0f}±{np.std(vals):.0f}")
        else:
            mean_row.append("—")
    cell_text.append(mean_row)

    table = ax.table(cellText=cell_text,
                     colLabels=["Seed", "ADD", "MUL", "SQ"],
                     loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.8)
    ax.set_title("Grokking Steps (WD=1.0)", fontsize=13, pad=20)

    fig.suptitle("Tri-Task Cross-Seed Summary", fontsize=14, y=1.02)
    fig.tight_layout()
    fname = "figTT_I_summary.png"
    fig.savefig(PLOT_DIR / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {fname}")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    PLOT_DIR.mkdir(exist_ok=True)

    all_pc1 = {}
    all_grok = {}

    for seed in SEEDS:
        # ── WD=1.0 (grokking) ──
        tag_grok = f"tritask_wd1_s{seed}"
        print(f"\n{'='*70}\n  Analyzing {tag_grok}\n{'='*70}")
        data_grok = load_result(tag_grok)

        fig_a_accuracy(data_grok, tag_grok)
        fig_c_expanding_pc1(data_grok, tag_grok)
        pc1_dict = fig_d_eigenspectrum(data_grok, tag_grok)
        fig_e_svd_deltas(data_grok, tag_grok)
        fig_f_head_alignment(data_grok, tag_grok)
        fig_g_heatmap(pc1_dict, tag_grok)

        all_pc1[seed] = pc1_dict
        all_grok[seed] = data_grok.get("grok_step", {})

        # ── WD=0.0 (control) ──
        tag_ctrl = f"tritask_wd0_s{seed}"
        ctrl_path = RESULTS_DIR / f"{tag_ctrl}.pt"
        if ctrl_path.exists():
            print(f"\n  Analyzing control: {tag_ctrl}")
            data_ctrl = load_result(tag_ctrl)
            fig_a_accuracy(data_ctrl, tag_ctrl)
            fig_h_comparison(data_grok, data_ctrl, seed)
        else:
            print(f"  Control {tag_ctrl} not found, skipping comparison")

    # ── Cross-seed summary ──
    print(f"\n{'='*70}\n  Cross-seed summary\n{'='*70}")
    fig_i_summary(all_pc1, all_grok)

    # Print summary table
    print(f"\n  {'Seed':>6s}  ", end="")
    for k in sorted(all_pc1.get(SEEDS[0], {}).keys()):
        print(f"{k:>10s}  ", end="")
    print()
    for seed in SEEDS:
        print(f"  {seed:>6d}  ", end="")
        for k in sorted(all_pc1.get(seed, {}).keys()):
            print(f"{all_pc1[seed].get(k, 0):>9.1f}%  ", end="")
        print()

    print(f"\n  Grokking steps (WD=1.0):")
    for seed in SEEDS:
        gs = all_grok.get(seed, {})
        print(f"    seed={seed}: add={gs.get('add')}, mul={gs.get('mul')}, sq={gs.get('sq')}")

    print(f"\nDone — {len(SEEDS)*7 + 1} figures saved.")


if __name__ == "__main__":
    main()
