#!/usr/bin/env python3
"""
Spectral analysis of tri-task grokking checkpoints.

Same pipeline as the single-task analysis (grok_weight_svd_gaps.py + grok_phase_portrait.py):
  1. Weight matrix SVD at each checkpoint → spectral gaps σ₁-σ₂, σ₂-σ₃
  2. Matrix commutator ||[W_Q, W_K]||_F
  3. Narrative test: all quantities normalized [0,1]
  4. Phase portrait: (σ₁-σ₂) vs ||[W_Q, W_K]||_F
  5. Grok vs control comparison

Tri-task model: shared trunk, 3 heads → (x+y)%p, (x*y)%p, (x²+y²)%p
"""

from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D

RESULTS_DIR = Path(__file__).parent / "results"
OUT_DIR = Path(__file__).parent / "spectral_plots"
OUT_DIR.mkdir(exist_ok=True)

SEEDS = [42, 137, 2024]
TASK_NAMES = ["add", "mul", "sq"]
TASK_COLORS = {"add": "#e41a1c", "mul": "#377eb8", "sq": "#4daf4a"}
SEED_COLORS = {42: "#1f77b4", 137: "#ff7f0e", 2024: "#2ca02c"}
N_SV = 5


# ═══════════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════════

def load_run(wd, seed):
    if wd == int(wd):
        tag = f"tritask_wd{int(wd)}_s{seed}"
    else:
        tag = f"tritask_wd{wd}_s{seed}"
    path = RESULTS_DIR / f"{tag}.pt"
    if not path.exists():
        print(f"  [SKIP] {path.name} not found")
        return None
    return torch.load(path, map_location="cpu", weights_only=False)


def extract_metrics(data):
    """Return (steps, per-task train/test acc dicts)."""
    m = data["metrics"]
    steps = np.array([e["step"] for e in m])
    out = {}
    for task in TASK_NAMES:
        out[task] = {
            "train": np.array([e[f"train_{task}"] for e in m]),
            "test": np.array([e[f"test_{task}"] for e in m]),
        }
    return steps, out


# ═══════════════════════════════════════════════════════════════════════════
# Core: SVD of weight matrices + commutator at each checkpoint
# ═══════════════════════════════════════════════════════════════════════════

def compute_weight_svd(data, layer_idx=0, n_sv=N_SV):
    """SVD of W_Q and W_K at each attn_log snapshot."""
    logs = data["attn_logs"]
    d_head = 32
    n_heads = 4

    steps = []
    sv_Q, sv_K = [], []
    gap_Q, gap_K = [], []
    comm_norms = []
    head_gap_Q, head_gap_K = [], []
    head_comm_norms = []

    for snap in logs:
        WQ = snap["layers"][layer_idx]["WQ"].float().numpy()
        WK = snap["layers"][layer_idx]["WK"].float().numpy()
        steps.append(snap["step"])

        SQ = np.linalg.svd(WQ, compute_uv=False)
        SK = np.linalg.svd(WK, compute_uv=False)

        k = min(n_sv, len(SQ))
        svq = np.zeros(n_sv); svq[:k] = SQ[:k]
        svk = np.zeros(n_sv); svk[:k] = SK[:k]
        sv_Q.append(svq)
        sv_K.append(svk)

        gap_Q.append([SQ[0] - SQ[1], SQ[1] - SQ[2]])
        gap_K.append([SK[0] - SK[1], SK[1] - SK[2]])

        comm = WQ @ WK - WK @ WQ
        comm_norms.append(np.linalg.norm(comm, "fro"))

        hgq, hgk, hcn = [], [], []
        for h in range(n_heads):
            s, e = h * d_head, (h + 1) * d_head
            q_block = WQ[s:e, s:e]
            k_block = WK[s:e, s:e]
            sq = np.linalg.svd(q_block, compute_uv=False)
            sk = np.linalg.svd(k_block, compute_uv=False)
            hgq.append([sq[0] - sq[1], sq[1] - sq[2]])
            hgk.append([sk[0] - sk[1], sk[1] - sk[2]])
            hcn.append(np.linalg.norm(q_block @ k_block - k_block @ q_block, "fro"))
        head_gap_Q.append(hgq)
        head_gap_K.append(hgk)
        head_comm_norms.append(hcn)

    return dict(
        steps=np.array(steps),
        sv_Q=np.array(sv_Q), sv_K=np.array(sv_K),
        gap_Q=np.array(gap_Q), gap_K=np.array(gap_K),
        comm_norms=np.array(comm_norms),
        head_gap_Q=np.array(head_gap_Q),
        head_gap_K=np.array(head_gap_K),
        head_comm_norms=np.array(head_comm_norms),
    )


# ═══════════════════════════════════════════════════════════════════════════
# Smoothing helper
# ═══════════════════════════════════════════════════════════════════════════

def smooth(x, window=3):
    if len(x) <= window:
        return x.copy()
    kernel = np.ones(window) / window
    padded = np.concatenate([[x[0]] * (window // 2), x, [x[-1]] * (window // 2)])
    return np.convolve(padded, kernel, mode="valid")[:len(x)]


def norm01(x):
    mn, mx = x.min(), x.max()
    return (x - mn) / max(mx - mn, 1e-30)


# ═══════════════════════════════════════════════════════════════════════════
# Figure 1: Master timeseries — per seed
# ═══════════════════════════════════════════════════════════════════════════

def plot_master_timeseries(all_data, save_path):
    """5-panel per seed: g₁₂, g₂₃, comm, per-task accuracy, combined accuracy."""
    n_seeds = len(all_data)
    fig, axes = plt.subplots(n_seeds, 5, figsize=(28, 4 * n_seeds), squeeze=False)

    for row, (seed, d) in enumerate(sorted(all_data.items())):
        sv = d["svd"]
        steps_sv = sv["steps"]
        steps_m, task_accs = d["metrics"]
        grok_steps = d["grok_steps"]

        # P1: σ₁-σ₂
        ax = axes[row, 0]
        ax.plot(steps_sv, sv["gap_Q"][:, 0], color="#e41a1c", lw=1.5, label="$W_Q$")
        ax.plot(steps_sv, sv["gap_K"][:, 0], color="#377eb8", lw=1, ls="--", alpha=0.6, label="$W_K$")
        for task in TASK_NAMES:
            gs = grok_steps.get(task)
            if gs:
                ax.axvline(gs, color=TASK_COLORS[task], ls=":", alpha=0.4, lw=1)
        ax.set_ylabel(f"s{seed}\n$\\sigma_1 - \\sigma_2$")
        if row == 0:
            ax.set_title("$\\sigma_1 - \\sigma_2$ (W_Q/W_K)", fontsize=10)
        ax.legend(fontsize=6)

        # P2: σ₂-σ₃
        ax = axes[row, 1]
        ax.plot(steps_sv, sv["gap_Q"][:, 1], color="#e41a1c", lw=1.5, label="$W_Q$")
        ax.plot(steps_sv, sv["gap_K"][:, 1], color="#377eb8", lw=1, ls="--", alpha=0.6, label="$W_K$")
        for task in TASK_NAMES:
            gs = grok_steps.get(task)
            if gs:
                ax.axvline(gs, color=TASK_COLORS[task], ls=":", alpha=0.4, lw=1)
        ax.set_ylabel("$\\sigma_2 - \\sigma_3$")
        if row == 0:
            ax.set_title("$\\sigma_2 - \\sigma_3$ (W_Q/W_K)", fontsize=10)
        ax.legend(fontsize=6)

        # P3: Matrix commutator
        ax = axes[row, 2]
        ax.plot(steps_sv, sv["comm_norms"], color="#9467bd", lw=2)
        for task in TASK_NAMES:
            gs = grok_steps.get(task)
            if gs:
                ax.axvline(gs, color=TASK_COLORS[task], ls=":", alpha=0.4, lw=1,
                           label=f"grok:{task}@{gs}")
        ax.set_ylabel("$\\|[W_Q, W_K]\\|_F$")
        if row == 0:
            ax.set_title("Matrix commutator", fontsize=10)
        ax.legend(fontsize=6)

        # P4: Per-task test accuracy
        ax = axes[row, 3]
        for task in TASK_NAMES:
            ax.plot(steps_m, task_accs[task]["test"], color=TASK_COLORS[task],
                    lw=1.5, label=f"test_{task}")
            ax.plot(steps_m, task_accs[task]["train"], color=TASK_COLORS[task],
                    lw=0.6, alpha=0.3)
        ax.set_ylabel("Accuracy")
        ax.set_ylim(-0.05, 1.1)
        if row == 0:
            ax.set_title("Per-task accuracy", fontsize=10)
        ax.legend(fontsize=6)

        # P5: Per-head commutator
        ax = axes[row, 4]
        head_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
        for h in range(4):
            ax.plot(steps_sv, sv["head_comm_norms"][:, h], color=head_colors[h],
                    lw=1.2, label=f"head {h}")
        for task in TASK_NAMES:
            gs = grok_steps.get(task)
            if gs:
                ax.axvline(gs, color=TASK_COLORS[task], ls=":", alpha=0.4, lw=1)
        ax.set_ylabel("Per-head $\\|[W_Q^h, W_K^h]\\|_F$")
        if row == 0:
            ax.set_title("Per-head commutator", fontsize=10)
        ax.legend(fontsize=6)

    for j in range(5):
        axes[-1, j].set_xlabel("Step")

    fig.suptitle("Tri-task: Weight SVD gaps & commutators (wd=1.0, 3 seeds)",
                 fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path.name}")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 2: Narrative test — all quantities normalized [0,1]
# ═══════════════════════════════════════════════════════════════════════════

def plot_narrative_test(all_data, save_path):
    """All quantities on [0,1], one panel per seed."""
    n = len(all_data)
    fig, axes = plt.subplots(n, 1, figsize=(16, 5 * n), squeeze=False)

    for row, (seed, d) in enumerate(sorted(all_data.items())):
        ax = axes[row, 0]
        sv = d["svd"]
        steps_sv = sv["steps"]
        steps_m, task_accs = d["metrics"]
        grok_steps = d["grok_steps"]

        g12 = sv["gap_Q"][:, 0]
        g23 = sv["gap_Q"][:, 1]
        mc = sv["comm_norms"]

        ax.plot(steps_sv, norm01(g12), color="#e41a1c", lw=2,
                label="$\\sigma_1-\\sigma_2$ ($W_Q$)")
        ax.plot(steps_sv, norm01(g23), color="#ff7f0e", lw=2,
                label="$\\sigma_2-\\sigma_3$ ($W_Q$)")
        ax.plot(steps_sv, norm01(mc), color="#9467bd", lw=2, ls="--",
                label="$\\|[W_Q,W_K]\\|_F$")

        # Average test accuracy across tasks
        common_steps = steps_m
        avg_test = np.mean([task_accs[t]["test"] for t in TASK_NAMES], axis=0)
        ax.plot(common_steps, avg_test, color="#17becf", lw=1.5, ls=":",
                label="Avg test acc")

        # Per-task grok markers
        for task in TASK_NAMES:
            gs = grok_steps.get(task)
            if gs:
                ax.axvline(gs, color=TASK_COLORS[task], ls="--", alpha=0.5, lw=1.5,
                           label=f"grok:{task}@{gs}")

        ax.set_ylabel(f"seed={seed}\nnormalized [0,1]")
        ax.set_ylim(-0.05, 1.15)
        ax.legend(fontsize=7, ncol=4, loc="upper right")
        if row == 0:
            ax.set_title("Narrative test: all quantities on common [0,1] scale",
                         fontsize=12)

    axes[-1, 0].set_xlabel("Training step")
    fig.suptitle("Tri-task: Testing spectral symmetry-breaking narrative\n"
                 "g₂₃↓ → σ₁≈σ₂ → comm peak → σ₁≫σ₂ → grok",
                 fontsize=12, y=1.01, style="italic")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path.name}")


# ═══════════════════════════════════════════════════════════════════════════
# Phase portrait helpers
# ═══════════════════════════════════════════════════════════════════════════

def make_arrow_trajectory(ax, x, y, colors, cmap, norm, lw=2.5, arrow_every=5):
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap=cmap, norm=norm, linewidths=lw,
                        capstyle="round", joinstyle="round")
    lc.set_array(colors[:-1])
    ax.add_collection(lc)

    for i in range(arrow_every, len(x) - 1, arrow_every):
        dx = x[min(i + 1, len(x) - 1)] - x[max(i - 1, 0)]
        dy = y[min(i + 1, len(y) - 1)] - y[max(i - 1, 0)]
        length = np.sqrt(dx**2 + dy**2)
        if length < 1e-10:
            continue
        scale = min(0.3 * length, 0.02)
        dx_n, dy_n = dx / length * scale, dy / length * scale
        color = cmap(norm(colors[i]))
        ax.annotate("", xy=(x[i] + dx_n, y[i] + dy_n),
                     xytext=(x[i], y[i]),
                     arrowprops=dict(arrowstyle="-|>", color=color,
                                     lw=1.8, mutation_scale=14))


def annotate_event(ax, x, y, label, marker, color, offset=(10, 10), fontsize=8):
    ax.plot(x, y, marker=marker, color=color, markersize=10, zorder=10,
            markeredgecolor="white", markeredgewidth=1.5)
    ax.annotate(label, (x, y), textcoords="offset points", xytext=offset,
                fontsize=fontsize, fontweight="bold", color=color,
                path_effects=[pe.withStroke(linewidth=3, foreground="white")],
                arrowprops=dict(arrowstyle="-", color=color, lw=0.8),
                zorder=11)


def add_phase_regions(ax, xlim, ylim, gap_thresh, comm_thresh):
    ax.axvspan(xlim[0], gap_thresh, alpha=0.045, color="#9467bd", zorder=0)
    ax.text(gap_thresh * 0.45, ylim[1] - 0.15 * (ylim[1] - ylim[0]),
            "competition\n$\\sigma_1 \\approx \\sigma_2$",
            fontsize=9, color="#7b5ea7", ha="center", va="top",
            style="italic", alpha=0.8,
            path_effects=[pe.withStroke(linewidth=2, foreground="white")])

    ax.fill_between([gap_thresh, xlim[1]], comm_thresh, ylim[1],
                    alpha=0.045, color="#d62728", zorder=0)
    ax.text(gap_thresh + 0.55 * (xlim[1] - gap_thresh),
            ylim[1] - 0.06 * (ylim[1] - ylim[0]),
            "instability", fontsize=9, color="#c44e52", ha="center", va="top",
            style="italic", alpha=0.8,
            path_effects=[pe.withStroke(linewidth=2, foreground="white")])

    ax.fill_between([gap_thresh, xlim[1]], ylim[0], comm_thresh,
                    alpha=0.045, color="#2ca02c", zorder=0)
    ax.text(gap_thresh + 0.55 * (xlim[1] - gap_thresh),
            ylim[0] + 0.06 * (ylim[1] - ylim[0]),
            "alignment", fontsize=9, color="#2e8b57", ha="center", va="bottom",
            style="italic", alpha=0.8,
            path_effects=[pe.withStroke(linewidth=2, foreground="white")])

    ax.axvline(gap_thresh, color="#666666", ls="--", lw=0.9, alpha=0.5, zorder=0)
    ax.axhline(comm_thresh,
               xmin=max(0, (gap_thresh - xlim[0]) / (xlim[1] - xlim[0])),
               xmax=1.0,
               color="#666666", ls="--", lw=0.9, alpha=0.5, zorder=0)


# ═══════════════════════════════════════════════════════════════════════════
# Figure 3: Hero phase portrait
# ═══════════════════════════════════════════════════════════════════════════

def plot_hero_portrait(all_data, save_path, hero_seed=42):
    """Hero figure: single seed, two panels (colored by step / by avg test acc)."""
    d = all_data.get(hero_seed)
    if d is None:
        print(f"  [SKIP] hero portrait — seed {hero_seed} not available")
        return

    sv = d["svd"]
    steps_sv = sv["steps"]
    steps_m, task_accs = d["metrics"]
    grok_steps = d["grok_steps"]
    SW = 3

    gaps = sv["gap_Q"][:, 0]
    comms = sv["comm_norms"]
    gaps_s = smooth(gaps, SW)
    comms_s = smooth(comms, SW)

    # Interpolate avg test accuracy to SVD step grid
    avg_test = np.mean([task_accs[t]["test"] for t in TASK_NAMES], axis=0)
    test_interp = np.interp(steps_sv, steps_m, avg_test)

    step_cadence = int(np.median(np.diff(steps_sv))) if len(steps_sv) > 1 else 100
    arrow_every = max(1, round(200 / step_cadence))

    fig, axes = plt.subplots(1, 2, figsize=(20, 9))

    for panel, (cval, clabel, cmap_name) in enumerate([
        (steps_sv, "Training step", "viridis"),
        (test_interp, "Avg test accuracy", "RdYlGn"),
    ]):
        ax = axes[panel]
        cmap = plt.get_cmap(cmap_name)
        norm = Normalize(vmin=cval.min(), vmax=cval.max())

        ax.plot(gaps, comms, color="#cccccc", alpha=0.3, lw=0.5, zorder=1)
        make_arrow_trajectory(ax, gaps_s, comms_s, cval, cmap, norm,
                              lw=3.5, arrow_every=arrow_every)

        # Events
        annotate_event(ax, gaps_s[0], comms_s[0], "init", "o", "#555555",
                       offset=(-20, -18))

        mc_peak = np.argmax(comms_s[3:]) + 3
        annotate_event(ax, gaps_s[mc_peak], comms_s[mc_peak],
                       f"comm peak\n(step {steps_sv[mc_peak]})", "D",
                       "#d62728", offset=(12, 12))

        g_min = np.argmin(gaps_s[3:]) + 3
        annotate_event(ax, gaps_s[g_min], comms_s[g_min],
                       f"σ₁≈σ₂\n(step {steps_sv[g_min]})", "v",
                       "#9467bd", offset=(-50, -22))

        # Per-task grok markers
        for task in TASK_NAMES:
            gs = grok_steps.get(task)
            if gs:
                idx = np.argmin(np.abs(steps_sv - gs))
                annotate_event(ax, gaps_s[idx], comms_s[idx],
                               f"GROK:{task}\n(step {gs})", "*",
                               TASK_COLORS[task], offset=(12, -22))

        ax.autoscale_view()
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        xpad = 0.05 * (xlim[1] - xlim[0])
        ypad = 0.05 * (ylim[1] - ylim[0])
        xlim = (xlim[0] - xpad, xlim[1] + xpad)
        ylim = (ylim[0] - ypad, ylim[1] + ypad)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        # Phase boundaries: use data-driven thresholds
        gap_range = gaps_s.max() - gaps_s.min()
        comm_range = comms_s.max() - comms_s.min()
        gap_thresh = gaps_s.min() + 0.25 * gap_range
        comm_thresh = comms_s.min() + 0.5 * comm_range
        add_phase_regions(ax, xlim, ylim, gap_thresh, comm_thresh)

        ax.set_xlabel("$\\sigma_1 - \\sigma_2$ ($W_Q$)    [spectral gap]", fontsize=13)
        ax.set_ylabel("$\\|[W_Q, W_K]\\|_F$    [non-commutativity]", fontsize=13)

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        cb = plt.colorbar(sm, ax=ax, shrink=0.75, pad=0.02)
        cb.set_label(clabel, fontsize=11)
        ax.set_title(f"colored by {clabel.lower()}", fontsize=12)

    fig.suptitle(f"Tri-task phase portrait  (seed {hero_seed}, layer 0)\n"
                 f"$x = \\sigma_1 - \\sigma_2$  vs  "
                 f"$y = \\|[W_Q, W_K]\\|_F$\n"
                 f"Tasks: add@{grok_steps.get('add','?')}, "
                 f"mul@{grok_steps.get('mul','?')}, "
                 f"sq@{grok_steps.get('sq','?')}",
                 fontsize=13, y=1.03)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path.name}")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 4: Grid — 3 seeds
# ═══════════════════════════════════════════════════════════════════════════

def plot_grid_portrait(all_data, save_path):
    """3 seeds, colored by step, with event markers."""
    SW = 3
    seeds = sorted(all_data.keys())
    fig, axes = plt.subplots(1, len(seeds), figsize=(8 * len(seeds), 8))
    if len(seeds) == 1:
        axes = [axes]

    cmap = plt.get_cmap("viridis")

    for col, seed in enumerate(seeds):
        ax = axes[col]
        d = all_data[seed]
        sv = d["svd"]
        steps_sv = sv["steps"]
        grok_steps = d["grok_steps"]

        gaps = sv["gap_Q"][:, 0]
        comms = sv["comm_norms"]
        gaps_s = smooth(gaps, SW)
        comms_s = smooth(comms, SW)
        norm = Normalize(vmin=steps_sv.min(), vmax=steps_sv.max())

        ax.plot(gaps, comms, color="gray", alpha=0.1, lw=0.4)
        make_arrow_trajectory(ax, gaps_s, comms_s, steps_sv, cmap, norm,
                              lw=2.5, arrow_every=4)

        annotate_event(ax, gaps_s[0], comms_s[0], "init", "o", "#555555",
                       offset=(-12, -12), fontsize=7)

        mc_peak = np.argmax(comms_s[3:]) + 3
        annotate_event(ax, gaps_s[mc_peak], comms_s[mc_peak], "peak", "D",
                       "#d62728", offset=(8, 8), fontsize=7)

        g_min = np.argmin(gaps_s[3:]) + 3
        annotate_event(ax, gaps_s[g_min], comms_s[g_min], "σ₁≈σ₂", "v",
                       "#9467bd", offset=(-25, -15), fontsize=7)

        for task in TASK_NAMES:
            gs = grok_steps.get(task)
            if gs:
                idx = np.argmin(np.abs(steps_sv - gs))
                annotate_event(ax, gaps_s[idx], comms_s[idx],
                               f"grok:{task}", "*",
                               TASK_COLORS[task], offset=(8, -12), fontsize=7)

        ax.set_xlabel("$\\sigma_1 - \\sigma_2$", fontsize=10)
        ax.set_ylabel("$\\|[W_Q, W_K]\\|_F$", fontsize=10)
        ax.set_title(f"seed={seed}", fontsize=11)

    fig.suptitle("Tri-task phase portraits: 3 seeds (wd=1.0, smoothed)",
                 fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=175, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path.name}")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 5: Grok (wd=1) vs control (wd=0) in same phase space
# ═══════════════════════════════════════════════════════════════════════════

def plot_grok_vs_control(all_data_grok, save_path, hero_seed=42):
    """Phase portrait: grokking (wd=1) vs memorizing (wd=0)."""
    SW = 3
    cmap = plt.get_cmap("viridis")

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    for col, (wd, label, ax) in enumerate([
        (1.0, "grokking (wd=1)", axes[0]),
        (0.0, "memorizing (wd=0)", axes[1]),
    ]):
        if wd == 1.0:
            d_raw = all_data_grok.get(hero_seed)
            if d_raw is None:
                ax.set_title(f"{label} (no data)")
                continue
            sv = d_raw["svd"]
            steps_sv = sv["steps"]
            grok_steps = d_raw["grok_steps"]
        else:
            ctrl = load_run(0.0, hero_seed)
            if ctrl is None:
                ax.set_title(f"{label} (no data)")
                continue
            sv = compute_weight_svd(ctrl)
            steps_sv = sv["steps"]
            grok_steps = ctrl.get("grok_step", {})

        gaps = sv["gap_Q"][:, 0]
        comms = sv["comm_norms"]
        gaps_s = smooth(gaps, SW)
        comms_s = smooth(comms, SW)
        norm = Normalize(vmin=steps_sv.min(), vmax=steps_sv.max())

        ax.plot(gaps, comms, color="gray", alpha=0.1, lw=0.4)
        make_arrow_trajectory(ax, gaps_s, comms_s, steps_sv, cmap, norm,
                              lw=2.5, arrow_every=max(1, len(steps_sv) // 15))

        annotate_event(ax, gaps_s[0], comms_s[0], "init", "o", "#555555",
                       offset=(-12, -12), fontsize=8)

        for task in TASK_NAMES:
            gs = grok_steps.get(task)
            if gs and gs <= steps_sv[-1]:
                idx = np.argmin(np.abs(steps_sv - gs))
                annotate_event(ax, gaps_s[idx], comms_s[idx],
                               f"grok:{task}", "*",
                               TASK_COLORS[task], offset=(8, -12), fontsize=8)

        ax.set_xlabel("$\\sigma_1 - \\sigma_2$", fontsize=11)
        ax.set_ylabel("$\\|[W_Q, W_K]\\|_F$", fontsize=11)
        ax.set_title(f"{label}  (seed={hero_seed})", fontsize=12)

    fig.suptitle("Tri-task: grokking vs memorizing in phase space\n"
                 "$\\sigma_1-\\sigma_2$ vs $\\|[W_Q,W_K]\\|_F$",
                 fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=175, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path.name}")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 6: SVD gap dynamics — grok vs control
# ═══════════════════════════════════════════════════════════════════════════

def plot_svd_grok_vs_control(all_data_grok, save_path, hero_seed=42):
    """Compare SVD gap timeseries for wd=1 vs wd=0."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    for row, (wd, label) in enumerate([(1.0, "grokking (wd=1)"),
                                         (0.0, "memorizing (wd=0)")]):
        if wd == 1.0:
            d_raw = all_data_grok.get(hero_seed)
            if d_raw is None:
                continue
            sv = d_raw["svd"]
            steps_sv = sv["steps"]
            grok_steps = d_raw["grok_steps"]
        else:
            ctrl = load_run(0.0, hero_seed)
            if ctrl is None:
                continue
            sv = compute_weight_svd(ctrl)
            steps_sv = sv["steps"]
            grok_steps = ctrl.get("grok_step", {})

        # P1: gaps
        ax = axes[row, 0]
        ax.plot(steps_sv, sv["gap_Q"][:, 0], color="#e41a1c", lw=2,
                label="$\\sigma_1-\\sigma_2$ ($W_Q$)")
        ax.plot(steps_sv, sv["gap_Q"][:, 1], color="#ff7f0e", lw=2,
                label="$\\sigma_2-\\sigma_3$ ($W_Q$)")
        for task in TASK_NAMES:
            gs = grok_steps.get(task)
            if gs and gs <= steps_sv[-1]:
                ax.axvline(gs, color=TASK_COLORS[task], ls=":", alpha=0.4)
        ax.set_ylabel(f"{label}\nSVD gap")
        ax.set_title(f"Spectral gaps — {label}", fontsize=10)
        ax.legend(fontsize=7)

        # P2: commutator
        ax = axes[row, 1]
        ax.plot(steps_sv, sv["comm_norms"], color="#9467bd", lw=2)
        for task in TASK_NAMES:
            gs = grok_steps.get(task)
            if gs and gs <= steps_sv[-1]:
                ax.axvline(gs, color=TASK_COLORS[task], ls=":", alpha=0.4,
                           label=f"grok:{task}")
        ax.set_ylabel("$\\|[W_Q,W_K]\\|_F$")
        ax.set_title(f"Matrix commutator — {label}", fontsize=10)
        ax.legend(fontsize=7)

    for j in range(2):
        axes[-1, j].set_xlabel("Step")

    fig.suptitle(f"Tri-task SVD dynamics: grokking vs control (seed={hero_seed})",
                 fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path.name}")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 7: Weight-decay sweep phase portraits
# ═══════════════════════════════════════════════════════════════════════════

def plot_wd_sweep_portraits(save_path, hero_seed=42):
    """Phase portraits across weight decay values."""
    SW = 3
    cmap = plt.get_cmap("viridis")
    wds = [0.0, 0.1, 0.2, 0.3, 0.5, 1.0]

    fig, axes = plt.subplots(2, 3, figsize=(22, 14))

    for idx, wd in enumerate(wds):
        ax = axes[idx // 3, idx % 3]
        data = load_run(wd, hero_seed)
        if data is None:
            ax.set_title(f"wd={wd} (no data)")
            continue

        sv = compute_weight_svd(data)
        steps_sv = sv["steps"]
        grok_steps = data.get("grok_step", {})

        gaps = sv["gap_Q"][:, 0]
        comms = sv["comm_norms"]
        gaps_s = smooth(gaps, SW)
        comms_s = smooth(comms, SW)
        norm = Normalize(vmin=steps_sv.min(), vmax=steps_sv.max())

        ax.plot(gaps, comms, color="gray", alpha=0.1, lw=0.4)
        make_arrow_trajectory(ax, gaps_s, comms_s, steps_sv, cmap, norm,
                              lw=2.5, arrow_every=max(1, len(steps_sv) // 15))

        annotate_event(ax, gaps_s[0], comms_s[0], "init", "o", "#555555",
                       offset=(-12, -12), fontsize=7)

        for task in TASK_NAMES:
            gs = grok_steps.get(task)
            if gs and gs <= steps_sv[-1]:
                i = np.argmin(np.abs(steps_sv - gs))
                annotate_event(ax, gaps_s[i], comms_s[i],
                               f"grok:{task}", "*",
                               TASK_COLORS[task], offset=(8, -12), fontsize=7)

        n_grokked = sum(1 for t in TASK_NAMES if grok_steps.get(t) is not None)
        ax.set_xlabel("$\\sigma_1 - \\sigma_2$", fontsize=9)
        ax.set_ylabel("$\\|[W_Q, W_K]\\|_F$", fontsize=9)
        ax.set_title(f"wd={wd}  ({n_grokked}/3 grokked)", fontsize=11)

    fig.suptitle(f"Tri-task phase portraits across weight decay (seed={hero_seed})",
                 fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=175, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path.name}")


# ═══════════════════════════════════════════════════════════════════════════
# Timing analysis
# ═══════════════════════════════════════════════════════════════════════════

def print_timing_analysis(all_data):
    print(f"\n{'='*72}")
    print("TRI-TASK TIMING ANALYSIS")
    print(f"{'='*72}")

    for seed, d in sorted(all_data.items()):
        sv = d["svd"]
        steps = sv["steps"]
        g12 = sv["gap_Q"][:, 0]
        g23 = sv["gap_Q"][:, 1]
        mc = sv["comm_norms"]
        grok_steps = d["grok_steps"]

        print(f"\n  seed={seed}:")
        print(f"    Grok steps: {grok_steps}")

        # Key events
        g23_peak_idx = np.argmax(g23[:min(15, len(g23))])
        mc_peak_idx = np.argmax(mc[3:]) + 3
        g12_min_idx = np.argmin(g12[3:]) + 3
        g12_max_idx = np.argmax(g12[3:]) + 3

        mc_peak_val = mc[mc_peak_idx]
        mid = mc[-1] + 0.5 * (mc_peak_val - mc[-1])
        mc_collapse_step = None
        for j in range(mc_peak_idx + 1, len(mc)):
            if mc[j] < mid:
                mc_collapse_step = steps[j]
                break

        events = [
            ("g₂₃ peak", steps[g23_peak_idx]),
            ("σ₁₂ min", steps[g12_min_idx]),
            ("comm peak", steps[mc_peak_idx]),
            ("σ₁₂ max", steps[g12_max_idx]),
        ]
        if mc_collapse_step:
            events.append(("comm collapse", mc_collapse_step))
        for task in TASK_NAMES:
            gs = grok_steps.get(task)
            if gs:
                events.append((f"grok:{task}", gs))

        events.sort(key=lambda x: x[1])
        print(f"    Order: {' → '.join(f'{e[0]}@{e[1]}' for e in events)}")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 72)
    print("TRI-TASK SPECTRAL ANALYSIS")
    print("=" * 72)

    # Load all wd=1 runs
    all_data = {}
    for seed in SEEDS:
        data = load_run(1.0, seed)
        if data is None:
            continue

        svd = compute_weight_svd(data)
        met = extract_metrics(data)
        grok_steps = data.get("grok_step", {})

        all_data[seed] = dict(
            svd=svd,
            metrics=met,
            grok_steps=grok_steps,
        )

        print(f"  seed={seed}: {len(svd['steps'])} snapshots, "
              f"grok={grok_steps}")

    if not all_data:
        print("No data found!")
        return

    print(f"\nGenerating figures → {OUT_DIR}/")

    plot_master_timeseries(all_data,
                           OUT_DIR / "fig1_master_timeseries.png")
    plot_narrative_test(all_data,
                        OUT_DIR / "fig2_narrative_test.png")
    plot_hero_portrait(all_data,
                       OUT_DIR / "fig3_hero_phase_portrait.png")
    plot_grid_portrait(all_data,
                       OUT_DIR / "fig4_grid_phase_portrait.png")
    plot_grok_vs_control(all_data,
                          OUT_DIR / "fig5_grok_vs_control_portrait.png")
    plot_svd_grok_vs_control(all_data,
                              OUT_DIR / "fig6_svd_grok_vs_control.png")
    plot_wd_sweep_portraits(OUT_DIR / "fig7_wd_sweep_portraits.png")

    print_timing_analysis(all_data)

    # Save computed SVD results
    save_data = {}
    for seed, d in all_data.items():
        save_data[seed] = d["svd"]
    torch.save(save_data, OUT_DIR / "spectral_results.pt")
    print(f"\nSaved: spectral_results.pt")


if __name__ == "__main__":
    main()
