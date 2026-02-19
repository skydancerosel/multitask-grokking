#!/usr/bin/env python3
"""
Aggregate Hessian results across all seeds and WD values.
Produce multi-seed scaling law with error bars.

Data sources:
  wd=1.0: plots/hessian_results.pt  (seeds 42, 137, 2024) — key: hessian[seed]["grok"]
  wd=0.5: plots/hessian_wd05_results.pt (seed 42) — key: hess_05
          plots/hessian_wd05_s{137,2024}.pt — key: hess
  wd=0.3: plots/hessian_wd03_s{42,137,2024}.pt — key: hess
  wd=0.1: plots/hessian_wd01_results.pt (seed 42) — key: hess_01
          plots/hessian_wd01_s{137,2024}.pt — key: hess
  wd=0.0: plots/hessian_results.pt  (seeds 42, 137, 2024) — key: hessian[seed]["control"]

  Training data (for grok steps):
  wd=1.0: results/multitask_s{seed}.pt
  wd=0.5: results/multitask_wd05_s{seed}.pt
  wd=0.1: results/multitask_wd01_s{seed}.pt
  wd=0.0: results/multitask_nowd_s{seed}.pt

Figures produced:
  figMT_H14 — Multi-seed grok timing vs WD (with error bars)
  figMT_H15 — Multi-seed λ_min vs WD scaling law (with error bars, at 50% of training)
  figMT_H16 — Multi-seed λ_min vs WD (all fracs, with shading)
  figMT_H17 — Per-task λ_min summary (add vs mul, all seeds)
  figMT_H18 — Combined: grok timing + curvature depth (dual axis)
"""

import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = Path(__file__).parent / "results"
PLOT_DIR = Path(__file__).parent / "plots"
SEEDS = [42, 137, 2024]
WDS = [0.0, 0.1, 0.2, 0.3, 0.5, 1.0]


def load_all():
    """Load all Hessian results and training metadata."""
    # wd=1.0 and wd=0.0 from the original hessian_results.pt
    orig = torch.load(PLOT_DIR / "hessian_results.pt", map_location="cpu", weights_only=False)

    hess = {}  # hess[wd][seed] = {"steps": ..., "total": ..., "add": ..., "mul": ...}
    meta = {}  # meta[wd][seed] = {"grok_step_add": ..., "grok_step_mul": ...}

    for wd in WDS:
        hess[wd] = {}
        meta[wd] = {}

    # ── wd=1.0 ──
    for seed in SEEDS:
        hess[1.0][seed] = orig["hessian"][seed]["grok"]
        d = torch.load(RESULTS_DIR / f"multitask_s{seed}.pt", map_location="cpu", weights_only=False)
        meta[1.0][seed] = {"grok_step_add": d["grok_step_add"], "grok_step_mul": d["grok_step_mul"]}

    # ── wd=0.0 ──
    for seed in SEEDS:
        hess[0.0][seed] = orig["hessian"][seed]["control"]
        meta[0.0][seed] = {"grok_step_add": None, "grok_step_mul": None}

    # ── wd=0.5 ──
    # seed 42 — from hessian_wd05_results.pt
    wd05_42 = torch.load(PLOT_DIR / "hessian_wd05_results.pt", map_location="cpu", weights_only=False)
    hess[0.5][42] = wd05_42["hess_05"]
    meta[0.5][42] = wd05_42["data_05_meta"]
    # seeds 137, 2024 — from individual files
    for seed in [137, 2024]:
        d = torch.load(PLOT_DIR / f"hessian_wd05_s{seed}.pt", map_location="cpu", weights_only=False)
        hess[0.5][seed] = d["hess"]
        meta[0.5][seed] = d["meta"]

    # ── wd=0.3 ──
    for seed in SEEDS:
        d = torch.load(PLOT_DIR / f"hessian_wd03_s{seed}.pt", map_location="cpu", weights_only=False)
        hess[0.3][seed] = d["hess"]
        meta[0.3][seed] = d["meta"]

    # ── wd=0.2 ──
    for seed in SEEDS:
        d = torch.load(PLOT_DIR / f"hessian_wd02_s{seed}.pt", map_location="cpu", weights_only=False)
        hess[0.2][seed] = d["hess"]
        meta[0.2][seed] = d["meta"]

    # ── wd=0.1 ──
    # seed 42 — from hessian_wd01_results.pt
    wd01_42 = torch.load(PLOT_DIR / "hessian_wd01_results.pt", map_location="cpu", weights_only=False)
    hess[0.1][42] = wd01_42["hess_01"]
    meta[0.1][42] = wd01_42.get("data_01_meta", {})
    # seeds 137, 2024
    for seed in [137, 2024]:
        d = torch.load(PLOT_DIR / f"hessian_wd01_s{seed}.pt", map_location="cpu", weights_only=False)
        hess[0.1][seed] = d["hess"]
        meta[0.1][seed] = d["meta"]

    return hess, meta


def get_min_eigenvalue(hess_entry, mode, frac=None):
    """Get the minimum eigenvalue across all checkpoints, or at a specific training fraction."""
    eigs = hess_entry[mode][:, 0]  # bottom eigenvalue at each checkpoint
    if frac is None:
        return np.min(eigs)
    steps = hess_entry["steps"]
    target = int(frac * steps[-1])
    ci = np.argmin(np.abs(steps - target))
    return eigs[ci]


def fig_h14_grok_timing(meta):
    """Multi-seed grokking time vs WD with error bars."""
    fig, ax = plt.subplots(figsize=(10, 6))

    grokking_wds = [wd for wd in WDS if wd > 0]  # exclude 0.0 (never groks)

    for task, color, marker in [("add", "#1f77b4", "o"), ("mul", "#d62728", "s")]:
        means, stds, wds_plot = [], [], []
        for wd in grokking_wds:
            steps = []
            for seed in SEEDS:
                gs = meta[wd][seed].get(f"grok_step_{task}")
                if gs is not None:
                    steps.append(gs)
            if steps:
                means.append(np.mean(steps))
                stds.append(np.std(steps))
                wds_plot.append(wd)

        ax.errorbar(wds_plot, means, yerr=stds, fmt=f"{marker}-", color=color,
                     lw=2.5, ms=10, capsize=6, capthick=2, label=f"{task.upper()} grok step")

    ax.set_xlabel("Weight Decay", fontsize=13)
    ax.set_ylabel("Grokking Step", fontsize=13)
    ax.set_title("Grokking Timing vs Weight Decay (3 seeds, mean ± std)", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "figMT_H14_grok_timing_multiseed.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  saved figMT_H14_grok_timing_multiseed.png")


def fig_h15_scaling_errorbars(hess):
    """λ_min vs WD at 50% training fraction, with error bars over seeds."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for idx, (mode, title) in enumerate([("total", "Total Loss"),
                                          ("add", "Add Loss"),
                                          ("mul", "Mul Loss")]):
        ax = axes[idx]
        for frac, color, marker in [(0.25, "#3498db", "o"), (0.50, "#e74c3c", "s"), (0.75, "#2ecc71", "^")]:
            means, stds, wds_plot = [], [], []
            for wd in WDS:
                vals = []
                for seed in SEEDS:
                    if seed in hess[wd]:
                        vals.append(get_min_eigenvalue(hess[wd][seed], mode, frac))
                if vals:
                    means.append(np.mean(vals))
                    stds.append(np.std(vals))
                    wds_plot.append(wd)

            ax.errorbar(wds_plot, means, yerr=stds, fmt=f"{marker}-", color=color,
                         lw=2, ms=8, capsize=5, capthick=1.5,
                         label=f"{int(frac*100)}% of training")

        ax.axhline(0, color="gray", ls="--", alpha=0.5)
        ax.set_xlabel("Weight Decay", fontsize=12)
        ax.set_ylabel("λ_min", fontsize=12)
        ax.set_title(title, fontsize=13)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    fig.suptitle("Scaling Law: λ_min vs Weight Decay (3 seeds, mean ± std)", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "figMT_H15_scaling_multiseed.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  saved figMT_H15_scaling_multiseed.png")


def fig_h16_scaling_shaded(hess):
    """λ_min vs WD with shaded bands showing seed variability."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for idx, (mode, title) in enumerate([("total", "Total Loss"),
                                          ("add", "Add Loss"),
                                          ("mul", "Mul Loss")]):
        ax = axes[idx]

        for frac, color, label in [(0.25, "#3498db", "25%"),
                                     (0.50, "#e74c3c", "50%"),
                                     (0.75, "#2ecc71", "75%")]:
            wd_vals = []
            means_arr = []
            lo_arr = []
            hi_arr = []
            for wd in WDS:
                vals = []
                for seed in SEEDS:
                    if seed in hess[wd]:
                        vals.append(get_min_eigenvalue(hess[wd][seed], mode, frac))
                if vals:
                    wd_vals.append(wd)
                    means_arr.append(np.mean(vals))
                    lo_arr.append(np.min(vals))
                    hi_arr.append(np.max(vals))

            ax.plot(wd_vals, means_arr, "o-", color=color, lw=2, ms=7, label=f"{label} mean")
            ax.fill_between(wd_vals, lo_arr, hi_arr, color=color, alpha=0.15)

        ax.axhline(0, color="gray", ls="--", alpha=0.5)
        ax.set_xlabel("Weight Decay", fontsize=12)
        ax.set_ylabel("λ_min", fontsize=12)
        ax.set_title(title, fontsize=13)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    fig.suptitle("λ_min vs WD — Shaded: min/max over 3 seeds", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "figMT_H16_scaling_shaded.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  saved figMT_H16_scaling_shaded.png")


def fig_h17_pertask_summary(hess):
    """Min eigenvalue (over entire training) for add vs mul across WD, per seed."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for idx, (mode, title, color) in enumerate([("add", "Add Task", "#1f77b4"),
                                                   ("mul", "Mul Task", "#d62728")]):
        ax = axes[idx]
        for si, (seed, marker) in enumerate(zip(SEEDS, ["o", "s", "^"])):
            vals = []
            wds_plot = []
            for wd in WDS:
                if seed in hess[wd]:
                    vals.append(get_min_eigenvalue(hess[wd][seed], mode))
                    wds_plot.append(wd)
            ax.plot(wds_plot, vals, f"{marker}-", color=color, alpha=0.5 + 0.2*si,
                     lw=1.5, ms=8, label=f"seed={seed}")

        # Mean
        means, stds_arr = [], []
        for wd in WDS:
            v = [get_min_eigenvalue(hess[wd][s], mode) for s in SEEDS if s in hess[wd]]
            means.append(np.mean(v))
            stds_arr.append(np.std(v))
        ax.errorbar(WDS, means, yerr=stds_arr, fmt="D-", color="black", lw=2.5, ms=10,
                     capsize=6, capthick=2, label="Mean ± std", zorder=10)

        ax.axhline(0, color="gray", ls="--", alpha=0.5)
        ax.set_xlabel("Weight Decay", fontsize=12)
        ax.set_ylabel("min λ_min (over training)", fontsize=12)
        ax.set_title(f"{title}: Deepest Curvature vs WD", fontsize=13)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(PLOT_DIR / "figMT_H17_pertask_summary.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  saved figMT_H17_pertask_summary.png")


def fig_h18_combined(hess, meta):
    """Dual-axis: grok timing (left) + deepest curvature (right) vs WD."""
    fig, ax1 = plt.subplots(figsize=(12, 7))
    ax2 = ax1.twinx()

    grokking_wds = [wd for wd in WDS if wd > 0]

    # Grok timing — left axis
    for task, color, marker in [("add", "#1f77b4", "o"), ("mul", "#d62728", "s")]:
        means, stds_arr = [], []
        for wd in grokking_wds:
            steps = [meta[wd][s].get(f"grok_step_{task}") for s in SEEDS
                     if meta[wd][s].get(f"grok_step_{task}") is not None]
            means.append(np.mean(steps) if steps else np.nan)
            stds_arr.append(np.std(steps) if steps else 0)
        ax1.errorbar(grokking_wds, means, yerr=stds_arr, fmt=f"{marker}-", color=color,
                     lw=2.5, ms=10, capsize=6, capthick=2, label=f"Grok step ({task})")

    ax1.set_xlabel("Weight Decay", fontsize=13)
    ax1.set_ylabel("Grokking Step", fontsize=13, color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax1.set_yscale("log")

    # Deepest curvature — right axis (total loss)
    means_c, stds_c = [], []
    for wd in WDS:
        vals = [get_min_eigenvalue(hess[wd][s], "total") for s in SEEDS if s in hess[wd]]
        means_c.append(np.mean(vals))
        stds_c.append(np.std(vals))
    ax2.errorbar(WDS, means_c, yerr=stds_c, fmt="D--", color="#2ca02c",
                 lw=2.5, ms=10, capsize=6, capthick=2, label="Deepest λ_min (total)")
    ax2.set_ylabel("Deepest λ_min (total loss)", fontsize=13, color="#2ca02c")
    ax2.tick_params(axis="y", labelcolor="#2ca02c")
    ax2.axhline(0, color="gray", ls=":", alpha=0.3)

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=10, loc="center left")

    ax1.set_title("Grokking Timing + Curvature Depth vs Weight Decay\n(3 seeds, mean ± std)",
                   fontsize=14)
    ax1.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "figMT_H18_combined_dual.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  saved figMT_H18_combined_dual.png")


def print_summary(hess, meta):
    """Print the complete multi-seed summary table."""
    print(f"\n{'='*90}")
    print("  MULTI-SEED HESSIAN SUMMARY (seeds = 42, 137, 2024)")
    print(f"{'='*90}")

    print(f"\n  {'WD':>5s} | {'Seed':>5s} | {'Grok ADD':>10s} | {'Grok MUL':>10s} | "
          f"{'λ_min(tot)':>12s} | {'λ_min(add)':>12s} | {'λ_min(mul)':>12s}")
    print(f"  {'-'*5}-+-{'-'*5}-+-{'-'*10}-+-{'-'*10}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}")

    for wd in sorted(WDS, reverse=True):
        for seed in SEEDS:
            if seed not in hess[wd]:
                continue
            m = meta[wd][seed]
            ga = str(m.get("grok_step_add") or "—")
            gm = str(m.get("grok_step_mul") or "—")
            lt = get_min_eigenvalue(hess[wd][seed], "total")
            la = get_min_eigenvalue(hess[wd][seed], "add")
            lm = get_min_eigenvalue(hess[wd][seed], "mul")
            print(f"  {wd:5.1f} | {seed:5d} | {ga:>10s} | {gm:>10s} | "
                  f"{lt:12.4f} | {la:12.4f} | {lm:12.4f}")
        # Mean row
        ga_list = [meta[wd][s].get("grok_step_add") for s in SEEDS
                   if s in meta[wd] and meta[wd][s].get("grok_step_add") is not None]
        gm_list = [meta[wd][s].get("grok_step_mul") for s in SEEDS
                   if s in meta[wd] and meta[wd][s].get("grok_step_mul") is not None]
        lt_list = [get_min_eigenvalue(hess[wd][s], "total") for s in SEEDS if s in hess[wd]]
        la_list = [get_min_eigenvalue(hess[wd][s], "add") for s in SEEDS if s in hess[wd]]
        lm_list = [get_min_eigenvalue(hess[wd][s], "mul") for s in SEEDS if s in hess[wd]]

        ga_str = f"{np.mean(ga_list):.0f}±{np.std(ga_list):.0f}" if ga_list else "—"
        gm_str = f"{np.mean(gm_list):.0f}±{np.std(gm_list):.0f}" if gm_list else "—"
        lt_str = f"{np.mean(lt_list):.2f}±{np.std(lt_list):.2f}"
        la_str = f"{np.mean(la_list):.2f}±{np.std(la_list):.2f}"
        lm_str = f"{np.mean(lm_list):.2f}±{np.std(lm_list):.2f}"
        print(f"  {'MEAN':>5s} | {'':>5s} | {ga_str:>10s} | {gm_str:>10s} | "
              f"{lt_str:>12s} | {la_str:>12s} | {lm_str:>12s}")
        print(f"  {'-'*5}-+-{'-'*5}-+-{'-'*10}-+-{'-'*10}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}")


def main():
    PLOT_DIR.mkdir(exist_ok=True)
    print("Loading all Hessian results...")
    hess, meta = load_all()
    print(f"  Loaded {sum(len(hess[wd]) for wd in WDS)} (WD, seed) combinations\n")

    print_summary(hess, meta)

    print(f"\n  Generating multi-seed figures...")
    fig_h14_grok_timing(meta)
    fig_h15_scaling_errorbars(hess)
    fig_h16_scaling_shaded(hess)
    fig_h17_pertask_summary(hess)
    fig_h18_combined(hess, meta)

    print("\nDone — 5 multi-seed figures saved.")


if __name__ == "__main__":
    main()
