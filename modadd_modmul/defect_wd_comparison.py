#!/usr/bin/env python3
"""
Defect analysis across weight-decay conditions (WD=1.0, 0.5, 0.1).

For each WD × seed:
  1. Load checkpoints
  2. Compute commutator defect at each checkpoint (total-loss mode)
  3. Detect defect onset (first sustained spike above baseline)
  4. Compare defect onset with grok time

Produces:
  figWD_A — Defect traces for all WD values (one per seed, overlay)
  figWD_B — Defect onset vs grok time scatter
  figWD_C — Lead time (grok - defect_onset) vs WD bar chart
"""

import sys, random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))
from pca_sweep_analysis import pca_on_trajectory, collect_trajectory

from train_multitask import (
    MultiTaskConfig, MultiTaskTransformer, build_dataset, sample_batch,
    get_device, extract_attn_matrices,
)
from commutator_analysis import (
    commutator_defect_median, build_pca_basis, projected_commutator,
    flatten_model_params, _write_params, attn_weight_mask,
    COMM_K, COMM_ETA, N_PCA_COMPONENTS,
)

RESULTS_DIR = Path(__file__).parent / "results"
PLOT_DIR = Path(__file__).parent / "plots"

# WD conditions: tag_prefix -> WD value
WD_CONDITIONS = {
    "multitask": 1.0,
    "multitask_wd05": 0.5,
    "multitask_wd01": 0.1,
}
SEEDS = [42, 137, 2024]
WD_COLORS = {1.0: "#1a5276", 0.5: "#e67e22", 0.1: "#27ae60"}
WD_LABELS = {1.0: "WD=1.0", 0.5: "WD=0.5", 0.1: "WD=0.1"}


def detect_defect_onset(steps, defects, baseline_frac=0.3, threshold_mult=3.0,
                        sustain_count=3):
    """
    Detect defect onset: first step where defect exceeds threshold_mult × baseline
    for at least sustain_count consecutive checkpoints.

    baseline = median defect over the first baseline_frac of training.
    """
    n_baseline = max(3, int(len(defects) * baseline_frac))
    baseline = np.median(defects[:n_baseline])
    threshold = baseline * threshold_mult

    count = 0
    for i, d in enumerate(defects):
        if d > threshold:
            count += 1
            if count >= sustain_count:
                onset_idx = i - sustain_count + 1
                return steps[onset_idx], baseline, threshold
        else:
            count = 0
    return None, baseline, threshold


def run_defect_for_condition(prefix, wd_val, seed, max_ckpts=120):
    """Compute commutator defect for one WD × seed combination."""
    tag = f"{prefix}_s{seed}"
    path = RESULTS_DIR / f"{tag}.pt"
    if not path.exists():
        print(f"  [{tag}] not found, skipping")
        return None

    print(f"\n  Loading {tag} (WD={wd_val})...")
    data = torch.load(path, map_location="cpu", weights_only=False)
    cfg_dict = data["cfg"]
    cfg = MultiTaskConfig(**cfg_dict)
    device = get_device()

    checkpoints = data["checkpoints"]
    attn_logs = data["attn_logs"]
    train_pairs, _ = build_dataset(cfg.P, cfg.TRAIN_FRACTION, cfg.SEED)

    # Subsample for speed
    if len(checkpoints) > max_ckpts:
        idx = np.linspace(0, len(checkpoints) - 1, max_ckpts, dtype=int)
        checkpoints = [checkpoints[i] for i in idx]

    print(f"  {len(checkpoints)} checkpoints, building PCA basis...")

    # Build PCA basis
    model = MultiTaskTransformer(cfg)
    B = build_pca_basis(model, attn_logs, n_components=N_PCA_COMPONENTS, device="cpu")
    if B is not None:
        print(f"  PCA basis: {B.shape}")

    # Measure defect at each checkpoint
    model = MultiTaskTransformer(cfg).to(device)
    amask = attn_weight_mask(model)

    def batch_fn():
        return sample_batch(train_pairs, cfg.BATCH_SIZE, cfg.P, device)

    results = []
    for ci, (step, sd) in enumerate(checkpoints):
        model.load_state_dict(sd)
        model.to(device)

        out = commutator_defect_median(model, batch_fn, device, K=COMM_K, eta=COMM_ETA, mode="total")

        results.append({
            "step": step,
            "defect_median": out["median"],
            "defect_p90": out["p90"],
        })

        if (ci + 1) % 20 == 0 or ci == len(checkpoints) - 1:
            print(f"    ckpt {ci+1}/{len(checkpoints)}: step={step}, defect={out['median']:.4f}")

    return {
        "tag": tag,
        "wd": wd_val,
        "seed": seed,
        "grok_step_add": data["grok_step_add"],
        "grok_step_mul": data["grok_step_mul"],
        "defect_results": results,
        "metrics": data.get("metrics", []),
    }


def plot_defect_traces(all_results):
    """figWD_A: Defect traces for each seed, all WD overlaid."""
    for seed in SEEDS:
        fig, ax = plt.subplots(figsize=(10, 5))

        for prefix, wd_val in WD_CONDITIONS.items():
            key = (prefix, seed)
            if key not in all_results:
                continue
            r = all_results[key]
            steps = [d["step"] for d in r["defect_results"]]
            defs = [d["defect_median"] for d in r["defect_results"]]

            ax.plot(steps, defs, color=WD_COLORS[wd_val], lw=2,
                    label=f"{WD_LABELS[wd_val]}", alpha=0.9)

            # Mark grok times
            if r["grok_step_add"]:
                ax.axvline(r["grok_step_add"], color=WD_COLORS[wd_val],
                           ls=":", alpha=0.5)
            if r["grok_step_mul"]:
                ax.axvline(r["grok_step_mul"], color=WD_COLORS[wd_val],
                           ls="--", alpha=0.5)

        ax.set_xlabel("Training step", fontsize=12)
        ax.set_ylabel("Commutator defect (median, K=9)", fontsize=12)
        ax.set_title(f"Defect Traces Across Weight Decay (seed={seed})", fontsize=13)
        ax.set_yscale("log")
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(PLOT_DIR / f"figWD_A_defect_traces_s{seed}.png", dpi=150)
        plt.close(fig)
        print(f"  saved figWD_A_defect_traces_s{seed}.png")


def plot_onset_vs_grok(all_results):
    """figWD_B: Defect onset vs grok time scatter, one point per (WD, seed)."""
    fig, ax = plt.subplots(figsize=(8, 7))

    onset_data = []

    for (prefix, seed), r in sorted(all_results.items()):
        wd_val = r["wd"]
        steps = [d["step"] for d in r["defect_results"]]
        defs = [d["defect_median"] for d in r["defect_results"]]

        onset_step, baseline, threshold = detect_defect_onset(steps, defs)
        grok_step = min(
            r["grok_step_add"] or float("inf"),
            r["grok_step_mul"] or float("inf"),
        )

        if onset_step is not None and grok_step < float("inf"):
            lead = grok_step - onset_step
            onset_data.append({
                "wd": wd_val, "seed": seed,
                "onset": onset_step, "grok": grok_step,
                "lead": lead, "baseline": baseline, "threshold": threshold,
            })

            ax.scatter(onset_step, grok_step, color=WD_COLORS[wd_val], s=100,
                       edgecolors="k", zorder=5)
            ax.annotate(f"s{seed}", (onset_step, grok_step),
                        textcoords="offset points", xytext=(5, 5), fontsize=8)

    # Diagonal y=x
    all_vals = ([d["onset"] for d in onset_data] +
                [d["grok"] for d in onset_data])
    if all_vals:
        lo, hi = min(all_vals) * 0.8, max(all_vals) * 1.2
        ax.plot([lo, hi], [lo, hi], "k--", alpha=0.3, label="onset = grok")

    # Legend for WD values
    for wd_val in sorted(WD_COLORS.keys(), reverse=True):
        ax.scatter([], [], color=WD_COLORS[wd_val], s=100, edgecolors="k",
                   label=WD_LABELS[wd_val])

    ax.set_xlabel("Defect onset step", fontsize=12)
    ax.set_ylabel("Grok step (first task)", fontsize=12)
    ax.set_title("Defect Onset vs Grokking Time", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / f"figWD_B_onset_vs_grok.png", dpi=150)
    plt.close(fig)
    print(f"  saved figWD_B_onset_vs_grok.png")

    return onset_data


def plot_lead_time_bars(onset_data):
    """figWD_C: Lead time (grok - defect_onset) grouped by WD."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: absolute lead time
    ax = axes[0]
    wd_vals = sorted(set(d["wd"] for d in onset_data), reverse=True)
    positions = []
    for i, wd_val in enumerate(wd_vals):
        leads = [d["lead"] for d in onset_data if d["wd"] == wd_val]
        seeds = [d["seed"] for d in onset_data if d["wd"] == wd_val]
        x_pos = np.arange(len(leads)) + i * (len(SEEDS) + 1)
        bars = ax.bar(x_pos, leads, color=WD_COLORS[wd_val], edgecolor="k",
                      label=WD_LABELS[wd_val], alpha=0.85)
        for j, (xp, l, s) in enumerate(zip(x_pos, leads, seeds)):
            ax.text(xp, l + max(leads) * 0.02, f"s{s}", ha="center", fontsize=8)
        positions.extend(x_pos)

    ax.set_ylabel("Lead time (steps): grok - defect onset", fontsize=11)
    ax.set_title("Absolute Lead Time", fontsize=12)
    ax.set_xticks([])
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3, axis="y")

    # Right: fractional lead time (lead / grok_step)
    ax = axes[1]
    for i, wd_val in enumerate(wd_vals):
        fracs = [d["lead"] / d["grok"] for d in onset_data if d["wd"] == wd_val]
        seeds = [d["seed"] for d in onset_data if d["wd"] == wd_val]
        x_pos = np.arange(len(fracs)) + i * (len(SEEDS) + 1)
        bars = ax.bar(x_pos, [f * 100 for f in fracs], color=WD_COLORS[wd_val],
                      edgecolor="k", label=WD_LABELS[wd_val], alpha=0.85)
        for j, (xp, f, s) in enumerate(zip(x_pos, fracs, seeds)):
            ax.text(xp, f * 100 + 1, f"s{s}", ha="center", fontsize=8)

    ax.set_ylabel("Lead time (% of grok step)", fontsize=11)
    ax.set_title("Fractional Lead Time", fontsize=12)
    ax.set_xticks([])
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle("Defect Onset Precedes Grokking Across Weight Decay", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / f"figWD_C_lead_time_bars.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved figWD_C_lead_time_bars.png")


def plot_normalized_overlay(all_results):
    """figWD_D: Normalized defect traces (x-axis = fraction of grok time)."""
    fig, ax = plt.subplots(figsize=(10, 5))

    for (prefix, seed), r in sorted(all_results.items()):
        wd_val = r["wd"]
        grok_step = min(
            r["grok_step_add"] or float("inf"),
            r["grok_step_mul"] or float("inf"),
        )
        if grok_step == float("inf"):
            continue

        steps = np.array([d["step"] for d in r["defect_results"]])
        defs = np.array([d["defect_median"] for d in r["defect_results"]])

        # Normalize steps by grok time
        frac = steps / grok_step
        ax.plot(frac, defs, color=WD_COLORS[wd_val], lw=1.5, alpha=0.7,
                label=f"{WD_LABELS[wd_val]} s{seed}")

    ax.axvline(1.0, color="k", ls="--", alpha=0.5, label="grok time")
    ax.set_xlabel("Training progress (fraction of grok step)", fontsize=12)
    ax.set_ylabel("Commutator defect", fontsize=12)
    ax.set_title("Normalized Defect Traces", fontsize=13)
    ax.set_yscale("log")
    ax.set_xlim(0, 1.3)
    ax.legend(fontsize=8, ncol=3)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / f"figWD_D_normalized_traces.png", dpi=150)
    plt.close(fig)
    print(f"  saved figWD_D_normalized_traces.png")


def main():
    PLOT_DIR.mkdir(exist_ok=True)
    device = get_device()
    print(f"Device: {device}")

    all_results = {}

    for prefix, wd_val in WD_CONDITIONS.items():
        for seed in SEEDS:
            r = run_defect_for_condition(prefix, wd_val, seed)
            if r is not None:
                all_results[(prefix, seed)] = r

    if not all_results:
        print("No results computed. Exiting.")
        sys.exit(1)

    # Save raw results
    save_path = PLOT_DIR / "defect_wd_comparison.pt"
    torch.save(all_results, save_path)
    print(f"\nSaved raw results to {save_path}")

    # === Plots ===
    print("\n" + "=" * 70)
    print("  PLOTTING")
    print("=" * 70)

    plot_defect_traces(all_results)
    onset_data = plot_onset_vs_grok(all_results)
    if onset_data:
        plot_lead_time_bars(onset_data)
    plot_normalized_overlay(all_results)

    # === Summary table ===
    print("\n" + "=" * 70)
    print("  DEFECT ONSET vs GROK TIME SUMMARY")
    print("=" * 70)
    print(f"{'WD':>5s} {'Seed':>6s} {'Onset':>8s} {'Grok':>8s} {'Lead':>8s} {'Lead%':>7s}")
    print("-" * 50)

    for (prefix, seed), r in sorted(all_results.items(), key=lambda x: (-x[1]["wd"], x[1]["seed"])):
        wd_val = r["wd"]
        steps = [d["step"] for d in r["defect_results"]]
        defs = [d["defect_median"] for d in r["defect_results"]]
        onset_step, _, _ = detect_defect_onset(steps, defs)
        grok_step = min(
            r["grok_step_add"] or float("inf"),
            r["grok_step_mul"] or float("inf"),
        )
        if onset_step and grok_step < float("inf"):
            lead = grok_step - onset_step
            pct = lead / grok_step * 100
            print(f"{wd_val:>5.1f} {seed:>6d} {onset_step:>8d} {grok_step:>8d} {lead:>8d} {pct:>6.1f}%")
        else:
            print(f"{wd_val:>5.1f} {seed:>6d} {'N/A':>8s} {grok_step:>8.0f} {'N/A':>8s} {'N/A':>7s}")

    print("\nDone.")


if __name__ == "__main__":
    main()
