#!/usr/bin/env python3
"""
Defect analysis for WD=0.2 and WD=0.3.
Computes commutator defect, saves results, then merges with existing
WD=1.0/0.5/0.1 data to produce combined 5-WD plots.
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
SEEDS = [42, 137, 2024]

# New conditions to compute
NEW_CONDITIONS = {
    "multitask_wd02": 0.2,
    "multitask_wd03": 0.3,
}

# All 5 conditions for combined plots
ALL_CONDITIONS = {
    "multitask": 1.0,
    "multitask_wd05": 0.5,
    "multitask_wd03": 0.3,
    "multitask_wd02": 0.2,
    "multitask_wd01": 0.1,
}

WD_COLORS = {
    1.0: "#1a5276",
    0.5: "#e67e22",
    0.3: "#8e44ad",
    0.2: "#c0392b",
    0.1: "#27ae60",
}
WD_LABELS = {
    1.0: "WD=1.0",
    0.5: "WD=0.5",
    0.3: "WD=0.3",
    0.2: "WD=0.2",
    0.1: "WD=0.1",
}


def rolling_max(arr, window=5):
    out = np.zeros_like(arr)
    for i in range(len(arr)):
        lo = max(0, i - window // 2)
        hi = min(len(arr), i + window // 2 + 1)
        out[i] = np.max(arr[lo:hi])
    return out


def rolling_median(arr, window=7):
    out = np.zeros_like(arr)
    for i in range(len(arr)):
        lo = max(0, i - window // 2)
        hi = min(len(arr), i + window // 2 + 1)
        out[i] = np.median(arr[lo:hi])
    return out


def detect_onset_envelope(steps, defects, baseline_frac=0.25,
                          threshold_mult=5.0, window=5):
    """
    Detect defect onset using rolling-max envelope in log space.
    """
    arr = np.array(defects, dtype=float)
    arr[arr == 0] = 0.01

    envelope = rolling_max(np.log10(arr + 1e-15), window=window)
    n_base = max(3, int(len(envelope) * baseline_frac))
    baseline = np.median(envelope[:n_base])
    threshold = baseline + np.log10(threshold_mult)

    for i in range(n_base, len(envelope)):
        if envelope[i] > threshold:
            return steps[i], 10**baseline, 10**threshold
    return None, 10**baseline, 10**threshold


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


# ═══════════════════════════════════════════════════════════════════════
# Combined 5-WD plots
# ═══════════════════════════════════════════════════════════════════════

def plot_all_traces(all_results):
    """figWD5_A: Per-seed defect traces, all 5 WD overlaid."""
    for seed in SEEDS:
        fig, ax = plt.subplots(figsize=(14, 5.5))

        for prefix in ["multitask", "multitask_wd05", "multitask_wd03",
                        "multitask_wd02", "multitask_wd01"]:
            key = (prefix, seed)
            if key not in all_results:
                continue
            r = all_results[key]
            wd_val = r["wd"]
            steps = np.array([d["step"] for d in r["defect_results"]])
            defs = np.array([d["defect_median"] for d in r["defect_results"]])
            defs_plot = np.where(defs > 0, defs, 0.01)

            # Raw trace (faint)
            ax.plot(steps, defs_plot, color=WD_COLORS[wd_val], lw=0.8,
                    alpha=0.3, zorder=2)

            # Rolling median (bold)
            rm = rolling_median(defs_plot, window=7)
            ax.plot(steps, rm, color=WD_COLORS[wd_val], lw=2.5,
                    label=WD_LABELS[wd_val], zorder=3)

            # Grok time (dashed vertical)
            grok = min(r["grok_step_add"] or 1e9, r["grok_step_mul"] or 1e9)
            if grok < 1e9:
                ax.axvline(grok, color=WD_COLORS[wd_val], ls="--",
                           alpha=0.5, lw=1.5, zorder=1)

            # Onset marker
            onset, _, _ = detect_onset_envelope(steps, defs)
            if onset is not None:
                idx = np.argmin(np.abs(steps - onset))
                ax.plot(onset, defs_plot[idx], "v", color=WD_COLORS[wd_val],
                        markersize=10, zorder=5, markeredgecolor="k",
                        markeredgewidth=0.5)

        ax.set_xlabel("Training step", fontsize=12)
        ax.set_ylabel("Commutator defect (median, K=9)", fontsize=12)
        ax.set_title(f"Defect Across 5 Weight-Decay Values (seed={seed})\n"
                     f"solid=rolling median · dashed=grok time · ▼=defect onset",
                     fontsize=11)
        ax.set_yscale("log")
        ax.legend(fontsize=9, loc="upper left")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(PLOT_DIR / f"figWD5_A_traces_s{seed}.png", dpi=150)
        plt.close(fig)
        print(f"  saved figWD5_A_traces_s{seed}.png")


def plot_onset_scatter(all_results):
    """figWD5_B: Onset vs grok scatter, all 5 WD."""
    fig, ax = plt.subplots(figsize=(9, 7))

    onset_table = []
    for key, r in sorted(all_results.items()):
        prefix, seed = key
        wd_val = r["wd"]
        steps = np.array([d["step"] for d in r["defect_results"]])
        defs = np.array([d["defect_median"] for d in r["defect_results"]])
        onset, _, _ = detect_onset_envelope(steps, defs)
        grok = min(r["grok_step_add"] or 1e9, r["grok_step_mul"] or 1e9)
        if grok >= 1e9:
            grok = None
        lead = (grok - onset) if (onset is not None and grok is not None) else None
        lead_pct = (lead / grok * 100) if (lead is not None and grok) else None

        onset_table.append({
            "wd": wd_val, "seed": seed, "onset": onset, "grok": grok,
            "lead": lead, "lead_pct": lead_pct,
        })

        if onset is not None and grok is not None:
            ax.scatter(onset, grok, color=WD_COLORS[wd_val], s=120,
                       edgecolors="k", zorder=5, linewidth=1.5)
            ax.annotate(f"s{seed}", (onset, grok), textcoords="offset points",
                        xytext=(6, 6), fontsize=8)

    # Diagonal
    all_vals = [r["onset"] for r in onset_table if r["onset"]] + \
               [r["grok"] for r in onset_table if r["grok"]]
    if all_vals:
        hi = max(all_vals) * 1.15
        ax.plot([0, hi], [0, hi], "k--", alpha=0.3, lw=1, label="onset = grok")

    for wd_val in sorted(WD_COLORS.keys(), reverse=True):
        ax.scatter([], [], color=WD_COLORS[wd_val], s=120, edgecolors="k",
                   label=WD_LABELS[wd_val])

    ax.set_xlabel("Defect onset step", fontsize=12)
    ax.set_ylabel("Grok step (first task)", fontsize=12)
    ax.set_title("Defect Onset Precedes Grokking\n(all points above diagonal)",
                 fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / f"figWD5_B_onset_vs_grok.png", dpi=150)
    plt.close(fig)
    print(f"  saved figWD5_B_onset_vs_grok.png")

    return onset_table


def plot_lead_bars(onset_table):
    """figWD5_C: Lead time bars grouped by WD."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))

    wd_order = [1.0, 0.5, 0.3, 0.2, 0.1]

    for ax_idx, (ylabel, key_fn, fmt) in enumerate([
        ("Lead time (steps)", lambda r: r["lead"], lambda v: f"{int(v)}"),
        ("Lead time (% of grok step)", lambda r: r["lead_pct"], lambda v: f"{v:.1f}%"),
    ]):
        ax = axes[ax_idx]
        x_offset = 0
        for wd_val in wd_order:
            rows = [r for r in onset_table
                    if r["wd"] == wd_val and key_fn(r) is not None]
            vals = [key_fn(r) for r in rows]
            seeds = [r["seed"] for r in rows]
            if not vals:
                x_offset += len(SEEDS) + 1
                continue
            x_pos = np.arange(len(vals)) + x_offset
            ax.bar(x_pos, vals, color=WD_COLORS[wd_val], edgecolor="k",
                   label=WD_LABELS[wd_val], alpha=0.85, width=0.8)
            for xp, v, s in zip(x_pos, vals, seeds):
                ax.text(xp, v + max(vals) * 0.02, f"s{s}", ha="center",
                        fontsize=7)
            x_offset += len(SEEDS) + 1

        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_xticks([])
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(alpha=0.3, axis="y")

    axes[0].set_title("Absolute Lead Time", fontsize=12)
    axes[1].set_title("Fractional Lead Time", fontsize=12)
    fig.suptitle("Defect Onset Lead Time Across 5 Weight-Decay Values",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / f"figWD5_C_lead_bars.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved figWD5_C_lead_bars.png")


def plot_normalized(all_results):
    """figWD5_D: Normalized traces (x = fraction of grok time)."""
    fig, ax = plt.subplots(figsize=(12, 5.5))

    for key, r in sorted(all_results.items()):
        prefix, seed = key
        wd_val = r["wd"]
        grok = min(r["grok_step_add"] or 1e9, r["grok_step_mul"] or 1e9)
        if grok >= 1e9:
            continue

        steps = np.array([d["step"] for d in r["defect_results"]])
        defs = np.array([d["defect_median"] for d in r["defect_results"]])
        defs[defs == 0] = 0.01
        frac = steps / grok

        rm = rolling_median(defs, window=7)
        ax.plot(frac, rm, color=WD_COLORS[wd_val], lw=1.5, alpha=0.7,
                label=f"{WD_LABELS[wd_val]} s{seed}")

    ax.axvline(1.0, color="k", ls="--", alpha=0.5, lw=1.5, label="grok time")
    ax.set_xlabel("Training progress (fraction of grok step)", fontsize=12)
    ax.set_ylabel("Commutator defect (rolling median)", fontsize=12)
    ax.set_title("Normalized Defect Traces (5 WD values)", fontsize=13)
    ax.set_yscale("log")
    ax.set_xlim(0, 1.15)
    ax.legend(fontsize=7, ncol=5, loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / f"figWD5_D_normalized.png", dpi=150)
    plt.close(fig)
    print(f"  saved figWD5_D_normalized.png")


def plot_wd_scaling(onset_table):
    """figWD5_E: Lead% vs WD — shows the relationship as a curve."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    wd_order = [0.1, 0.2, 0.3, 0.5, 1.0]

    # Left: Grok time vs WD
    ax = axes[0]
    for wd_val in wd_order:
        groks = [r["grok"] for r in onset_table
                 if r["wd"] == wd_val and r["grok"] is not None]
        for g in groks:
            ax.scatter(wd_val, g, color=WD_COLORS[wd_val], s=80,
                       edgecolors="k", zorder=5)
    # Mean line
    wd_means = []
    for wd_val in wd_order:
        groks = [r["grok"] for r in onset_table
                 if r["wd"] == wd_val and r["grok"] is not None]
        if groks:
            wd_means.append((wd_val, np.mean(groks)))
    if wd_means:
        ax.plot([w[0] for w in wd_means], [w[1] for w in wd_means],
                "k-", lw=1.5, alpha=0.5, zorder=1)
    ax.set_xlabel("Weight Decay", fontsize=11)
    ax.set_ylabel("Grok step", fontsize=11)
    ax.set_title("Grok Time vs WD", fontsize=12)
    ax.grid(alpha=0.3)

    # Middle: Onset vs WD
    ax = axes[1]
    for wd_val in wd_order:
        onsets = [r["onset"] for r in onset_table
                  if r["wd"] == wd_val and r["onset"] is not None]
        for o in onsets:
            ax.scatter(wd_val, o, color=WD_COLORS[wd_val], s=80,
                       edgecolors="k", zorder=5)
    wd_means_onset = []
    for wd_val in wd_order:
        onsets = [r["onset"] for r in onset_table
                  if r["wd"] == wd_val and r["onset"] is not None]
        if onsets:
            wd_means_onset.append((wd_val, np.mean(onsets)))
    if wd_means_onset:
        ax.plot([w[0] for w in wd_means_onset], [w[1] for w in wd_means_onset],
                "k-", lw=1.5, alpha=0.5, zorder=1)
    ax.set_xlabel("Weight Decay", fontsize=11)
    ax.set_ylabel("Defect onset step", fontsize=11)
    ax.set_title("Defect Onset vs WD", fontsize=12)
    ax.grid(alpha=0.3)

    # Right: Lead% vs WD
    ax = axes[2]
    for wd_val in wd_order:
        pcts = [r["lead_pct"] for r in onset_table
                if r["wd"] == wd_val and r["lead_pct"] is not None]
        for p in pcts:
            ax.scatter(wd_val, p, color=WD_COLORS[wd_val], s=80,
                       edgecolors="k", zorder=5)
    wd_means_pct = []
    for wd_val in wd_order:
        pcts = [r["lead_pct"] for r in onset_table
                if r["wd"] == wd_val and r["lead_pct"] is not None]
        if pcts:
            wd_means_pct.append((wd_val, np.mean(pcts)))
    if wd_means_pct:
        ax.plot([w[0] for w in wd_means_pct], [w[1] for w in wd_means_pct],
                "k-", lw=1.5, alpha=0.5, zorder=1)
    ax.set_xlabel("Weight Decay", fontsize=11)
    ax.set_ylabel("Lead time (% of grok step)", fontsize=11)
    ax.set_title("Fractional Lead vs WD", fontsize=12)
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 100)

    fig.suptitle("Scaling of Grok Time, Onset, and Lead with Weight Decay",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / f"figWD5_E_wd_scaling.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved figWD5_E_wd_scaling.png")


def plot_summary_table(onset_table):
    """figWD5_F: Summary table figure."""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis("off")

    headers = ["WD", "Seed", "Defect Onset", "Grok Step", "Lead (steps)", "Lead (%)"]
    rows = []
    wd_bg = {1.0: "#d4e6f1", 0.5: "#fdebd0", 0.3: "#e8daef",
             0.2: "#fadbd8", 0.1: "#d5f5e3"}
    cell_colors = []

    for r in sorted(onset_table, key=lambda x: (-x["wd"], x["seed"])):
        rows.append([
            f"{r['wd']:.1f}",
            str(r["seed"]),
            str(int(r["onset"])) if r["onset"] else "N/A",
            str(int(r["grok"])) if r["grok"] else "N/A",
            str(int(r["lead"])) if r["lead"] else "N/A",
            f"{r['lead_pct']:.1f}%" if r["lead_pct"] else "N/A",
        ])
        cell_colors.append([wd_bg.get(r["wd"], "#ffffff")] * len(headers))

    table = ax.table(cellText=rows, colLabels=headers, cellColours=cell_colors,
                     loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.4)

    for j in range(len(headers)):
        table[0, j].set_text_props(fontweight="bold")
        table[0, j].set_facecolor("#aab7b8")

    ax.set_title("Defect Onset vs Grok Time — Full Summary (5 WD values)",
                 fontsize=13, pad=20)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / f"figWD5_F_summary_table.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved figWD5_F_summary_table.png")


def main():
    PLOT_DIR.mkdir(exist_ok=True)
    device = get_device()
    print(f"Device: {device}")

    # ─── Step 1: Load existing WD=1.0/0.5/0.1 results ───
    existing_path = PLOT_DIR / "defect_wd_comparison.pt"
    if existing_path.exists():
        all_results = torch.load(existing_path, map_location="cpu",
                                 weights_only=False)
        print(f"Loaded {len(all_results)} existing conditions from cache")
    else:
        all_results = {}
        print("No cached results found, starting fresh")

    # ─── Step 2: Compute defect for WD=0.2 and WD=0.3 ───
    new_results = {}
    for prefix, wd_val in NEW_CONDITIONS.items():
        for seed in SEEDS:
            key = (prefix, seed)
            if key in all_results:
                print(f"  [{prefix}_s{seed}] already in cache, skipping")
                continue
            r = run_defect_for_condition(prefix, wd_val, seed)
            if r is not None:
                new_results[key] = r

    # Merge
    all_results.update(new_results)
    print(f"\nTotal conditions: {len(all_results)}")

    # Save merged results
    save_path = PLOT_DIR / "defect_wd_all5.pt"
    torch.save(all_results, save_path)
    print(f"Saved all results to {save_path}")

    # ─── Step 3: Plots ───
    print("\n" + "=" * 70)
    print("  PLOTTING (5 WD values)")
    print("=" * 70)

    plot_all_traces(all_results)
    onset_table = plot_onset_scatter(all_results)
    plot_lead_bars(onset_table)
    plot_normalized(all_results)
    plot_wd_scaling(onset_table)
    plot_summary_table(onset_table)

    # ─── Step 4: Summary ───
    print(f"\n{'='*70}")
    print(f"  DEFECT ONSET vs GROK TIME — ALL 5 WD VALUES")
    print(f"{'='*70}")
    print(f"{'WD':>5s} {'Seed':>6s} {'Onset':>8s} {'Grok':>8s} {'Lead':>8s} {'Lead%':>7s}")
    print("-" * 50)
    for r in sorted(onset_table, key=lambda x: (-x["wd"], x["seed"])):
        o = str(int(r["onset"])) if r["onset"] else "N/A"
        g = str(int(r["grok"])) if r["grok"] else "N/A"
        l = str(int(r["lead"])) if r["lead"] else "N/A"
        p = f"{r['lead_pct']:.1f}%" if r["lead_pct"] else "N/A"
        print(f"{r['wd']:>5.1f} {r['seed']:>6d} {o:>8s} {g:>8s} {l:>8s} {p:>7s}")

    print(f"\n{'WD':>5s} {'Mean Grok':>10s} {'Mean Onset':>11s} {'Mean Lead':>10s} {'Mean Lead%':>11s}")
    print("-" * 55)
    for wd_val in [1.0, 0.5, 0.3, 0.2, 0.1]:
        groks = [r["grok"] for r in onset_table if r["wd"] == wd_val and r["grok"]]
        onsets = [r["onset"] for r in onset_table if r["wd"] == wd_val and r["onset"]]
        leads = [r["lead"] for r in onset_table if r["wd"] == wd_val and r["lead"]]
        pcts = [r["lead_pct"] for r in onset_table if r["wd"] == wd_val and r["lead_pct"]]
        mg = f"{np.mean(groks):.0f}" if groks else "N/A"
        mo = f"{np.mean(onsets):.0f}" if onsets else "N/A"
        ml = f"{np.mean(leads):.0f}" if leads else "N/A"
        mp = f"{np.mean(pcts):.1f}%" if pcts else "N/A"
        print(f"{wd_val:>5.1f} {mg:>10s} {mo:>11s} {ml:>10s} {mp:>11s}")

    print("\nDone.")


if __name__ == "__main__":
    main()
