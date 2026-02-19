#!/usr/bin/env python3
"""
Replot defect WD comparison from saved results with improved onset detection.

For WD=0.1 the defect is bimodal (low baseline + intermittent spikes).
Use rolling-max envelope to smooth out the transient nature of spikes.
"""

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

PLOT_DIR = Path(__file__).parent / "plots"
SEEDS = [42, 137, 2024]
WD_COLORS = {1.0: "#1a5276", 0.5: "#e67e22", 0.1: "#27ae60"}
WD_LABELS = {1.0: "WD=1.0", 0.5: "WD=0.5", 0.1: "WD=0.1"}


def rolling_max(arr, window=5):
    """Rolling maximum with given window."""
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
    Detect defect onset using rolling-max envelope.
    More robust for WD=0.1's intermittent spike pattern.
    """
    arr = np.array(defects, dtype=float)
    # Replace exact zeros with small value for log-space analysis
    arr[arr == 0] = 0.01

    envelope = rolling_max(np.log10(arr + 1e-15), window=window)
    n_base = max(3, int(len(envelope) * baseline_frac))
    baseline = np.median(envelope[:n_base])
    threshold = baseline + np.log10(threshold_mult)  # log-space threshold

    # First point where envelope exceeds threshold
    for i in range(n_base, len(envelope)):
        if envelope[i] > threshold:
            return steps[i], 10**baseline, 10**threshold
    return None, 10**baseline, 10**threshold


def main():
    data = torch.load(PLOT_DIR / "defect_wd_comparison.pt",
                      map_location="cpu", weights_only=False)
    print(f"Loaded {len(data)} conditions\n")

    # ══════════════════════════════════════════════════════════════════════
    # Figure 1: Per-seed traces with onset markers
    # ══════════════════════════════════════════════════════════════════════
    for seed in SEEDS:
        fig, ax = plt.subplots(figsize=(12, 5))

        for prefix in ["multitask", "multitask_wd05", "multitask_wd01"]:
            key = (prefix, seed)
            if key not in data:
                continue
            r = data[key]
            wd_val = r["wd"]
            steps = np.array([d["step"] for d in r["defect_results"]])
            defs = np.array([d["defect_median"] for d in r["defect_results"]])
            defs_plot = np.where(defs > 0, defs, 0.01)

            # Raw trace
            ax.plot(steps, defs_plot, color=WD_COLORS[wd_val], lw=1.2,
                    alpha=0.5, zorder=2)

            # Rolling median (smooth)
            rm = rolling_median(defs_plot, window=7)
            ax.plot(steps, rm, color=WD_COLORS[wd_val], lw=2.5,
                    label=f"{WD_LABELS[wd_val]}", zorder=3)

            # Mark grok time
            grok = min(r["grok_step_add"] or 1e9, r["grok_step_mul"] or 1e9)
            if grok < 1e9:
                ax.axvline(grok, color=WD_COLORS[wd_val], ls="--", alpha=0.6,
                           lw=1.5, zorder=1)

            # Onset detection
            onset, base, thresh = detect_onset_envelope(steps, defs)
            if onset is not None:
                ax.axvline(onset, color=WD_COLORS[wd_val], ls=":",
                           alpha=0.8, lw=2, zorder=4)
                ax.plot(onset, defs_plot[np.argmin(np.abs(steps - onset))],
                        "v", color=WD_COLORS[wd_val], markersize=10, zorder=5)

        ax.set_xlabel("Training step", fontsize=12)
        ax.set_ylabel("Commutator defect (median, K=9)", fontsize=12)
        ax.set_title(f"Defect Across Weight Decay (seed={seed})\n"
                     f"(solid: rolling median; dashed: grok time; dotted/triangle: defect onset)",
                     fontsize=11)
        ax.set_yscale("log")
        ax.legend(fontsize=10, loc="upper left")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(PLOT_DIR / f"figWD_A2_defect_traces_s{seed}.png", dpi=150)
        plt.close(fig)
        print(f"  saved figWD_A2_defect_traces_s{seed}.png")

    # ══════════════════════════════════════════════════════════════════════
    # Build onset table
    # ══════════════════════════════════════════════════════════════════════
    onset_table = []
    for key, r in sorted(data.items()):
        prefix, seed = key
        wd_val = r["wd"]
        steps = np.array([d["step"] for d in r["defect_results"]])
        defs = np.array([d["defect_median"] for d in r["defect_results"]])
        onset, base, thresh = detect_onset_envelope(steps, defs)
        grok = min(r["grok_step_add"] or 1e9, r["grok_step_mul"] or 1e9)
        if grok >= 1e9:
            grok = None

        lead = (grok - onset) if (onset is not None and grok is not None) else None
        lead_pct = (lead / grok * 100) if (lead is not None and grok) else None

        onset_table.append({
            "wd": wd_val, "seed": seed, "onset": onset, "grok": grok,
            "lead": lead, "lead_pct": lead_pct,
        })

    # ══════════════════════════════════════════════════════════════════════
    # Figure 2: Onset vs Grok scatter (improved)
    # ══════════════════════════════════════════════════════════════════════
    fig, ax = plt.subplots(figsize=(8, 7))

    for row in onset_table:
        if row["onset"] is None or row["grok"] is None:
            continue
        ax.scatter(row["onset"], row["grok"], color=WD_COLORS[row["wd"]],
                   s=120, edgecolors="k", zorder=5, linewidth=1.5)
        ax.annotate(f"s{row['seed']}", (row["onset"], row["grok"]),
                    textcoords="offset points", xytext=(6, 6), fontsize=9)

    all_vals = [row["onset"] for row in onset_table if row["onset"]] + \
               [row["grok"] for row in onset_table if row["grok"]]
    if all_vals:
        lo, hi = 0, max(all_vals) * 1.15
        ax.plot([lo, hi], [lo, hi], "k--", alpha=0.3, lw=1, label="onset = grok")

    for wd_val in sorted(WD_COLORS.keys(), reverse=True):
        ax.scatter([], [], color=WD_COLORS[wd_val], s=120, edgecolors="k",
                   label=WD_LABELS[wd_val])

    ax.set_xlabel("Defect onset step", fontsize=12)
    ax.set_ylabel("Grok step (first task)", fontsize=12)
    ax.set_title("Defect Onset Precedes Grokking\n(all points above diagonal)", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / f"figWD_B2_onset_vs_grok.png", dpi=150)
    plt.close(fig)
    print(f"  saved figWD_B2_onset_vs_grok.png")

    # ══════════════════════════════════════════════════════════════════════
    # Figure 3: Lead time grouped bars
    # ══════════════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    wd_vals = [1.0, 0.5, 0.1]
    x_offset = 0
    for ax_idx, (ylabel, key_fn) in enumerate([
        ("Lead time (steps)", lambda r: r["lead"]),
        ("Lead time (% of grok step)", lambda r: r["lead_pct"]),
    ]):
        ax = axes[ax_idx]
        x_offset = 0
        for wd_val in wd_vals:
            rows = [r for r in onset_table if r["wd"] == wd_val and key_fn(r) is not None]
            if not rows:
                x_offset += len(SEEDS) + 1
                continue
            vals = [key_fn(r) for r in rows]
            seeds = [r["seed"] for r in rows]
            x_pos = np.arange(len(vals)) + x_offset
            bars = ax.bar(x_pos, vals, color=WD_COLORS[wd_val], edgecolor="k",
                          label=WD_LABELS[wd_val], alpha=0.85, width=0.8)
            for xp, v, s in zip(x_pos, vals, seeds):
                ax.text(xp, v + max(vals) * 0.02, f"s{s}", ha="center", fontsize=8)
            x_offset += len(SEEDS) + 1

        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_xticks([])
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3, axis="y")

    axes[0].set_title("Absolute Lead Time", fontsize=12)
    axes[1].set_title("Fractional Lead Time", fontsize=12)
    fig.suptitle("Defect Onset Precedes Grokking Across Weight Decay",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / f"figWD_C2_lead_time_bars.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved figWD_C2_lead_time_bars.png")

    # ══════════════════════════════════════════════════════════════════════
    # Figure 4: Normalized traces (x = fraction of grok time)
    # ══════════════════════════════════════════════════════════════════════
    fig, ax = plt.subplots(figsize=(10, 5))

    for key, r in sorted(data.items()):
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
        ax.plot(frac, rm, color=WD_COLORS[wd_val], lw=1.8, alpha=0.7,
                label=f"{WD_LABELS[wd_val]} s{seed}")

    ax.axvline(1.0, color="k", ls="--", alpha=0.5, lw=1.5, label="grok time")
    ax.set_xlabel("Training progress (fraction of grok step)", fontsize=12)
    ax.set_ylabel("Commutator defect (rolling median)", fontsize=12)
    ax.set_title("Normalized Defect Traces", fontsize=13)
    ax.set_yscale("log")
    ax.set_xlim(0, 1.15)
    ax.legend(fontsize=8, ncol=3, loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / f"figWD_D2_normalized_traces.png", dpi=150)
    plt.close(fig)
    print(f"  saved figWD_D2_normalized_traces.png")

    # ══════════════════════════════════════════════════════════════════════
    # Figure 5: Summary table plot
    # ══════════════════════════════════════════════════════════════════════
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis("off")

    headers = ["WD", "Seed", "Defect Onset", "Grok Step", "Lead (steps)", "Lead (%)"]
    rows = []
    for r in sorted(onset_table, key=lambda x: (-x["wd"], x["seed"])):
        rows.append([
            f"{r['wd']:.1f}",
            str(r["seed"]),
            str(r["onset"]) if r["onset"] else "N/A",
            str(int(r["grok"])) if r["grok"] else "N/A",
            str(int(r["lead"])) if r["lead"] else "N/A",
            f"{r['lead_pct']:.1f}%" if r["lead_pct"] else "N/A",
        ])

    # Color by WD
    cell_colors = []
    wd_bg = {1.0: "#d4e6f1", 0.5: "#fdebd0", 0.1: "#d5f5e3"}
    for r in sorted(onset_table, key=lambda x: (-x["wd"], x["seed"])):
        cell_colors.append([wd_bg[r["wd"]]] * len(headers))

    table = ax.table(cellText=rows, colLabels=headers, cellColours=cell_colors,
                     loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.5)

    # Bold header
    for j, h in enumerate(headers):
        table[0, j].set_text_props(fontweight="bold")
        table[0, j].set_facecolor("#aab7b8")

    ax.set_title("Defect Onset vs Grok Time Summary", fontsize=13, pad=20)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / f"figWD_E_summary_table.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved figWD_E_summary_table.png")

    # ══════════════════════════════════════════════════════════════════════
    # Print summary
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"  DEFECT ONSET vs GROK TIME SUMMARY (improved detection)")
    print(f"{'='*60}")
    print(f"{'WD':>5s} {'Seed':>6s} {'Onset':>8s} {'Grok':>8s} {'Lead':>8s} {'Lead%':>7s}")
    print("-" * 50)
    for r in sorted(onset_table, key=lambda x: (-x["wd"], x["seed"])):
        onset_str = str(r["onset"]) if r["onset"] else "N/A"
        grok_str = str(int(r["grok"])) if r["grok"] else "N/A"
        lead_str = str(int(r["lead"])) if r["lead"] else "N/A"
        pct_str = f"{r['lead_pct']:.1f}%" if r["lead_pct"] else "N/A"
        print(f"{r['wd']:>5.1f} {r['seed']:>6d} {onset_str:>8s} {grok_str:>8s} {lead_str:>8s} {pct_str:>7s}")

    # Per-WD averages
    print(f"\n{'WD':>5s} {'Mean Lead':>10s} {'Mean Lead%':>11s}")
    print("-" * 30)
    for wd_val in [1.0, 0.5, 0.1]:
        leads = [r["lead"] for r in onset_table if r["wd"] == wd_val and r["lead"] is not None]
        pcts = [r["lead_pct"] for r in onset_table if r["wd"] == wd_val and r["lead_pct"] is not None]
        if leads:
            print(f"{wd_val:>5.1f} {np.mean(leads):>10.0f} {np.mean(pcts):>10.1f}%")
        else:
            print(f"{wd_val:>5.1f} {'N/A':>10s} {'N/A':>11s}")


if __name__ == "__main__":
    main()
