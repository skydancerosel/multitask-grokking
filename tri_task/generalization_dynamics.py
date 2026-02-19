#!/usr/bin/env python3
"""
Defect onset lead time experiment for tri-task grokking.

Adapts grok_generalization_dynamics.py for the 3-task setting.
Trains models from scratch with inline commutator defect + test accuracy
measurement every COMM_EVERY steps, then computes when the defect spike
precedes grokking for each task.

Sweeps WD ∈ {1.0, 0.5, 0.1} × seeds {42, 137, 2024}, plus WD=0.0 control.

Produces:
  figTT_W  — Dual-axis defect vs test accuracy (per task) for each WD
  figTT_X  — Lead-time scatter + bar chart
  figTT_W2 — Hero figure (best lead example)
  figTT_W3 — Cross-WD lead time comparison
"""

import math, time, random, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).parent))
from train_tritask import (
    TriTaskConfig, TriTaskTransformer, build_dataset, sample_batch,
    get_device, eval_accuracy, TASK_NAMES,
)

sys.path.insert(0, str(Path(__file__).parent.parent))
from grok_commutator_analysis import flatten_model_params, _param_offsets

# ── config ───────────────────────────────────────────────────────────────
PLOT_DIR = Path(__file__).parent / "plots"
CACHE_PATH = PLOT_DIR / "generalization_dynamics_results.pt"

WD_VALUES = [1.0, 0.5, 0.1]
WD_CONTROL = 0.0
SEEDS = [42, 137, 2024]

COMM_EVERY = 100        # commutator measurement interval (steps)
COMM_K = 5              # commutator samples per checkpoint (lighter than 9)
COMM_ETA = 1e-3

# WD-dependent training budgets
WD_BUDGETS = {1.0: 100_000, 0.5: 100_000, 0.1: 350_000, 0.0: 50_000}

POST_GROK_STEPS = 2000  # keep training after all 3 tasks grok

TASK_COLORS = {"add": "#1f77b4", "mul": "#d62728", "sq": "#2ca02c"}
WD_COLORS = {1.0: "#2ca02c", 0.5: "#1f77b4", 0.1: "#ff7f0e", 0.0: "#d62728"}
SEED_STYLES = {42: "-", 137: "--", 2024: ":"}


# ═══════════════════════════════════════════════════════════════════════════
# Commutator defect (inline — simplified for speed)
# ═══════════════════════════════════════════════════════════════════════════

def _write_params(model, theta):
    with torch.no_grad():
        offset = 0
        for p in model.parameters():
            if not p.requires_grad:
                continue
            n = p.numel()
            p.copy_(theta[offset:offset+n].view_as(p))
            offset += n


def commutator_defect_total(model, batch_fn, device, eta=1e-3, eps=1e-12):
    """Commutator defect using total (add+mul+sq) loss for both batches."""
    was_training = model.training
    model.train()

    a1, b1, ya1, ym1, ys1 = batch_fn()
    a2, b2, ya2, ym2, ys2 = batch_fn()

    theta0 = torch.cat([p.detach().flatten() for p in model.parameters() if p.requires_grad])

    # Gradient A
    model.zero_grad(set_to_none=True)
    la, lm, ls = model(a1, b1)
    loss = F.cross_entropy(la, ya1) + F.cross_entropy(lm, ym1) + F.cross_entropy(ls, ys1)
    loss.backward()
    gA = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).flatten()
                     for p in model.parameters() if p.requires_grad])

    # Gradient B
    model.zero_grad(set_to_none=True)
    la, lm, ls = model(a2, b2)
    loss = F.cross_entropy(la, ya2) + F.cross_entropy(lm, ym2) + F.cross_entropy(ls, ys2)
    loss.backward()
    gB = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).flatten()
                     for p in model.parameters() if p.requires_grad])

    grad_cos = (gA @ gB) / (gA.norm() * gB.norm() + eps)

    # Path AB
    _write_params(model, theta0 - eta * gA)
    model.zero_grad(set_to_none=True)
    la, lm, ls = model(a2, b2)
    loss = F.cross_entropy(la, ya2) + F.cross_entropy(lm, ym2) + F.cross_entropy(ls, ys2)
    loss.backward()
    gB1 = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).flatten()
                      for p in model.parameters() if p.requires_grad])
    thetaAB = theta0 - eta * gA - eta * gB1

    # Path BA
    _write_params(model, theta0 - eta * gB)
    model.zero_grad(set_to_none=True)
    la, lm, ls = model(a1, b1)
    loss = F.cross_entropy(la, ya1) + F.cross_entropy(lm, ym1) + F.cross_entropy(ls, ys1)
    loss.backward()
    gA1 = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).flatten()
                      for p in model.parameters() if p.requires_grad])
    thetaBA = theta0 - eta * gB - eta * gA1

    _write_params(model, theta0)
    if not was_training:
        model.eval()

    normA = (eta * gA).norm()
    normB = (eta * gB).norm()
    delta = thetaAB - thetaBA
    defect = (delta.norm() / (normA * normB + eps)).item()
    return defect, grad_cos.item()


# ═══════════════════════════════════════════════════════════════════════════
# Training with inline defect tracking
# ═══════════════════════════════════════════════════════════════════════════

def train_with_defect_tracking(wd, seed, max_steps=None):
    """
    Train a tri-task model, measuring commutator defect + per-task test
    accuracy every COMM_EVERY steps.
    """
    device = get_device()
    steps = max_steps if max_steps is not None else WD_BUDGETS.get(wd, 200_000)
    cfg = TriTaskConfig(SEED=seed, WEIGHT_DECAY=wd, STEPS=steps)

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    train_pairs, test_pairs = build_dataset(cfg.P, cfg.TRAIN_FRACTION, cfg.SEED)

    model = TriTaskTransformer(cfg).to(device)
    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg.LR, weight_decay=wd,
        betas=(cfg.ADAM_BETA1, cfg.ADAM_BETA2)
    )
    loss_fn = nn.CrossEntropyLoss()

    def batch_fn():
        return sample_batch(train_pairs, cfg.BATCH_SIZE, cfg.P, device)

    records = []
    grokked = {t: False for t in TASK_NAMES}
    grok_step = {t: None for t in TASK_NAMES}
    patience = {t: 0 for t in TASK_NAMES}
    all_grokked = False
    steps_after_all_grok = 0
    t0 = time.time()

    # Measure at step 0
    model.eval()
    tr_add, tr_mul, tr_sq = eval_accuracy(model, train_pairs, cfg, device)
    te_add, te_mul, te_sq = eval_accuracy(model, test_pairs, cfg, device)
    defects = []
    for _ in range(COMM_K):
        D, gcos = commutator_defect_total(model, batch_fn, device, eta=COMM_ETA)
        defects.append(D)
    records.append({
        "step": 0,
        "defect_median": float(np.median(defects)),
        "defect_p25": float(np.percentile(defects, 25)),
        "defect_p75": float(np.percentile(defects, 75)),
        "test_add": te_add, "test_mul": te_mul, "test_sq": te_sq,
        "train_add": tr_add, "train_mul": tr_mul, "train_sq": tr_sq,
    })

    for step in range(1, cfg.STEPS + 1):
        model.train()
        a, b, y_add, y_mul, y_sq = sample_batch(train_pairs, cfg.BATCH_SIZE, cfg.P, device)
        logits_add, logits_mul, logits_sq = model(a, b)
        loss = (loss_fn(logits_add, y_add) + loss_fn(logits_mul, y_mul) +
                loss_fn(logits_sq, y_sq))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
        opt.step()

        # Measure at regular intervals
        if step % COMM_EVERY == 0:
            model.eval()
            tr_add, tr_mul, tr_sq = eval_accuracy(model, train_pairs, cfg, device)
            te_add, te_mul, te_sq = eval_accuracy(model, test_pairs, cfg, device)

            defects = []
            for _ in range(COMM_K):
                D, gcos = commutator_defect_total(model, batch_fn, device, eta=COMM_ETA)
                defects.append(D)

            records.append({
                "step": step,
                "defect_median": float(np.median(defects)),
                "defect_p25": float(np.percentile(defects, 25)),
                "defect_p75": float(np.percentile(defects, 75)),
                "test_add": te_add, "test_mul": te_mul, "test_sq": te_sq,
                "train_add": tr_add, "train_mul": tr_mul, "train_sq": tr_sq,
            })

            # Check for grokking per task
            test_accs = {"add": te_add, "mul": te_mul, "sq": te_sq}
            for t in TASK_NAMES:
                if not grokked[t]:
                    if test_accs[t] >= cfg.STOP_ACC:
                        patience[t] += 1
                        if patience[t] >= cfg.STOP_PATIENCE:
                            grokked[t] = True
                            grok_step[t] = step
                            print(f"      {t.upper()} GROKKED at step {step}")
                    else:
                        patience[t] = 0

            if all(grokked.values()) and not all_grokked:
                all_grokked = True
                print(f"      ALL GROKKED at step {step}")

            model.train()

        # Post-grok: continue a bit then stop
        if all_grokked:
            steps_after_all_grok += 1
            if steps_after_all_grok >= POST_GROK_STEPS:
                break

        # Progress logging
        if step % 2000 == 0:
            elapsed = (time.time() - t0) / 60
            last_r = records[-1] if records else {}
            d = last_r.get("defect_median", 0)
            ta = last_r.get("test_add", 0)
            tm = last_r.get("test_mul", 0)
            ts = last_r.get("test_sq", 0)
            print(f"      step {step:6d} | add={ta:.3f} mul={tm:.3f} sq={ts:.3f} | "
                  f"defect={d:.1f} | {elapsed:.1f}m")

    return {
        "records": records,
        "grokked": grokked,
        "grok_step": grok_step,
        "wd": wd,
        "seed": seed,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Analysis: spike + grok detection
# ═══════════════════════════════════════════════════════════════════════════

def find_spike_step(records, threshold_factor=10, min_defect=20):
    """First step where defect > threshold_factor × baseline AND > min_defect."""
    if len(records) < 3:
        return None
    baseline = np.median([r["defect_median"] for r in records[:3]])
    baseline = max(baseline, 0.1)
    for i in range(2, len(records)):
        d = records[i]["defect_median"]
        if d > threshold_factor * baseline and d > min_defect:
            return records[i]["step"]
    return None


def find_grok_step_from_records(records, task, threshold=0.9):
    """First step where test accuracy for `task` exceeds threshold."""
    key = f"test_{task}"
    for r in records:
        if r.get(key, 0) >= threshold:
            return r["step"]
    return None


# ═══════════════════════════════════════════════════════════════════════════
# Figures
# ═══════════════════════════════════════════════════════════════════════════

def plot_figW(all_runs):
    """
    FigTT_W: For each WD, show defect (left axis) vs per-task test accuracy
    (right axis), 3 seeds overlaid.  One row per WD.
    """
    wd_list = [wd for wd in WD_VALUES if any((wd, s) in all_runs for s in SEEDS)]
    n_wd = len(wd_list)
    if n_wd == 0:
        return

    fig, axes = plt.subplots(n_wd, 1, figsize=(14, 5 * n_wd), squeeze=False)

    for row, wd in enumerate(wd_list):
        ax = axes[row, 0]
        ax2 = ax.twinx()

        for seed in SEEDS:
            key = (wd, seed)
            if key not in all_runs:
                continue
            data = all_runs[key]
            recs = data["records"]
            steps = [r["step"] for r in recs]
            defects = [r["defect_median"] for r in recs]
            ls = SEED_STYLES[seed]

            # Defect on left axis (gray/black)
            ax.plot(steps, defects, color="k", linewidth=1.5, alpha=0.7,
                    linestyle=ls, label=f"defect s={seed}" if row == 0 else "")

            # Per-task test accuracy on right axis
            for task in TASK_NAMES:
                accs = [r[f"test_{task}"] for r in recs]
                ax2.plot(steps, accs, color=TASK_COLORS[task], linewidth=1.2,
                         linestyle=ls, alpha=0.7,
                         label=f"{task} s={seed}" if row == 0 else "")

            # Mark spike step
            spike = find_spike_step(recs)
            if spike is not None:
                ax.axvline(x=spike, color="gray", linestyle=":", alpha=0.4, linewidth=1)

        # Mark per-task grok regions
        for task in TASK_NAMES:
            grok_steps = []
            for seed in SEEDS:
                key = (wd, seed)
                if key not in all_runs:
                    continue
                gs = all_runs[key]["grok_step"].get(task)
                if gs is not None:
                    grok_steps.append(gs)
            if grok_steps:
                ax.axvspan(min(grok_steps) - 100, max(grok_steps) + 100,
                           alpha=0.08, color=TASK_COLORS[task])

        ax.set_yscale("log")
        ax.set_ylabel("Commutator defect", fontsize=10)
        ax2.set_ylabel("Test accuracy", fontsize=10)
        ax2.set_ylim(-0.05, 1.1)
        ax.set_xlabel("Training step")
        ax.set_title(f"WD = {wd}", fontsize=12, fontweight="bold")
        ax.grid(alpha=0.2)

    # Global legend
    handles = []
    handles.append(Line2D([0], [0], color="k", linewidth=2, label="Defect (left axis)"))
    for task in TASK_NAMES:
        handles.append(Line2D([0], [0], color=TASK_COLORS[task], linewidth=2,
                              label=f"{task} test acc (right)"))
    handles.append(Line2D([0], [0], color="gray", linewidth=1, linestyle=":",
                          label="Defect spike"))

    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=10,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Tri-Task: Commutator Defect Predicts Per-Task Grokking",
                 fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "figTT_W_defect_predicts_grokking.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved figTT_W_defect_predicts_grokking.png")


def plot_figX(all_runs, lead_data):
    """
    FigTT_X: (a) scatter of spike_step vs grok_step, (b) bar chart of lead
    times by task, (c) lead times by WD.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # ── Panel A: Scatter — spike vs grok (all WD, all tasks) ──────────
    ax = axes[0]
    for entry in lead_data:
        marker = "o" if entry["wd"] == 1.0 else ("s" if entry["wd"] == 0.5 else "^")
        ax.scatter(entry["spike_step"], entry["grok_step"],
                   color=TASK_COLORS[entry["task"]], marker=marker,
                   s=70, alpha=0.8, edgecolors="k", linewidth=0.5)

    # Diagonal
    all_vals = [e["spike_step"] for e in lead_data] + [e["grok_step"] for e in lead_data]
    if all_vals:
        lo, hi = min(all_vals) - 500, max(all_vals) + 500
        ax.plot([lo, hi], [lo, hi], "k--", alpha=0.3, label="spike = grok")
        ax.set_xlim(max(0, lo), hi)
        ax.set_ylim(max(0, lo), hi)

    ax.set_xlabel("Defect spike step", fontsize=11)
    ax.set_ylabel("Grok step (90% test acc)", fontsize=11)
    ax.set_title("Defect Spike vs Grokking Step", fontsize=12)
    ax.text(0.05, 0.92, "← spike precedes grok",
            transform=ax.transAxes, fontsize=9, color="green", alpha=0.7)
    ax.text(0.6, 0.08, "spike follows grok →",
            transform=ax.transAxes, fontsize=9, color="red", alpha=0.7)

    # Legend for markers
    for task in TASK_NAMES:
        ax.scatter([], [], color=TASK_COLORS[task], s=50, label=task)
    for wd, marker in [(1.0, "o"), (0.5, "s"), (0.1, "^")]:
        ax.scatter([], [], color="gray", marker=marker, s=50, label=f"wd={wd}")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)

    # ── Panel B: Bar chart of lead times by TASK ──────────────────────
    ax = axes[1]
    task_leads = {t: [] for t in TASK_NAMES}
    for entry in lead_data:
        task_leads[entry["task"]].append(entry["lead_time"])

    x_pos = np.arange(len(TASK_NAMES))
    means = [np.mean(task_leads[t]) if task_leads[t] else 0 for t in TASK_NAMES]
    stds = [np.std(task_leads[t]) if len(task_leads[t]) > 1 else 0 for t in TASK_NAMES]

    bars = ax.bar(x_pos, means, yerr=stds,
                  color=[TASK_COLORS[t] for t in TASK_NAMES],
                  alpha=0.8, capsize=5, edgecolor="k", linewidth=0.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(TASK_NAMES, fontsize=11)
    ax.set_ylabel("Lead time (steps)", fontsize=11)
    ax.set_title("Lead Time by Task\n(positive = spike precedes grok)", fontsize=12)
    ax.axhline(y=0, color="k", linestyle="-", linewidth=0.5)
    ax.grid(alpha=0.3, axis="y")

    # Individual data points
    for i, task in enumerate(TASK_NAMES):
        for j, lead in enumerate(task_leads[task]):
            jitter = (j - len(task_leads[task]) / 2) * 0.06
            ax.scatter(i + jitter, lead, color="black", s=20, zorder=5, alpha=0.7)

    # ── Panel C: Bar chart of lead times by WD ────────────────────────
    ax = axes[2]
    wd_leads = {wd: [] for wd in WD_VALUES}
    for entry in lead_data:
        wd_leads[entry["wd"]].append(entry["lead_time"])

    wd_with_data = [wd for wd in WD_VALUES if wd_leads[wd]]
    x_pos = np.arange(len(wd_with_data))
    means = [np.mean(wd_leads[wd]) for wd in wd_with_data]
    stds = [np.std(wd_leads[wd]) if len(wd_leads[wd]) > 1 else 0 for wd in wd_with_data]

    bars = ax.bar(x_pos, means, yerr=stds,
                  color=[WD_COLORS[wd] for wd in wd_with_data],
                  alpha=0.8, capsize=5, edgecolor="k", linewidth=0.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"wd={wd}" for wd in wd_with_data], fontsize=11)
    ax.set_ylabel("Lead time (steps)", fontsize=11)
    ax.set_title("Lead Time by Weight Decay", fontsize=12)
    ax.axhline(y=0, color="k", linestyle="-", linewidth=0.5)
    ax.grid(alpha=0.3, axis="y")

    for i, wd in enumerate(wd_with_data):
        for j, lead in enumerate(wd_leads[wd]):
            jitter = (j - len(wd_leads[wd]) / 2) * 0.04
            ax.scatter(i + jitter, lead, color="black", s=20, zorder=5, alpha=0.7)

    # Sign test annotation
    all_leads = [e["lead_time"] for e in lead_data]
    if all_leads:
        n_pos = sum(1 for l in all_leads if l > 0)
        n_tot = len(all_leads)
        p_val = 2 ** (-n_tot)
        ax.text(0.98, 0.95,
                f"Sign test: {n_pos}/{n_tot} positive\n"
                f"p = 2$^{{-{n_tot}}}$",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=9, bbox=dict(boxstyle="round,pad=0.3",
                                      facecolor="lightyellow", edgecolor="gray"))

    fig.suptitle("Tri-Task: Commutator Defect as Early Warning Signal",
                 fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "figTT_X_defect_lead_time.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved figTT_X_defect_lead_time.png")


def plot_figW2(all_runs, lead_data):
    """
    FigTT_W2: Single hero figure showing the best lead-time example with
    annotated spike → grok interval.
    """
    if not lead_data:
        return

    # Pick example with clearest lead time
    best = max(lead_data, key=lambda e: e["lead_time"])
    wd, seed, task = best["wd"], best["seed"], best["task"]
    data = all_runs[(wd, seed)]
    recs = data["records"]

    steps = [r["step"] for r in recs]
    defects = [r["defect_median"] for r in recs]
    defect_25 = [r["defect_p25"] for r in recs]
    defect_75 = [r["defect_p75"] for r in recs]
    test_accs = [r[f"test_{task}"] for r in recs]
    train_accs = [r[f"train_{task}"] for r in recs]

    spike = best["spike_step"]
    grok = best["grok_step"]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax2 = ax.twinx()

    # Defect with IQR ribbon
    ax.fill_between(steps, defect_25, defect_75, alpha=0.15, color="#1a5276")
    ax.plot(steps, defects, color="#1a5276", linewidth=2.5,
            label="Commutator defect")

    # Accuracies on right axis
    ax2.plot(steps, test_accs, color="#e74c3c", linewidth=2.5,
             linestyle="--", label=f"Test acc ({task})")
    ax2.plot(steps, train_accs, color="#e74c3c", linewidth=1.5,
             linestyle=":", alpha=0.5, label=f"Train acc ({task})")

    # Mark spike and grok points
    ax.axvline(x=spike, color="#1a5276", linestyle=":", linewidth=2,
               alpha=0.7, label=f"Defect spike (step {spike})")
    ax.axvline(x=grok, color="#e74c3c", linestyle=":", linewidth=2,
               alpha=0.7, label=f"90% test acc (step {grok})")

    # Annotate the lead time
    mid = (spike + grok) / 2
    ymax = ax.get_ylim()[1]
    ax.annotate("", xy=(spike, ymax * 0.7),
                xytext=(grok, ymax * 0.7),
                arrowprops=dict(arrowstyle="<->", color="black", linewidth=1.5))
    ax.text(mid, ymax * 0.8,
            f"Δ = {grok - spike} steps",
            ha="center", fontsize=11, fontweight="bold")

    ax.set_yscale("log")
    ax.set_xlabel("Training step", fontsize=12)
    ax.set_ylabel("Commutator defect", fontsize=12, color="#1a5276")
    ax.tick_params(axis="y", labelcolor="#1a5276")
    ax2.set_ylabel("Accuracy", fontsize=12, color="#e74c3c")
    ax2.tick_params(axis="y", labelcolor="#e74c3c")
    ax2.set_ylim(-0.05, 1.1)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="center left")
    ax.grid(alpha=0.2)

    lead = grok - spike
    fig.suptitle(f"Tri-Task Hero: Defect Predicts Grokking ({task}, wd={wd}, s={seed})\n"
                 f"(spike at step {spike}, grok at step {grok}; lead = {lead} steps)",
                 fontsize=13, y=1.03)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "figTT_W2_hero_defect_predicts_grok.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved figTT_W2_hero_defect_predicts_grok.png "
          f"(best: {task} wd={wd} s={seed}, lead={lead})")


def plot_figW3(all_runs, lead_data):
    """
    FigTT_W3: Cross-WD comparison — defect time-series + per-task grok markers
    overlaid across WD values for one seed.
    """
    seed = SEEDS[0]  # Use first seed for clarity

    wd_with_data = [wd for wd in WD_VALUES if (wd, seed) in all_runs]
    if len(wd_with_data) < 2:
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # Panel A: defect time-series across WD
    ax = axes[0]
    for wd in wd_with_data:
        data = all_runs[(wd, seed)]
        recs = data["records"]
        steps = [r["step"] for r in recs]
        defs = [r["defect_median"] for r in recs]
        ax.plot(steps, defs, color=WD_COLORS[wd], lw=2, label=f"wd={wd}")

        # Mark per-task grok steps
        for task in TASK_NAMES:
            gs = data["grok_step"].get(task)
            if gs is not None:
                ax.axvline(gs, color=WD_COLORS[wd], ls=":", alpha=0.4, lw=1)

    ax.set_yscale("log")
    ax.set_xlabel("Training step", fontsize=11)
    ax.set_ylabel("Commutator defect", fontsize=11)
    ax.set_title(f"Defect Across WD (seed={seed})")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # Panel B: fractional lead time (lead / grok_step) by WD
    ax = axes[1]
    for task in TASK_NAMES:
        fracs_by_wd = {}
        for entry in lead_data:
            if entry["task"] == task and entry["grok_step"] > 0:
                wd = entry["wd"]
                frac = entry["lead_time"] / entry["grok_step"]
                fracs_by_wd.setdefault(wd, []).append(frac)

        wds_plot = sorted(fracs_by_wd.keys())
        means = [np.mean(fracs_by_wd[wd]) for wd in wds_plot]
        stds = [np.std(fracs_by_wd[wd]) if len(fracs_by_wd[wd]) > 1 else 0
                for wd in wds_plot]
        ax.errorbar(wds_plot, means, yerr=stds, marker="o", lw=2,
                    color=TASK_COLORS[task], label=task, capsize=4)

    ax.set_xlabel("Weight Decay", fontsize=11)
    ax.set_ylabel("Fractional lead time (lead / grok_step)", fontsize=11)
    ax.set_title("Fractional Lead Time vs Weight Decay")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_ylim(bottom=0)

    fig.suptitle("Tri-Task: Cross-WD Defect Onset Analysis",
                 fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "figTT_W3_cross_wd_lead_time.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved figTT_W3_cross_wd_lead_time.png")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    PLOT_DIR.mkdir(exist_ok=True)
    device = get_device()
    print(f"Device: {device}")

    # ── Load cached results ───────────────────────────────────────────────
    all_runs = {}
    if CACHE_PATH.exists():
        cached = torch.load(CACHE_PATH, map_location="cpu", weights_only=False)
        if "all_runs" in cached:
            all_runs = cached["all_runs"]
            print(f"  Loaded {len(all_runs)} cached runs")

    # ── Run all WD × seeds ───────────────────────────────────────────────
    all_wd = WD_VALUES + [WD_CONTROL]
    total_runs = len(all_wd) * len(SEEDS)
    run_i = 0

    for wd in all_wd:
        for seed in SEEDS:
            run_i += 1
            key = (wd, seed)
            tag = f"wd={wd}_s{seed}"

            if key in all_runs:
                print(f"\n  [{run_i}/{total_runs}] {tag} — cached, skipping")
                continue

            print(f"\n  [{run_i}/{total_runs}] {tag}")
            data = train_with_defect_tracking(wd, seed)
            all_runs[key] = data

            print(f"    → grokked={data['grokked']} "
                  f"(steps={data['grok_step']}), "
                  f"{len(data['records'])} measurements")

            # Save incrementally
            torch.save({"all_runs": all_runs}, CACHE_PATH)

    # ── Compute lead times ────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("  DEFECT SPIKE vs GROKKING TIMING (TRI-TASK)")
    print(f"{'='*80}")
    print(f"  {'WD':>5s}  {'Seed':>5s}  {'Task':>5s}  {'spike':>8s}  "
          f"{'grok90%':>8s}  {'lead':>8s}  {'frac':>8s}")

    lead_data = []

    for wd in WD_VALUES:
        for seed in SEEDS:
            key = (wd, seed)
            if key not in all_runs:
                continue
            data = all_runs[key]
            recs = data["records"]
            spike = find_spike_step(recs)

            for task in TASK_NAMES:
                grok = find_grok_step_from_records(recs, task, threshold=0.9)
                lead = None
                frac = None
                if spike is not None and grok is not None:
                    lead = grok - spike
                    frac = lead / grok if grok > 0 else None
                    lead_data.append({
                        "wd": wd, "seed": seed, "task": task,
                        "spike_step": spike, "grok_step": grok,
                        "lead_time": lead,
                    })

                print(f"  {wd:5.1f}  {seed:5d}  {task:>5s}  "
                      f"{str(spike):>8s}  {str(grok):>8s}  "
                      f"{str(lead):>8s}  "
                      f"{f'{frac:.3f}' if frac is not None else 'N/A':>8s}")

    # Summary statistics
    if lead_data:
        all_leads = [e["lead_time"] for e in lead_data]
        n_pos = sum(1 for l in all_leads if l > 0)
        n_tot = len(all_leads)
        p_val = 2 ** (-n_tot)

        print(f"\n  Lead time stats:")
        print(f"    mean  = {np.mean(all_leads):.0f} steps")
        print(f"    median = {np.median(all_leads):.0f} steps")
        print(f"    range = [{min(all_leads)}, {max(all_leads)}] steps")
        print(f"\n  Sign test: {n_pos}/{n_tot} positive, "
              f"p = 2^{{-{n_tot}}} = {p_val:.2e}")

        # Per-task
        for task in TASK_NAMES:
            t_leads = [e["lead_time"] for e in lead_data if e["task"] == task]
            if t_leads:
                print(f"    {task}: mean={np.mean(t_leads):.0f}, "
                      f"median={np.median(t_leads):.0f}, "
                      f"range=[{min(t_leads)}, {max(t_leads)}]")

        # Per-WD
        for wd in WD_VALUES:
            w_leads = [e["lead_time"] for e in lead_data if e["wd"] == wd]
            if w_leads:
                print(f"    wd={wd}: mean={np.mean(w_leads):.0f}, "
                      f"median={np.median(w_leads):.0f}, "
                      f"range=[{min(w_leads)}, {max(w_leads)}]")

    # ── Figures ───────────────────────────────────────────────────────────
    print("\n  Generating figures...")
    plot_figW(all_runs)
    if lead_data:
        plot_figX(all_runs, lead_data)
        plot_figW2(all_runs, lead_data)
        plot_figW3(all_runs, lead_data)

    # ── Save ──────────────────────────────────────────────────────────────
    torch.save({
        "all_runs": all_runs,
        "lead_data": lead_data,
    }, CACHE_PATH)
    print(f"\n  Saved results to {CACHE_PATH}")
    print("\nDone.")


if __name__ == "__main__":
    main()
