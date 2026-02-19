#!/usr/bin/env python3
"""
Hessian bottom-eigenvalue analysis for tri-task grokking.

Adapts modadd_modmul/hessian_analysis.py for the 3-task setting:
  (x+y) mod p, (x*y) mod p, (x^2+y^2) mod p

For each checkpoint, compute bottom-k Hessian eigenvalues via Lanczos
iteration (Hessian-vector products, no full Hessian materialized).

Four loss modes:
  - "total": CE(add) + CE(mul) + CE(sq)
  - "add":   CE(add) only
  - "mul":   CE(mul) only
  - "sq":    CE(sq)  only

Produces:
  figTT_H1 — Bottom eigenvalues over training (all 4 modes)
  figTT_H2 — Per-task bottom eigenvalue comparison (add vs mul vs sq)
  figTT_H3 — Grok vs no-WD control comparison
  figTT_H4 — Eigenvalue gap / ratio analysis
  figTT_H5 — Bottom eigenvalue onset detection vs grok step (cross-seed)
"""

import sys, random, time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from train_tritask import (
    TriTaskConfig, TriTaskTransformer, build_dataset, sample_batch,
    get_device, extract_attn_matrices, eval_accuracy,
)

RESULTS_DIR = Path(__file__).parent / "results"
PLOT_DIR = Path(__file__).parent / "plots"
TASK_NAMES = ["add", "mul", "sq"]
TASK_COLORS = {"add": "#1f77b4", "mul": "#d62728", "sq": "#2ca02c"}
SEEDS = [42, 137, 2024]
WD_VALUES = [1.0, 0.5, 0.1, 0.0]
WD_COLORS = {1.0: "#2ca02c", 0.5: "#1f77b4", 0.1: "#ff7f0e", 0.0: "#d62728"}


# ═══════════════════════════════════════════════════════════════════════════
# Hessian-vector product & Lanczos
# ═══════════════════════════════════════════════════════════════════════════

def hessian_vector_product(model, loss_fn_closure, v):
    """
    Compute Hv where H = d^2L/d theta^2 using two backward passes.
    loss_fn_closure: callable returning scalar loss (called once).
    v: flat vector same size as params.
    """
    params = [p for p in model.parameters() if p.requires_grad]

    # First backward: get gradients with computation graph
    loss = loss_fn_closure()
    grads = torch.autograd.grad(loss, params, create_graph=True, allow_unused=True)
    grads = [g if g is not None else torch.zeros_like(p) for g, p in zip(grads, params)]
    flat_grad = torch.cat([g.flatten() for g in grads])

    # grad^T v
    grad_v = flat_grad @ v

    # Second backward: d(grad^T v)/d theta = Hv
    hvp_list = torch.autograd.grad(grad_v, params, retain_graph=False, allow_unused=True)
    hvp_list = [h if h is not None else torch.zeros_like(p) for h, p in zip(hvp_list, params)]
    hvp = torch.cat([h.flatten() for h in hvp_list])
    return hvp.detach()


def lanczos_bottom_k(model, loss_fn_closure, k=5, n_iter=50, device="cpu"):
    """
    Lanczos iteration to find bottom-k eigenvalues of the Hessian.
    Uses implicit Hessian-vector products (no full Hessian materialized).
    Returns: eigenvalues (sorted ascending, bottom-k)
    """
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    q = torch.randn(n_params, device=device)
    q = q / q.norm()

    alphas = []
    betas = [0.0]
    Q = [torch.zeros(n_params, device=device)]  # placeholder q_{-1}
    Q.append(q)

    for j in range(1, n_iter + 1):
        w = hessian_vector_product(model, loss_fn_closure, Q[j])

        alpha_j = (w @ Q[j]).item()
        alphas.append(alpha_j)

        w = w - alpha_j * Q[j] - betas[-1] * Q[j-1]

        # Reorthogonalize
        for qi in Q[1:j+1]:
            w = w - (w @ qi) * qi

        beta_j = w.norm().item()

        if beta_j < 1e-10:
            while len(alphas) < n_iter:
                alphas.append(alphas[-1])
                betas.append(0.0)
            break

        betas.append(beta_j)
        Q.append(w / beta_j)

    # Build tridiagonal and diagonalize
    m = len(alphas)
    T = torch.zeros(m, m)
    for i in range(m):
        T[i, i] = alphas[i]
        if i > 0:
            T[i, i-1] = betas[i]
            T[i-1, i] = betas[i]

    eigenvalues, _ = torch.linalg.eigh(T)
    return eigenvalues[:k].cpu().numpy()


def compute_hessian_eigs_at_checkpoint(model, sd, train_pairs, cfg, device,
                                        k=5, n_iter=60, mode="total"):
    """
    Load checkpoint, compute bottom-k Hessian eigenvalues.
    mode: "total", "add", "mul", "sq"

    NOTE: Forces CPU — MPS doesn't support create_graph=True.
    """
    hess_device = "cpu"

    model.load_state_dict(sd)
    model.to(hess_device)
    model.train()

    p = cfg.P
    n_hess = min(512, len(train_pairs))
    hess_idx = np.random.choice(len(train_pairs), n_hess, replace=False)
    hess_pairs = [train_pairs[i] for i in hess_idx]
    ab = torch.tensor(hess_pairs, device=hess_device)
    a_h, b_h = ab[:, 0], ab[:, 1]
    y_add_h = (a_h + b_h) % p
    y_mul_h = (a_h * b_h) % p
    y_sq_h  = (a_h * a_h + b_h * b_h) % p

    def loss_closure():
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            logits_add, logits_mul, logits_sq = model(a_h, b_h)
        if mode == "total":
            return (F.cross_entropy(logits_add, y_add_h) +
                    F.cross_entropy(logits_mul, y_mul_h) +
                    F.cross_entropy(logits_sq, y_sq_h))
        elif mode == "add":
            return F.cross_entropy(logits_add, y_add_h)
        elif mode == "mul":
            return F.cross_entropy(logits_mul, y_mul_h)
        elif mode == "sq":
            return F.cross_entropy(logits_sq, y_sq_h)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    eigs = lanczos_bottom_k(model, loss_closure, k=k, n_iter=n_iter, device=hess_device)
    return eigs


# ═══════════════════════════════════════════════════════════════════════════
# Main analysis
# ═══════════════════════════════════════════════════════════════════════════

def analyze_hessian(data, label, seed, k=5, n_iter=60):
    """Run Hessian analysis on checkpoints for one run."""
    device = "cpu"
    cfg_dict = data["cfg"]
    cfg = TriTaskConfig(**cfg_dict)

    checkpoints = data["checkpoints"]
    train_pairs, _ = build_dataset(cfg.P, cfg.TRAIN_FRACTION, cfg.SEED)

    # Subsample checkpoints — Hessian is expensive
    max_ckpts = 30
    if len(checkpoints) > max_ckpts:
        idx = np.linspace(0, len(checkpoints)-1, max_ckpts, dtype=int)
        checkpoints = [checkpoints[i] for i in idx]
    print(f"  Analyzing {len(checkpoints)} checkpoints for {label}")

    model = TriTaskTransformer(cfg).to(device)

    results = {"total": [], "add": [], "mul": [], "sq": [], "steps": []}

    for ci, (step, sd) in enumerate(checkpoints):
        results["steps"].append(step)

        for mode in ["total", "add", "mul", "sq"]:
            eigs = compute_hessian_eigs_at_checkpoint(
                model, sd, train_pairs, cfg, device,
                k=k, n_iter=n_iter, mode=mode
            )
            results[mode].append(eigs)

        if (ci+1) % 5 == 0 or ci == len(checkpoints) - 1:
            eigs_t = results["total"][-1]
            eigs_a = results["add"][-1]
            eigs_m = results["mul"][-1]
            eigs_s = results["sq"][-1]
            print(f"    ckpt {ci+1}/{len(checkpoints)}: step={step}, "
                  f"lam_min(total)={eigs_t[0]:.4f}, "
                  f"lam_min(add)={eigs_a[0]:.4f}, "
                  f"lam_min(mul)={eigs_m[0]:.4f}, "
                  f"lam_min(sq)={eigs_s[0]:.4f}")

    for mode in ["total", "add", "mul", "sq"]:
        results[mode] = np.array(results[mode])
    results["steps"] = np.array(results["steps"])

    return results


def find_negative_onset(steps, eig_trace, threshold=-0.01):
    """
    First step where bottom eigenvalue drops below threshold
    and stays below for >= 3 consecutive measurements.
    """
    below = eig_trace < threshold
    count = 0
    for i, b in enumerate(below):
        if b:
            count += 1
            if count >= 3:
                return steps[i - 2]
        else:
            count = 0
    return None


# ═══════════════════════════════════════════════════════════════════════════
# Figures
# ═══════════════════════════════════════════════════════════════════════════

def plot_hessian_figures(grok_results, control_results, grok_data, control_data, seed):

    grok_step = grok_data.get("grok_step", {})

    # ── Figure H1: Bottom eigenvalues over training (all 4 modes) ─────────
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))

    mode_specs = [
        ("total", "Total Loss (add+mul+sq)", "#1a5276"),
        ("add",   "Add Loss Only",           "#1f77b4"),
        ("mul",   "Mul Loss Only",           "#d62728"),
        ("sq",    "Sq Loss Only",            "#2ca02c"),
    ]

    for idx, (mode, title, color) in enumerate(mode_specs):
        ax = axes[idx]
        steps = grok_results["steps"]
        eigs = grok_results[mode]

        for ki in range(min(5, eigs.shape[1])):
            alpha = 1.0 if ki == 0 else 0.4
            lw = 2.5 if ki == 0 else 1.0
            ax.plot(steps, eigs[:, ki], color=color, alpha=alpha, lw=lw,
                    label=f"lam_{ki+1}" if ki < 3 else None)

        ax.axhline(0, color="gray", ls="--", alpha=0.5)
        for t in TASK_NAMES:
            gs = grok_step.get(t)
            if gs:
                ax.axvline(gs, color=TASK_COLORS[t], ls=":", alpha=0.5, label=f"{t} groks")
        ax.set_xlabel("Training step")
        ax.set_ylabel("Eigenvalue")
        ax.set_title(title)
        ax.legend(fontsize=6)
        ax.grid(alpha=0.3)

    fig.suptitle(f"Hessian Bottom Eigenvalues — Tri-Task Grokking (seed={seed})",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / f"figTT_H1_hessian_eigs_s{seed}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved figTT_H1_hessian_eigs_s{seed}.png")

    # ── Figure H2: Per-task comparison (bottom eig overlay) ───────────────
    fig, ax = plt.subplots(figsize=(14, 6))

    steps = grok_results["steps"]
    for task, color in TASK_COLORS.items():
        ax.plot(steps, grok_results[task][:, 0], color=color, lw=2.5,
                label=f"lam_min ({task} loss)")
    ax.plot(steps, grok_results["total"][:, 0], color="#1a5276", lw=1.5,
            ls="--", label="lam_min (total loss)")
    ax.axhline(0, color="gray", ls="--", alpha=0.5)

    # Detect onsets
    onsets = {}
    annotations = []
    for task in TASK_NAMES:
        onset = find_negative_onset(steps, grok_results[task][:, 0])
        onsets[task] = onset
        if onset is not None:
            ax.axvline(onset, color=TASK_COLORS[task], ls="-.", alpha=0.7, lw=1.5)
            annotations.append(f"{task} onset: {onset}")
    onset_total = find_negative_onset(steps, grok_results["total"][:, 0])
    onsets["total"] = onset_total

    for t in TASK_NAMES:
        gs = grok_step.get(t)
        if gs:
            ax.axvline(gs, color=TASK_COLORS[t], ls=":", alpha=0.4)
            annotations.append(f"{t} groks: {gs}")

    ax.text(0.02, 0.02, "\n".join(annotations), transform=ax.transAxes,
            fontsize=9, verticalalignment="bottom",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8))

    ax.set_xlabel("Training step", fontsize=12)
    ax.set_ylabel("Bottom Hessian eigenvalue", fontsize=12)
    ax.set_title(f"Per-Task Hessian Bottom Eigenvalues (seed={seed})\n"
                 f"(negative outlier = loss landscape bifurcation / saddle escape)",
                 fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(PLOT_DIR / f"figTT_H2_pertask_bottom_eig_s{seed}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved figTT_H2_pertask_bottom_eig_s{seed}.png")

    # ── Figure H3: Grok vs no-WD comparison ───────────────────────────────
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))

    for idx, (mode, title) in enumerate([
        ("total", "Total"), ("add", "Add"), ("mul", "Mul"), ("sq", "Sq")
    ]):
        ax = axes[idx]
        steps_g = grok_results["steps"]
        ax.plot(steps_g, grok_results[mode][:, 0], color="#2ca02c", lw=2.5,
                label="wd=1.0 (grok)")

        if control_results is not None:
            steps_c = control_results["steps"]
            ax.plot(steps_c, control_results[mode][:, 0], color="#d62728", lw=2,
                    ls="--", label="wd=0 (no grok)")

        ax.axhline(0, color="gray", ls="--", alpha=0.5)
        ax.set_xlabel("Training step")
        ax.set_ylabel("lam_min")
        ax.set_title(f"{title} Loss")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    fig.suptitle(f"Hessian: Grokking vs No-WD Control (seed={seed})",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / f"figTT_H3_grok_vs_nowd_s{seed}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved figTT_H3_grok_vs_nowd_s{seed}.png")

    # ── Figure H4: Eigenvalue gap / ratio analysis ────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1: |lam2|/|lam1| ratio (spectral gap)
    ax = axes[0]
    for mode, color, label in [("total", "#1a5276", "Total"),
                                ("add", "#1f77b4", "Add"),
                                ("mul", "#d62728", "Mul"),
                                ("sq", "#2ca02c", "Sq")]:
        steps = grok_results["steps"]
        eigs = grok_results[mode]
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.abs(eigs[:, 1]) / (np.abs(eigs[:, 0]) + 1e-12)
        ax.plot(steps, ratio, color=color, lw=2, label=label)
    ax.set_xlabel("Training step")
    ax.set_ylabel("|lam_2| / |lam_1|")
    ax.set_title("Spectral Gap")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 2)

    # Panel 2: Pairwise eigenvalue differences
    ax = axes[1]
    steps = grok_results["steps"]
    pairs = [("add", "mul"), ("add", "sq"), ("mul", "sq")]
    pair_colors = ["#9467bd", "#ff7f0e", "#8c564b"]
    for (t1, t2), pc in zip(pairs, pair_colors):
        diff = grok_results[t1][:, 0] - grok_results[t2][:, 0]
        ax.plot(steps, diff, color=pc, lw=2, label=f"lam_min({t1}) - lam_min({t2})")
    ax.axhline(0, color="gray", ls="--", alpha=0.5)
    for t in TASK_NAMES:
        gs = grok_step.get(t)
        if gs:
            ax.axvline(gs, color=TASK_COLORS[t], ls=":", alpha=0.5)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Eigenvalue difference")
    ax.set_title("Per-Task Eigenvalue Differences")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Panel 3: Cumulative bottom-5 sum
    ax = axes[2]
    for mode, color, label in [("total", "#1a5276", "Total"),
                                ("add", "#1f77b4", "Add"),
                                ("mul", "#d62728", "Mul"),
                                ("sq", "#2ca02c", "Sq")]:
        steps = grok_results["steps"]
        eigs = grok_results[mode]
        cumsum = eigs.sum(axis=1)
        ax.plot(steps, cumsum, color=color, lw=2, label=label)

    if control_results is not None:
        steps_c = control_results["steps"]
        cumsum_c = control_results["total"].sum(axis=1)
        ax.plot(steps_c, cumsum_c, color="gray", lw=1.5, ls="--", label="control total")

    ax.axhline(0, color="gray", ls="--", alpha=0.3)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Sum of bottom-5 eigenvalues")
    ax.set_title("Cumulative Bottom Eigenvalue Sum")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.suptitle(f"Hessian Eigenvalue Ratio Analysis (seed={seed})", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / f"figTT_H4_ratio_analysis_s{seed}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved figTT_H4_ratio_analysis_s{seed}.png")

    return {
        "onset_add": onsets.get("add"),
        "onset_mul": onsets.get("mul"),
        "onset_sq":  onsets.get("sq"),
        "onset_total": onset_total,
        "grok_step_add": grok_step.get("add"),
        "grok_step_mul": grok_step.get("mul"),
        "grok_step_sq":  grok_step.get("sq"),
    }


def plot_onset_summary(all_onsets):
    """Cross-seed onset summary figure."""
    fig, ax = plt.subplots(figsize=(12, 6))

    seeds = sorted(all_onsets.keys())
    x = np.arange(len(seeds))
    width = 0.12

    bars = [
        ("onset_add", "#1f77b4", "lam_min(add) onset"),
        ("onset_mul", "#d62728", "lam_min(mul) onset"),
        ("onset_sq",  "#2ca02c", "lam_min(sq) onset"),
        ("grok_step_add", "#aec7e8", "add groks"),
        ("grok_step_mul", "#ff9896", "mul groks"),
        ("grok_step_sq",  "#98df8a", "sq groks"),
    ]

    for i, (key, color, label) in enumerate(bars):
        vals = [all_onsets[s].get(key, 0) or 0 for s in seeds]
        ax.bar(x + i * width, vals, width, label=label, color=color, alpha=0.85)

    ax.set_xlabel("Seed")
    ax.set_ylabel("Training step")
    ax.set_title("Hessian Negative Eigenvalue Onset vs Grokking Step")
    ax.set_xticks(x + 2.5 * width)
    ax.set_xticklabels([f"s{s}" for s in seeds])
    ax.legend(fontsize=8, ncol=2)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(PLOT_DIR / "figTT_H5_onset_summary.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved figTT_H5_onset_summary.png")


def plot_cross_wd_comparison(all_wd_results):
    """
    Cross-WD comparison: how bottom eigenvalue evolves for each WD value.
    One figure per seed, with all WD values overlaid.
    """
    for seed in SEEDS:
        seed_data = {wd: all_wd_results.get((wd, seed)) for wd in WD_VALUES}
        seed_data = {wd: d for wd, d in seed_data.items() if d is not None}
        if len(seed_data) < 2:
            continue

        # ── Figure H6: Total loss bottom eigenvalue across WD values ──────
        fig, axes = plt.subplots(1, 4, figsize=(24, 5))

        for idx, mode in enumerate(["total", "add", "mul", "sq"]):
            ax = axes[idx]
            for wd, res in sorted(seed_data.items(), reverse=True):
                steps = res["steps"]
                eigs = res[mode][:, 0]  # bottom eigenvalue
                ax.plot(steps, eigs, color=WD_COLORS[wd], lw=2,
                        label=f"wd={wd}", alpha=0.85)
            ax.axhline(0, color="gray", ls="--", alpha=0.5)
            ax.set_xlabel("Training step")
            ax.set_ylabel("lam_min")
            ax.set_title(f"{mode.capitalize()} Loss")
            ax.legend(fontsize=9)
            ax.grid(alpha=0.3)

        fig.suptitle(f"Bottom Hessian Eigenvalue Across Weight Decay (seed={seed})",
                     fontsize=13, y=1.02)
        fig.tight_layout()
        fig.savefig(PLOT_DIR / f"figTT_H6_cross_wd_s{seed}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved figTT_H6_cross_wd_s{seed}.png")

    # ── Figure H7: Mean bottom eigenvalue (post-memorization) vs WD ───────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for si, seed in enumerate(SEEDS):
        ax = axes[si]
        for mode, color in [("total", "#1a5276"), ("add", "#1f77b4"),
                             ("mul", "#d62728"), ("sq", "#2ca02c")]:
            wd_vals_plot, mean_vals, std_vals = [], [], []
            for wd in WD_VALUES:
                res = all_wd_results.get((wd, seed))
                if res is None:
                    continue
                # Use second half of training as "post-memorization"
                n = len(res["steps"])
                half = n // 2
                if half < 3:
                    continue
                vals = res[mode][half:, 0]
                wd_vals_plot.append(wd)
                mean_vals.append(np.mean(vals))
                std_vals.append(np.std(vals))
            if wd_vals_plot:
                ax.errorbar(wd_vals_plot, mean_vals, yerr=std_vals,
                           marker="o", color=color, lw=2, capsize=4, label=mode)
        ax.axhline(0, color="gray", ls="--", alpha=0.5)
        ax.set_xlabel("Weight Decay")
        ax.set_ylabel("Mean lam_min (2nd half)")
        ax.set_title(f"seed={seed}")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    fig.suptitle("Mean Bottom Eigenvalue vs Weight Decay (post-memorization)",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "figTT_H7_wd_vs_eigval.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved figTT_H7_wd_vs_eigval.png")

    # ── Figure H8: Bottom-5 eigenvalue sum across WD ─────────────────────
    fig, axes = plt.subplots(1, len(SEEDS), figsize=(7*len(SEEDS), 5))

    for si, seed in enumerate(SEEDS):
        ax = axes[si]
        for wd in WD_VALUES:
            res = all_wd_results.get((wd, seed))
            if res is None:
                continue
            steps = res["steps"]
            cumsum = res["total"].sum(axis=1)
            ax.plot(steps, cumsum, color=WD_COLORS[wd], lw=2, label=f"wd={wd}")
        ax.axhline(0, color="gray", ls="--", alpha=0.3)
        ax.set_xlabel("Training step")
        ax.set_ylabel("Sum of bottom-5 eigenvalues")
        ax.set_title(f"seed={seed}")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    fig.suptitle("Cumulative Bottom Eigenvalue Sum Across Weight Decay",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "figTT_H8_wd_cumsum.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved figTT_H8_wd_cumsum.png")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def wd_tag(wd):
    return f"wd{wd:.0f}" if wd == int(wd) else f"wd{wd}"


def main():
    PLOT_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    all_hessian = {}
    all_onsets = {}
    all_wd_results = {}   # keyed by (wd, seed)

    for wd in WD_VALUES:
        for seed in SEEDS:
            tag = wd_tag(wd)
            fpath = RESULTS_DIR / f"tritask_{tag}_s{seed}.pt"

            if not fpath.exists():
                print(f"[skip] {fpath.name} not found")
                continue

            print(f"\n{'='*70}")
            print(f"  Hessian analysis: wd={wd}, seed={seed}")
            print(f"{'='*70}")

            data = torch.load(fpath, map_location="cpu", weights_only=False)

            print(f"\n  [WD={wd}] Hessian eigenvalues...")
            hessian_res = analyze_hessian(data, f"{tag}_s{seed}", seed,
                                           k=5, n_iter=25)

            all_hessian[(wd, seed)] = hessian_res
            all_wd_results[(wd, seed)] = hessian_res

    # Per-seed original-style plots (wd=1.0 as grok, wd=0.0 as control)
    for seed in SEEDS:
        grok_h = all_hessian.get((1.0, seed))
        ctrl_h = all_hessian.get((0.0, seed))
        if grok_h is None:
            continue

        grok_path = RESULTS_DIR / f"tritask_wd1_s{seed}.pt"
        grok_data = torch.load(grok_path, map_location="cpu", weights_only=False)
        ctrl_data = None
        ctrl_path = RESULTS_DIR / f"tritask_wd0_s{seed}.pt"
        if ctrl_path.exists():
            ctrl_data = torch.load(ctrl_path, map_location="cpu", weights_only=False)

        print(f"\n  Generating per-seed figures (seed={seed})...")
        onsets = plot_hessian_figures(grok_h, ctrl_h, grok_data, ctrl_data, seed)
        all_onsets[seed] = onsets

    # Cross-seed summary
    if all_onsets:
        plot_onset_summary(all_onsets)

    # Cross-WD comparison figures
    print(f"\n  Generating cross-WD comparison figures...")
    plot_cross_wd_comparison(all_wd_results)

    # Save
    save_path = PLOT_DIR / "hessian_results.pt"
    torch.save({"hessian": all_hessian, "onsets": all_onsets,
                "wd_results": all_wd_results}, save_path)
    print(f"\n  Saved all results to {save_path}")

    # Print summary
    print(f"\n{'='*70}")
    print("  HESSIAN SUMMARY (ALL WD VALUES)")
    print(f"{'='*70}")
    header = (f"  {'wd':>5s}  {'seed':>5s}  {'onset_add':>10s}  {'onset_mul':>10s}  "
              f"{'onset_sq':>10s}  {'mean_lam':>10s}  {'grok_add':>10s}  "
              f"{'grok_mul':>10s}  {'grok_sq':>10s}")
    print(header)

    for wd in WD_VALUES:
        for seed in SEEDS:
            res = all_hessian.get((wd, seed))
            if res is None:
                continue

            # Onset
            onset_add = find_negative_onset(res["steps"], res["add"][:, 0])
            onset_mul = find_negative_onset(res["steps"], res["mul"][:, 0])
            onset_sq  = find_negative_onset(res["steps"], res["sq"][:, 0])

            # Mean bottom eigenvalue (2nd half)
            n = len(res["steps"])
            half = n // 2
            mean_lam = np.mean(res["total"][half:, 0]) if half > 0 else float("nan")

            # Grok steps
            tag = wd_tag(wd)
            fpath = RESULTS_DIR / f"tritask_{tag}_s{seed}.pt"
            data = torch.load(fpath, map_location="cpu", weights_only=False)
            grok_step = data.get("grok_step", {})

            print(f"  {wd:5.1f}  {seed:5d}  {str(onset_add):>10s}  "
                  f"{str(onset_mul):>10s}  {str(onset_sq):>10s}  "
                  f"{mean_lam:10.2f}  "
                  f"{str(grok_step.get('add')):>10s}  "
                  f"{str(grok_step.get('mul')):>10s}  "
                  f"{str(grok_step.get('sq')):>10s}")

    print("\nDone.")


if __name__ == "__main__":
    main()
