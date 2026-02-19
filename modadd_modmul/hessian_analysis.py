#!/usr/bin/env python3
"""
Hessian bottom-eigenvalue analysis for multi-task grokking.

1. Train a no-WD control (wd=0) for comparison
2. For each checkpoint (grok + control), compute bottom-k Hessian eigenvalues
   via Lanczos iteration (Hessian-vector products, no full Hessian)
3. Also compute per-task Hessian bottom eigs (add-only loss, mul-only loss)
4. Ratio analysis: scaling of eigenvalue gaps

Predictions to test:
  - Negative eigenvalue outlier appears first for one task
  - Second outlier for the other task
  - Gap between outlier onsets ≈ grokking delay

Produces:
  figMT_H1 — Bottom eigenvalues over training (total loss)
  figMT_H2 — Per-task bottom eigenvalues (add vs mul)
  figMT_H3 — Grok vs no-WD control comparison
  figMT_H4 — Eigenvalue gap / ratio analysis
  figMT_H5 — Bottom eigenvalue onset detection vs grok step
"""

import math, sys, random, time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from train_multitask import (
    MultiTaskConfig, MultiTaskTransformer, build_dataset, sample_batch,
    get_device, extract_attn_matrices, eval_accuracy,
)

RESULTS_DIR = Path(__file__).parent / "results"
PLOT_DIR = Path(__file__).parent / "plots"


# ═══════════════════════════════════════════════════════════════════════════
# Step 1: Train no-WD control
# ═══════════════════════════════════════════════════════════════════════════

def train_control(seed=42):
    """Train wd=0 control (shorter, just needs to memorize)."""
    out_path = RESULTS_DIR / f"multitask_nowd_s{seed}.pt"
    if out_path.exists():
        print(f"  [control] already exists: {out_path.name}")
        return torch.load(out_path, map_location="cpu", weights_only=False)

    device = get_device()
    cfg = MultiTaskConfig(
        SEED=seed,
        WEIGHT_DECAY=0.0,
        STEPS=50_000,           # shorter — won't grok
        CHECKPOINT_EVERY=500,   # coarser
        MODEL_LOG_EVERY=500,
    )

    torch.manual_seed(cfg.SEED)
    np.random.seed(cfg.SEED)
    random.seed(cfg.SEED)

    train_pairs, test_pairs = build_dataset(cfg.P, cfg.TRAIN_FRACTION, cfg.SEED)
    model = MultiTaskTransformer(cfg).to(device)
    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY,
        betas=(cfg.ADAM_BETA1, cfg.ADAM_BETA2)
    )
    loss_fn = nn.CrossEntropyLoss()

    init_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    attn_logs = [{"step": 0, "layers": extract_attn_matrices(model)}]
    checkpoints = [(0, {k: v.cpu().clone() for k, v in model.state_dict().items()})]
    metrics = []
    t0 = time.time()

    for step in range(1, cfg.STEPS + 1):
        model.train()
        a, b, y_add, y_mul = sample_batch(train_pairs, cfg.BATCH_SIZE, cfg.P, device)
        logits_add, logits_mul = model(a, b)
        loss = loss_fn(logits_add, y_add) + loss_fn(logits_mul, y_mul)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
        opt.step()

        if step % cfg.MODEL_LOG_EVERY == 0:
            attn_logs.append({"step": step, "layers": extract_attn_matrices(model)})
        if step % cfg.CHECKPOINT_EVERY == 0:
            checkpoints.append(
                (step, {k: v.cpu().clone() for k, v in model.state_dict().items()})
            )

        if step % cfg.EVAL_EVERY == 0 or step == 1:
            train_add, train_mul = eval_accuracy(model, train_pairs, cfg, device)
            test_add, test_mul = eval_accuracy(model, test_pairs, cfg, device)
            metrics.append({
                "step": step,
                "train_add": train_add, "train_mul": train_mul,
                "test_add": test_add, "test_mul": test_mul,
                "loss": loss.item(),
            })
            if step % 5000 == 0:
                elapsed = (time.time() - t0) / 60
                print(f"    [nowd] step {step:6d} | add: {train_add:.3f}/{test_add:.3f} | "
                      f"mul: {train_mul:.3f}/{test_mul:.3f} | {elapsed:.1f}m")

    result = {
        "cfg": asdict(cfg),
        "attn_logs": attn_logs,
        "checkpoints": checkpoints,
        "metrics": metrics,
        "grokked_add": False,
        "grokked_mul": False,
        "grok_step_add": None,
        "grok_step_mul": None,
        "final_step": step,
        "init_state": init_state,
    }
    torch.save(result, out_path)
    print(f"  [control] saved → {out_path.name} ({len(checkpoints)} ckpts)")
    return result


# ═══════════════════════════════════════════════════════════════════════════
# Hessian-vector product & Lanczos
# ═══════════════════════════════════════════════════════════════════════════

def hessian_vector_product(model, loss_fn_closure, v):
    """
    Compute Hv where H = d²L/dθ² using two backward passes.
    loss_fn_closure: callable that returns scalar loss (will be called twice).
    v: flat vector same size as params.
    """
    params = [p for p in model.parameters() if p.requires_grad]

    # First backward: get gradients
    loss = loss_fn_closure()
    grads = torch.autograd.grad(loss, params, create_graph=True, allow_unused=True)
    # Replace None grads (unused params) with zeros
    grads = [g if g is not None else torch.zeros_like(p) for g, p in zip(grads, params)]
    flat_grad = torch.cat([g.flatten() for g in grads])

    # Compute grad^T v
    grad_v = flat_grad @ v

    # Second backward: d(grad^T v)/dθ = Hv
    hvp_list = torch.autograd.grad(grad_v, params, retain_graph=False, allow_unused=True)
    hvp_list = [h if h is not None else torch.zeros_like(p) for h, p in zip(hvp_list, params)]
    hvp = torch.cat([h.flatten() for h in hvp_list])
    return hvp.detach()


def lanczos_bottom_k(model, loss_fn_closure, k=5, n_iter=50, device="cpu"):
    """
    Lanczos iteration to find bottom-k eigenvalues of the Hessian.
    Uses implicit Hessian-vector products (no full Hessian materialized).

    Returns: eigenvalues (sorted ascending), eigenvectors
    """
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Random starting vector
    q = torch.randn(n_params, device=device)
    q = q / q.norm()

    # Lanczos tridiagonal matrix
    alphas = []
    betas = [0.0]
    Q = [torch.zeros(n_params, device=device)]  # placeholder q_{-1}
    Q.append(q)

    for j in range(1, n_iter + 1):
        # Hv product
        w = hessian_vector_product(model, loss_fn_closure, Q[j])

        alpha_j = (w @ Q[j]).item()
        alphas.append(alpha_j)

        w = w - alpha_j * Q[j] - betas[-1] * Q[j-1]

        # Reorthogonalize (important for numerical stability)
        for qi in Q[1:j+1]:
            w = w - (w @ qi) * qi

        beta_j = w.norm().item()

        if beta_j < 1e-10:
            # Invariant subspace found — pad and break
            while len(alphas) < n_iter:
                alphas.append(alphas[-1])
                betas.append(0.0)
            break

        betas.append(beta_j)
        Q.append(w / beta_j)

    # Build tridiagonal matrix and diagonalize
    m = len(alphas)
    T = torch.zeros(m, m)
    for i in range(m):
        T[i, i] = alphas[i]
        if i > 0:
            T[i, i-1] = betas[i]
            T[i-1, i] = betas[i]

    eigenvalues, eigvecs_T = torch.linalg.eigh(T)

    # Return bottom-k
    return eigenvalues[:k].cpu().numpy()


def compute_hessian_eigs_at_checkpoint(model, sd, train_pairs, cfg, device,
                                        k=5, n_iter=60, mode="total"):
    """
    Load checkpoint, compute bottom-k Hessian eigenvalues.
    mode: "total" (add+mul), "add" (add only), "mul" (mul only)

    NOTE: Forces CPU for Hessian-vector products because MPS does not
    reliably support create_graph=True (second-order gradients).
    """
    # Force CPU for second-order gradient computation
    hess_device = "cpu"

    model.load_state_dict(sd)
    model.to(hess_device)
    model.train()

    p = cfg.P
    # Use a small subsample for Hessian (CPU speed tradeoff)
    # 512 samples is enough for stable eigenvalue estimates
    n_hess = min(512, len(train_pairs))
    hess_idx = np.random.choice(len(train_pairs), n_hess, replace=False)
    hess_pairs = [train_pairs[i] for i in hess_idx]
    ab = torch.tensor(hess_pairs, device=hess_device)
    a_h, b_h = ab[:, 0], ab[:, 1]
    y_add_h = (a_h + b_h) % p
    y_mul_h = (a_h * b_h) % p

    def loss_closure():
        # Disable flash/efficient attention — their backward doesn't support
        # create_graph=True (needed for Hessian-vector products)
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            logits_add, logits_mul = model(a_h, b_h)
        if mode == "total":
            return F.cross_entropy(logits_add, y_add_h) + F.cross_entropy(logits_mul, y_mul_h)
        elif mode == "add":
            return F.cross_entropy(logits_add, y_add_h)
        elif mode == "mul":
            return F.cross_entropy(logits_mul, y_mul_h)

    eigs = lanczos_bottom_k(model, loss_closure, k=k, n_iter=n_iter, device=hess_device)
    return eigs


# ═══════════════════════════════════════════════════════════════════════════
# Main analysis
# ═══════════════════════════════════════════════════════════════════════════

def analyze_hessian(data, label, seed, k=5, n_iter=60):
    """Run Hessian analysis on checkpoints for one run."""
    device = "cpu"  # Hessian must run on CPU for second-order grads
    cfg_dict = data["cfg"]
    cfg = MultiTaskConfig(**cfg_dict)

    checkpoints = data["checkpoints"]
    train_pairs, _ = build_dataset(cfg.P, cfg.TRAIN_FRACTION, cfg.SEED)

    # Subsample checkpoints — Hessian is expensive (HVP on CPU)
    max_ckpts = 30
    if len(checkpoints) > max_ckpts:
        idx = np.linspace(0, len(checkpoints)-1, max_ckpts, dtype=int)
        checkpoints = [checkpoints[i] for i in idx]
    print(f"  Analyzing {len(checkpoints)} checkpoints for {label}")

    model = MultiTaskTransformer(cfg).to(device)

    results = {"total": [], "add": [], "mul": [], "steps": []}

    for ci, (step, sd) in enumerate(checkpoints):
        results["steps"].append(step)

        for mode in ["total", "add", "mul"]:
            eigs = compute_hessian_eigs_at_checkpoint(
                model, sd, train_pairs, cfg, device,
                k=k, n_iter=n_iter, mode=mode
            )
            results[mode].append(eigs)

        if (ci+1) % 10 == 0 or ci == len(checkpoints) - 1:
            eigs_t = results["total"][-1]
            eigs_a = results["add"][-1]
            eigs_m = results["mul"][-1]
            print(f"    ckpt {ci+1}/{len(checkpoints)}: step={step}, "
                  f"λ_min(total)={eigs_t[0]:.4f}, "
                  f"λ_min(add)={eigs_a[0]:.4f}, "
                  f"λ_min(mul)={eigs_m[0]:.4f}")

    # Convert to arrays
    for mode in ["total", "add", "mul"]:
        results[mode] = np.array(results[mode])  # shape [n_ckpts, k]
    results["steps"] = np.array(results["steps"])

    return results


def find_negative_onset(steps, eig_trace, threshold=-0.01):
    """
    Find the first step where bottom eigenvalue drops below threshold
    and stays below for at least 3 consecutive measurements.
    """
    below = eig_trace < threshold
    count = 0
    for i, b in enumerate(below):
        if b:
            count += 1
            if count >= 3:
                return steps[i - 2]  # onset = first of the 3 consecutive
        else:
            count = 0
    return None


# ═══════════════════════════════════════════════════════════════════════════
# Figures
# ═══════════════════════════════════════════════════════════════════════════

def plot_hessian_figures(grok_results, control_results, grok_data, control_data, seed):

    # ── Figure H1: Bottom eigenvalues over training (total loss) ──────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for idx, (mode, title, color) in enumerate([
        ("total", "Total Loss (add+mul)", "#1a5276"),
        ("add", "Add Loss Only", "#1f77b4"),
        ("mul", "Mul Loss Only", "#d62728"),
    ]):
        ax = axes[idx]
        steps = grok_results["steps"]
        eigs = grok_results[mode]  # [n_ckpts, k]

        for ki in range(min(5, eigs.shape[1])):
            alpha = 1.0 if ki == 0 else 0.4
            lw = 2.5 if ki == 0 else 1.0
            ax.plot(steps, eigs[:, ki], color=color, alpha=alpha, lw=lw,
                    label=f"λ_{ki+1}" if ki < 3 else None)

        ax.axhline(0, color="gray", ls="--", alpha=0.5)
        if grok_data["grok_step_add"]:
            ax.axvline(grok_data["grok_step_add"], color="#1f77b4", ls=":", alpha=0.5, label="add groks")
        if grok_data["grok_step_mul"]:
            ax.axvline(grok_data["grok_step_mul"], color="#d62728", ls=":", alpha=0.5, label="mul groks")
        ax.set_xlabel("Training step")
        ax.set_ylabel("Eigenvalue")
        ax.set_title(title)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    fig.suptitle(f"Hessian Bottom Eigenvalues During Multi-Task Grokking (seed={seed})",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / f"figMT_H1_hessian_eigs_s{seed}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved figMT_H1_hessian_eigs_s{seed}.png")

    # ── Figure H2: Per-task comparison (add vs mul bottom eig) ────────────
    fig, ax = plt.subplots(figsize=(12, 6))

    steps = grok_results["steps"]
    ax.plot(steps, grok_results["add"][:, 0], color="#1f77b4", lw=2.5,
            label="λ_min (add loss)")
    ax.plot(steps, grok_results["mul"][:, 0], color="#d62728", lw=2.5,
            label="λ_min (mul loss)")
    ax.plot(steps, grok_results["total"][:, 0], color="#2ecc71", lw=1.5,
            ls="--", label="λ_min (total loss)")
    ax.axhline(0, color="gray", ls="--", alpha=0.5)

    # Detect onsets
    onset_add = find_negative_onset(steps, grok_results["add"][:, 0])
    onset_mul = find_negative_onset(steps, grok_results["mul"][:, 0])
    onset_total = find_negative_onset(steps, grok_results["total"][:, 0])

    annotations = []
    if onset_add is not None:
        ax.axvline(onset_add, color="#1f77b4", ls="-.", alpha=0.7, lw=1.5)
        annotations.append(f"add onset: {onset_add}")
    if onset_mul is not None:
        ax.axvline(onset_mul, color="#d62728", ls="-.", alpha=0.7, lw=1.5)
        annotations.append(f"mul onset: {onset_mul}")
    if grok_data["grok_step_add"]:
        ax.axvline(grok_data["grok_step_add"], color="#1f77b4", ls=":", alpha=0.4)
        annotations.append(f"add groks: {grok_data['grok_step_add']}")
    if grok_data["grok_step_mul"]:
        ax.axvline(grok_data["grok_step_mul"], color="#d62728", ls=":", alpha=0.4)
        annotations.append(f"mul groks: {grok_data['grok_step_mul']}")

    # Annotation box
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
    fig.savefig(PLOT_DIR / f"figMT_H2_pertask_bottom_eig_s{seed}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved figMT_H2_pertask_bottom_eig_s{seed}.png")

    # ── Figure H3: Grok vs no-WD comparison ──────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for idx, (mode, title) in enumerate([("total", "Total"), ("add", "Add"), ("mul", "Mul")]):
        ax = axes[idx]

        # Grok
        steps_g = grok_results["steps"]
        ax.plot(steps_g, grok_results[mode][:, 0], color="#2ca02c", lw=2.5,
                label="wd=1.0 (grok)")

        # Control
        if control_results is not None:
            steps_c = control_results["steps"]
            ax.plot(steps_c, control_results[mode][:, 0], color="#d62728", lw=2,
                    ls="--", label="wd=0 (no grok)")

        ax.axhline(0, color="gray", ls="--", alpha=0.5)
        ax.set_xlabel("Training step")
        ax.set_ylabel("λ_min")
        ax.set_title(f"{title} Loss")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    fig.suptitle(f"Hessian Bottom Eigenvalue: Grokking vs No-WD Control (seed={seed})",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / f"figMT_H3_grok_vs_nowd_s{seed}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved figMT_H3_grok_vs_nowd_s{seed}.png")

    # ── Figure H4: Eigenvalue gap / ratio analysis ────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1: λ2/λ1 ratio (spectral gap) over training
    ax = axes[0]
    for mode, color, label in [("total", "#1a5276", "Total"),
                                ("add", "#1f77b4", "Add"),
                                ("mul", "#d62728", "Mul")]:
        steps = grok_results["steps"]
        eigs = grok_results[mode]
        # Ratio of 2nd to 1st eigenvalue (both could be negative)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.abs(eigs[:, 1]) / (np.abs(eigs[:, 0]) + 1e-12)
        ax.plot(steps, ratio, color=color, lw=2, label=label)

    ax.set_xlabel("Training step")
    ax.set_ylabel("|λ₂| / |λ₁|")
    ax.set_title("Spectral Gap: |λ₂/λ₁|")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 2)

    # Panel 2: λ_min(add) - λ_min(mul) difference
    ax = axes[1]
    steps = grok_results["steps"]
    diff = grok_results["add"][:, 0] - grok_results["mul"][:, 0]
    ax.plot(steps, diff, color="#8e44ad", lw=2)
    ax.axhline(0, color="gray", ls="--", alpha=0.5)
    if grok_data["grok_step_add"]:
        ax.axvline(grok_data["grok_step_add"], color="#1f77b4", ls=":", alpha=0.5)
    if grok_data["grok_step_mul"]:
        ax.axvline(grok_data["grok_step_mul"], color="#d62728", ls=":", alpha=0.5)
    ax.set_xlabel("Training step")
    ax.set_ylabel("λ_min(add) − λ_min(mul)")
    ax.set_title("Per-Task Eigenvalue Difference\n(positive = add more negative)")
    ax.grid(alpha=0.3)

    # Panel 3: Cumulative eigenvalue sum (sum of bottom-5)
    ax = axes[2]
    for mode, color, label in [("total", "#1a5276", "Total"),
                                ("add", "#1f77b4", "Add"),
                                ("mul", "#d62728", "Mul")]:
        steps = grok_results["steps"]
        eigs = grok_results[mode]
        cumsum = eigs.sum(axis=1)
        ax.plot(steps, cumsum, color=color, lw=2, label=label)

    if control_results is not None:
        for mode, color, ls in [("total", "gray", "--")]:
            steps_c = control_results["steps"]
            cumsum_c = control_results[mode].sum(axis=1)
            ax.plot(steps_c, cumsum_c, color=color, lw=1.5, ls=ls, label="control total")

    ax.axhline(0, color="gray", ls="--", alpha=0.3)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Sum of bottom-5 eigenvalues")
    ax.set_title("Cumulative Bottom Eigenvalue Sum")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.suptitle(f"Hessian Eigenvalue Ratio Analysis (seed={seed})", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / f"figMT_H4_ratio_analysis_s{seed}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved figMT_H4_ratio_analysis_s{seed}.png")

    # ── Figure H5: Onset detection summary ────────────────────────────────
    return {
        "onset_add": onset_add,
        "onset_mul": onset_mul,
        "onset_total": onset_total,
        "grok_step_add": grok_data["grok_step_add"],
        "grok_step_mul": grok_data["grok_step_mul"],
    }


def plot_onset_summary(all_onsets):
    """Cross-seed onset summary figure."""
    fig, ax = plt.subplots(figsize=(10, 6))

    seeds = sorted(all_onsets.keys())
    x = np.arange(len(seeds))
    width = 0.15

    for i, (key, color, label) in enumerate([
        ("onset_add", "#1f77b4", "λ_min(add) onset"),
        ("onset_mul", "#d62728", "λ_min(mul) onset"),
        ("grok_step_add", "#aec7e8", "add groks"),
        ("grok_step_mul", "#ff9896", "mul groks"),
    ]):
        vals = [all_onsets[s].get(key, 0) or 0 for s in seeds]
        ax.bar(x + i * width, vals, width, label=label, color=color, alpha=0.85)

    ax.set_xlabel("Seed")
    ax.set_ylabel("Training step")
    ax.set_title("Hessian Negative Eigenvalue Onset vs Grokking Step")
    ax.set_xticks(x + 1.5 * width)
    ax.set_xticklabels([f"s{s}" for s in seeds])
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(PLOT_DIR / "figMT_H5_onset_summary.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved figMT_H5_onset_summary.png")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    PLOT_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    seeds = [42, 137, 2024]
    all_hessian = {}
    all_onsets = {}

    for seed in seeds:
        grok_path = RESULTS_DIR / f"multitask_s{seed}.pt"
        if not grok_path.exists():
            print(f"[skip] {grok_path} not found")
            continue

        print(f"\n{'='*70}")
        print(f"  Hessian analysis: seed={seed}")
        print(f"{'='*70}")

        # Load grok run
        grok_data = torch.load(grok_path, map_location="cpu", weights_only=False)

        # Train / load control
        print(f"  Training no-WD control (seed={seed})...")
        control_data = train_control(seed=seed)

        # Hessian on grok run
        print(f"\n  [GROK] Hessian eigenvalues...")
        grok_hessian = analyze_hessian(grok_data, f"grok_s{seed}", seed,
                                        k=5, n_iter=25)

        # Hessian on control
        print(f"\n  [CONTROL] Hessian eigenvalues...")
        control_hessian = analyze_hessian(control_data, f"nowd_s{seed}", seed,
                                           k=5, n_iter=25)

        # Plot
        print(f"\n  Generating figures...")
        onsets = plot_hessian_figures(grok_hessian, control_hessian,
                                      grok_data, control_data, seed)

        all_hessian[seed] = {
            "grok": grok_hessian,
            "control": control_hessian,
        }
        all_onsets[seed] = onsets

    # Cross-seed summary
    if all_onsets:
        plot_onset_summary(all_onsets)

    # Save all results
    save_path = PLOT_DIR / "hessian_results.pt"
    torch.save({"hessian": all_hessian, "onsets": all_onsets}, save_path)
    print(f"\n  Saved all results to {save_path}")

    # Print summary
    print(f"\n{'='*70}")
    print("  HESSIAN ONSET SUMMARY")
    print(f"{'='*70}")
    print(f"  {'seed':>5s}  {'onset_add':>10s}  {'onset_mul':>10s}  {'grok_add':>10s}  "
          f"{'grok_mul':>10s}  {'onset_gap':>10s}  {'grok_gap':>10s}  {'first_neg':>10s}")
    for seed, o in sorted(all_onsets.items()):
        onset_gap = ""
        if o["onset_add"] is not None and o["onset_mul"] is not None:
            onset_gap = str(abs(o["onset_add"] - o["onset_mul"]))
        grok_gap = ""
        if o["grok_step_add"] is not None and o["grok_step_mul"] is not None:
            grok_gap = str(abs(o["grok_step_add"] - o["grok_step_mul"]))
        first_neg = "?"
        if o["onset_add"] is not None and o["onset_mul"] is not None:
            first_neg = "ADD" if o["onset_add"] < o["onset_mul"] else "MUL"
        elif o["onset_add"] is not None:
            first_neg = "ADD"
        elif o["onset_mul"] is not None:
            first_neg = "MUL"

        print(f"  {seed:5d}  {str(o['onset_add']):>10s}  {str(o['onset_mul']):>10s}  "
              f"{str(o['grok_step_add']):>10s}  {str(o['grok_step_mul']):>10s}  "
              f"{onset_gap:>10s}  {grok_gap:>10s}  {first_neg:>10s}")

    print("\nDone.")


if __name__ == "__main__":
    main()
