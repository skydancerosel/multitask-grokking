#!/usr/bin/env python3
"""
WD=0.1 run + 4-way Hessian comparison (wd=1.0 / 0.5 / 0.1 / 0.0).

Produces:
  figMT_H10 — Accuracy curves for wd=0.1
  figMT_H11 — Per-task bottom eig for wd=0.1
  figMT_H12 — 4-way Hessian comparison
  figMT_H13 — Scaling law: λ_min vs WD (4 points)
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
from hessian_analysis import analyze_hessian

RESULTS_DIR = Path(__file__).parent / "results"
PLOT_DIR = Path(__file__).parent / "plots"


# ═══════════════════════════════════════════════════════════════════════════
# Train wd=0.1
# ═══════════════════════════════════════════════════════════════════════════

def train_wd01(seed=42):
    tag = f"multitask_wd01_s{seed}"
    out_path = RESULTS_DIR / f"{tag}.pt"
    if out_path.exists():
        print(f"  [{tag}] already exists, loading...")
        return torch.load(out_path, map_location="cpu", weights_only=False)

    device = get_device()
    cfg = MultiTaskConfig(
        SEED=seed,
        WEIGHT_DECAY=0.1,
        STEPS=200_000,
        CHECKPOINT_EVERY=200,
        MODEL_LOG_EVERY=200,
    )

    torch.manual_seed(cfg.SEED)
    np.random.seed(cfg.SEED)
    random.seed(cfg.SEED)

    train_pairs, test_pairs = build_dataset(cfg.P, cfg.TRAIN_FRACTION, cfg.SEED)
    model = MultiTaskTransformer(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model: {n_params:,} params, wd={cfg.WEIGHT_DECAY}")

    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY,
        betas=(cfg.ADAM_BETA1, cfg.ADAM_BETA2)
    )
    loss_fn = nn.CrossEntropyLoss()

    init_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    attn_logs = [{"step": 0, "layers": extract_attn_matrices(model)}]
    checkpoints = [(0, {k: v.cpu().clone() for k, v in model.state_dict().items()})]
    metrics = []
    patience_add, patience_mul = 0, 0
    grokked_add, grokked_mul = False, False
    grok_step_add, grok_step_mul = None, None
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
                print(f"    [wd=0.1] step {step:6d} | add: {train_add:.3f}/{test_add:.3f} | "
                      f"mul: {train_mul:.3f}/{test_mul:.3f} | {elapsed:.1f}m")

            if not grokked_add:
                if test_add >= cfg.STOP_ACC:
                    patience_add += 1
                    if patience_add >= cfg.STOP_PATIENCE:
                        grokked_add = True
                        grok_step_add = step
                        print(f"    >>> ADD GROKKED at step {step}")
                else:
                    patience_add = 0
            if not grokked_mul:
                if test_mul >= cfg.STOP_ACC:
                    patience_mul += 1
                    if patience_mul >= cfg.STOP_PATIENCE:
                        grokked_mul = True
                        grok_step_mul = step
                        print(f"    >>> MUL GROKKED at step {step}")
                else:
                    patience_mul = 0
            if grokked_add and grokked_mul:
                print(f"    >>> BOTH GROKKED — stopping at step {step}")
                break

    elapsed = (time.time() - t0) / 60
    print(f"  Training done in {elapsed:.1f}m — "
          f"add grokked={grokked_add}@{grok_step_add}, "
          f"mul grokked={grokked_mul}@{grok_step_mul}")

    result = {
        "cfg": asdict(cfg),
        "attn_logs": attn_logs,
        "checkpoints": checkpoints,
        "metrics": metrics,
        "grokked_add": grokked_add,
        "grokked_mul": grokked_mul,
        "grok_step_add": grok_step_add,
        "grok_step_mul": grok_step_mul,
        "final_step": step,
        "init_state": init_state,
    }
    torch.save(result, out_path)
    print(f"  saved → {out_path.name} ({len(checkpoints)} ckpts)")
    return result


# ═══════════════════════════════════════════════════════════════════════════
# Figures
# ═══════════════════════════════════════════════════════════════════════════

def fig_h10_accuracy(data, seed):
    metrics = data["metrics"]
    steps = [m["step"] for m in metrics]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(steps, [m["train_add"] for m in metrics], label="train add", color="#1f77b4", lw=2)
    ax1.plot(steps, [m["test_add"] for m in metrics], label="test add", color="#1f77b4", ls="--", lw=2)
    ax1.plot(steps, [m["train_mul"] for m in metrics], label="train mul", color="#d62728", lw=2)
    ax1.plot(steps, [m["test_mul"] for m in metrics], label="test mul", color="#d62728", ls="--", lw=2)
    ax1.axhline(0.98, color="gray", ls=":", alpha=0.5)
    if data["grok_step_add"]:
        ax1.axvline(data["grok_step_add"], color="#1f77b4", ls=":", alpha=0.7,
                     label=f"add groks @{data['grok_step_add']}")
    if data["grok_step_mul"]:
        ax1.axvline(data["grok_step_mul"], color="#d62728", ls=":", alpha=0.7,
                     label=f"mul groks @{data['grok_step_mul']}")
    ax1.set_xlabel("Training step"); ax1.set_ylabel("Accuracy")
    ax1.set_title(f"Multi-Task wd=0.1 (seed={seed})")
    ax1.legend(fontsize=8, ncol=2); ax1.grid(alpha=0.3)

    ax2.plot(steps, [m["loss"] for m in metrics], color="black", lw=2)
    ax2.set_xlabel("Training step"); ax2.set_ylabel("Total loss")
    ax2.set_title(f"Loss (wd=0.1, seed={seed})"); ax2.set_yscale("log"); ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(PLOT_DIR / f"figMT_H10_accuracy_wd01_s{seed}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved figMT_H10_accuracy_wd01_s{seed}.png")


def fig_h11_pertask(hess, data, seed):
    fig, ax = plt.subplots(figsize=(12, 6))
    steps = hess["steps"]
    ax.plot(steps, hess["add"][:, 0], color="#1f77b4", lw=2.5, label="λ_min (add)")
    ax.plot(steps, hess["mul"][:, 0], color="#d62728", lw=2.5, label="λ_min (mul)")
    ax.plot(steps, hess["total"][:, 0], color="#2ecc71", lw=1.5, ls="--", label="λ_min (total)")
    ax.axhline(0, color="gray", ls="--", alpha=0.5)
    if data["grok_step_add"]:
        ax.axvline(data["grok_step_add"], color="#1f77b4", ls=":", alpha=0.5)
    if data["grok_step_mul"]:
        ax.axvline(data["grok_step_mul"], color="#d62728", ls=":", alpha=0.5)
    ax.set_xlabel("Training step", fontsize=12)
    ax.set_ylabel("Bottom Hessian eigenvalue", fontsize=12)
    ax.set_title(f"Per-Task Hessian Bottom Eigenvalues — wd=0.1 (seed={seed})")
    ax.legend(fontsize=10); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / f"figMT_H11_pertask_wd01_s{seed}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved figMT_H11_pertask_wd01_s{seed}.png")


def fig_h12_four_way(hess_all, data_all, seed):
    """4-way comparison: wd=1.0 / 0.5 / 0.1 / 0.0."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    wd_styles = [
        (1.0, "wd=1.0", "#2ca02c", "-", 2.5),
        (0.5, "wd=0.5", "#ff7f0e", "-", 2.0),
        (0.1, "wd=0.1", "#9467bd", "-", 2.0),
        (0.0, "wd=0.0", "#d62728", "--", 1.5),
    ]

    for idx, (mode, title) in enumerate([("total", "Total Loss"),
                                          ("add", "Add Loss"),
                                          ("mul", "Mul Loss")]):
        ax = axes[idx]
        for wd, label, color, ls, lw in wd_styles:
            if wd not in hess_all:
                continue
            hess = hess_all[wd]
            ax.plot(hess["steps"], hess[mode][:, 0], color=color, lw=lw,
                    ls=ls, label=label)

        ax.axhline(0, color="gray", ls="--", alpha=0.5)
        # Mark grok steps
        for wd, _, color, _, _ in wd_styles:
            if wd in data_all and data_all[wd].get("grok_step_mul"):
                ax.axvline(data_all[wd]["grok_step_mul"], color=color, ls=":", alpha=0.25)

        ax.set_xlabel("Training step")
        ax.set_ylabel("λ_min")
        ax.set_title(title)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    fig.suptitle(f"4-Way Hessian Comparison (seed={seed})", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / f"figMT_H12_four_way_s{seed}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved figMT_H12_four_way_s{seed}.png")


def fig_h13_scaling(hess_all, data_all, seed):
    """Scaling law with 4 WD values."""
    wds = sorted(hess_all.keys())
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for idx, (mode, title) in enumerate([("total", "Total"), ("add", "Add"), ("mul", "Mul")]):
        ax = axes[idx]
        fracs = [0.25, 0.50, 0.75]
        frac_colors = ["#3498db", "#e74c3c", "#2ecc71"]

        for fi, (frac, fc) in enumerate(zip(fracs, frac_colors)):
            wd_vals, lam_vals = [], []
            for wd in wds:
                hess = hess_all[wd]
                steps = hess["steps"]
                eigs = hess[mode][:, 0]
                target_step = int(frac * steps[-1])
                ci = np.argmin(np.abs(steps - target_step))
                wd_vals.append(wd)
                lam_vals.append(eigs[ci])
            ax.plot(wd_vals, lam_vals, "o-", color=fc, lw=2, ms=8,
                    label=f"{int(frac*100)}% of training")

        ax.axhline(0, color="gray", ls="--", alpha=0.5)
        ax.set_xlabel("Weight Decay")
        ax.set_ylabel("λ_min")
        ax.set_title(f"{title} Loss")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    fig.suptitle(f"Scaling Law: λ_min vs Weight Decay — 4 Points (seed={seed})",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / f"figMT_H13_scaling_4pt_s{seed}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved figMT_H13_scaling_4pt_s{seed}.png")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    PLOT_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    seed = 42

    # ── 1. Train wd=0.1 ──────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  Training wd=0.1 (seed={seed})")
    print(f"{'='*70}")
    data_01 = train_wd01(seed=seed)
    fig_h10_accuracy(data_01, seed)

    # ── 2. Hessian on wd=0.1 ─────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  Hessian analysis: wd=0.1 (seed={seed})")
    print(f"{'='*70}")
    hess_01 = analyze_hessian(data_01, f"wd01_s{seed}", seed, k=5, n_iter=25)
    fig_h11_pertask(hess_01, data_01, seed)

    # ── 3. Load all existing results ─────────────────────────────────────
    print(f"\n  Loading existing results...")

    existing_hess = torch.load(PLOT_DIR / "hessian_results.pt",
                                map_location="cpu", weights_only=False)
    existing_wd05 = torch.load(PLOT_DIR / "hessian_wd05_results.pt",
                                map_location="cpu", weights_only=False)

    hess_all = {}
    data_all = {}

    # wd=1.0
    hess_10 = existing_hess["hessian"].get(seed, {}).get("grok")
    if hess_10:
        hess_all[1.0] = hess_10
        d10 = torch.load(RESULTS_DIR / f"multitask_s{seed}.pt",
                          map_location="cpu", weights_only=False)
        data_all[1.0] = {"grok_step_add": d10["grok_step_add"],
                          "grok_step_mul": d10["grok_step_mul"]}

    # wd=0.5
    hess_05 = existing_wd05.get("hess_05")
    if hess_05:
        hess_all[0.5] = hess_05
        data_all[0.5] = existing_wd05.get("data_05_meta", {})

    # wd=0.1
    hess_all[0.1] = hess_01
    data_all[0.1] = {"grok_step_add": data_01["grok_step_add"],
                      "grok_step_mul": data_01["grok_step_mul"]}

    # wd=0.0
    hess_00 = existing_hess["hessian"].get(seed, {}).get("control")
    if hess_00:
        hess_all[0.0] = hess_00
        data_all[0.0] = {"grok_step_add": None, "grok_step_mul": None}

    # ── 4. Comparison figures ────────────────────────────────────────────
    print(f"\n  Generating comparison figures...")
    fig_h12_four_way(hess_all, data_all, seed)
    fig_h13_scaling(hess_all, data_all, seed)

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  4-WAY HESSIAN SUMMARY (seed=42)")
    print(f"{'='*70}")
    print(f"  {'WD':>6s}  {'grok_add':>10s}  {'grok_mul':>10s}  "
          f"{'λ_min(tot)':>12s}  {'λ_min(add)':>12s}  {'λ_min(mul)':>12s}")
    for wd in sorted(hess_all.keys(), reverse=True):
        hess = hess_all[wd]
        d = data_all.get(wd, {})
        ga = str(d.get("grok_step_add", "—"))
        gm = str(d.get("grok_step_mul", "—"))
        lt = hess["total"][-1, 0]
        la = hess["add"][-1, 0]
        lm = hess["mul"][-1, 0]
        print(f"  {wd:6.1f}  {ga:>10s}  {gm:>10s}  {lt:12.4f}  {la:12.4f}  {lm:12.4f}")

    # Save
    save_path = PLOT_DIR / "hessian_wd01_results.pt"
    torch.save({"hess_01": hess_01, "data_01_meta": {
        "grokked_add": data_01["grokked_add"], "grokked_mul": data_01["grokked_mul"],
        "grok_step_add": data_01["grok_step_add"], "grok_step_mul": data_01["grok_step_mul"],
    }}, save_path)
    print(f"\n  Saved → {save_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
