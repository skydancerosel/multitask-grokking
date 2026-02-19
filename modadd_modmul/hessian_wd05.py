#!/usr/bin/env python3
"""
WD=0.5 intermediate run + 3-way Hessian comparison.

1. Train multi-task model with wd=0.5 (intermediate regime)
2. Run Hessian bottom-eig analysis on checkpoints
3. Load existing wd=1.0 and wd=0.0 Hessian results
4. Produce 3-way comparison: scaling of λ_min with weight decay

Figures:
  figMT_H6 — 3-way Hessian comparison (wd=1.0 / 0.5 / 0.0)
  figMT_H7 — Per-task λ_min for wd=0.5 run
  figMT_H8 — Scaling law: λ_min vs WD at matched training fractions
  figMT_H9 — Accuracy curves for wd=0.5 (does it grok?)
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
from hessian_analysis import (
    analyze_hessian, find_negative_onset,
)

RESULTS_DIR = Path(__file__).parent / "results"
PLOT_DIR = Path(__file__).parent / "plots"


# ═══════════════════════════════════════════════════════════════════════════
# Train wd=0.5
# ═══════════════════════════════════════════════════════════════════════════

def train_wd05(seed=42):
    tag = f"multitask_wd05_s{seed}"
    out_path = RESULTS_DIR / f"{tag}.pt"
    if out_path.exists():
        print(f"  [{tag}] already exists, loading...")
        return torch.load(out_path, map_location="cpu", weights_only=False)

    device = get_device()
    cfg = MultiTaskConfig(
        SEED=seed,
        WEIGHT_DECAY=0.5,
        STEPS=200_000,          # generous budget
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
            if step % 2000 == 0:
                elapsed = (time.time() - t0) / 60
                print(f"    [wd=0.5] step {step:6d} | add: {train_add:.3f}/{test_add:.3f} | "
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

def fig_h9_accuracy_wd05(data, seed):
    """Accuracy curves for wd=0.5 run."""
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
    ax1.set_xlabel("Training step")
    ax1.set_ylabel("Accuracy")
    ax1.set_title(f"Multi-Task wd=0.5 (seed={seed})")
    ax1.legend(fontsize=8, ncol=2)
    ax1.grid(alpha=0.3)

    ax2.plot(steps, [m["loss"] for m in metrics], color="black", lw=2)
    ax2.set_xlabel("Training step")
    ax2.set_ylabel("Total loss")
    ax2.set_title(f"Loss (wd=0.5, seed={seed})")
    ax2.set_yscale("log")
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(PLOT_DIR / f"figMT_H9_accuracy_wd05_s{seed}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved figMT_H9_accuracy_wd05_s{seed}.png")


def fig_h7_pertask_wd05(hess_results, data, seed):
    """Per-task bottom eig for wd=0.5."""
    fig, ax = plt.subplots(figsize=(12, 6))
    steps = hess_results["steps"]
    ax.plot(steps, hess_results["add"][:, 0], color="#1f77b4", lw=2.5, label="λ_min (add loss)")
    ax.plot(steps, hess_results["mul"][:, 0], color="#d62728", lw=2.5, label="λ_min (mul loss)")
    ax.plot(steps, hess_results["total"][:, 0], color="#2ecc71", lw=1.5, ls="--", label="λ_min (total)")
    ax.axhline(0, color="gray", ls="--", alpha=0.5)
    if data["grok_step_add"]:
        ax.axvline(data["grok_step_add"], color="#1f77b4", ls=":", alpha=0.5)
    if data["grok_step_mul"]:
        ax.axvline(data["grok_step_mul"], color="#d62728", ls=":", alpha=0.5)
    ax.set_xlabel("Training step", fontsize=12)
    ax.set_ylabel("Bottom Hessian eigenvalue", fontsize=12)
    ax.set_title(f"Per-Task Hessian Bottom Eigenvalues — wd=0.5 (seed={seed})")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / f"figMT_H7_pertask_wd05_s{seed}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved figMT_H7_pertask_wd05_s{seed}.png")


def fig_h6_three_way(hess_10, hess_05, hess_00, data_10, data_05, seed):
    """3-way comparison: wd=1.0 vs 0.5 vs 0.0 bottom eigenvalues."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    configs = [
        ("total", "Total Loss"),
        ("add", "Add Loss"),
        ("mul", "Mul Loss"),
    ]
    for idx, (mode, title) in enumerate(configs):
        ax = axes[idx]
        for hess, wd_label, color, ls, lw in [
            (hess_10, "wd=1.0", "#2ca02c", "-", 2.5),
            (hess_05, "wd=0.5", "#ff7f0e", "-", 2.5),
            (hess_00, "wd=0.0", "#d62728", "--", 1.5),
        ]:
            if hess is not None:
                ax.plot(hess["steps"], hess[mode][:, 0], color=color, lw=lw,
                        ls=ls, label=wd_label)

        ax.axhline(0, color="gray", ls="--", alpha=0.5)
        # Mark grok steps for wd=1.0
        if data_10["grok_step_add"]:
            ax.axvline(data_10["grok_step_add"], color="#2ca02c", ls=":", alpha=0.3)
        if data_10["grok_step_mul"]:
            ax.axvline(data_10["grok_step_mul"], color="#2ca02c", ls=":", alpha=0.3)
        # Mark grok steps for wd=0.5
        if data_05["grok_step_add"]:
            ax.axvline(data_05["grok_step_add"], color="#ff7f0e", ls=":", alpha=0.3)
        if data_05["grok_step_mul"]:
            ax.axvline(data_05["grok_step_mul"], color="#ff7f0e", ls=":", alpha=0.3)

        ax.set_xlabel("Training step")
        ax.set_ylabel("λ_min")
        ax.set_title(title)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    fig.suptitle(f"3-Way Hessian Comparison: wd=1.0 vs 0.5 vs 0.0 (seed={seed})",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / f"figMT_H6_three_way_s{seed}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved figMT_H6_three_way_s{seed}.png")


def fig_h8_scaling(hess_dict, data_dict, seed):
    """
    Scaling law: how does λ_min scale with WD?
    Sample at matched training fraction (25%, 50%, 75% of each run).
    """
    wds = sorted(hess_dict.keys())

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for idx, (mode, title) in enumerate([("total", "Total"), ("add", "Add"), ("mul", "Mul")]):
        ax = axes[idx]

        # Collect λ_min at 25%, 50%, 75% of training for each WD
        fracs = [0.25, 0.50, 0.75]
        frac_colors = ["#3498db", "#e74c3c", "#2ecc71"]

        for fi, (frac, fc) in enumerate(zip(fracs, frac_colors)):
            wd_vals = []
            lam_vals = []
            for wd in wds:
                hess = hess_dict[wd]
                steps = hess["steps"]
                eigs = hess[mode][:, 0]
                # Find checkpoint closest to frac of total steps
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

    fig.suptitle(f"Scaling Law: λ_min vs Weight Decay (seed={seed})",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / f"figMT_H8_scaling_law_s{seed}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved figMT_H8_scaling_law_s{seed}.png")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    PLOT_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    seed = 42  # single seed for now

    # ── 1. Train wd=0.5 ──────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  Training wd=0.5 (seed={seed})")
    print(f"{'='*70}")
    data_05 = train_wd05(seed=seed)
    fig_h9_accuracy_wd05(data_05, seed)

    # ── 2. Hessian on wd=0.5 ─────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  Hessian analysis: wd=0.5 (seed={seed})")
    print(f"{'='*70}")
    hess_05 = analyze_hessian(data_05, f"wd05_s{seed}", seed, k=5, n_iter=25)
    fig_h7_pertask_wd05(hess_05, data_05, seed)

    # ── 3. Load existing results ──────────────────────────────────────────
    print(f"\n  Loading existing Hessian results...")
    existing = torch.load(PLOT_DIR / "hessian_results.pt", map_location="cpu",
                          weights_only=False)
    hess_10 = existing["hessian"].get(seed, {}).get("grok")
    hess_00 = existing["hessian"].get(seed, {}).get("control")

    data_10 = torch.load(RESULTS_DIR / f"multitask_s{seed}.pt",
                          map_location="cpu", weights_only=False)

    # ── 4. 3-way comparison ──────────────────────────────────────────────
    print(f"\n  Generating comparison figures...")
    fig_h6_three_way(hess_10, hess_05, hess_00, data_10, data_05, seed)

    # ── 5. Scaling law ───────────────────────────────────────────────────
    hess_dict = {}
    data_dict = {}
    if hess_10 is not None:
        hess_dict[1.0] = hess_10
        data_dict[1.0] = data_10
    hess_dict[0.5] = hess_05
    data_dict[0.5] = data_05
    if hess_00 is not None:
        hess_dict[0.0] = hess_00

    fig_h8_scaling(hess_dict, data_dict, seed)

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  WD=0.5 SUMMARY")
    print(f"{'='*70}")
    print(f"  Grokked add: {data_05['grokked_add']} at step {data_05['grok_step_add']}")
    print(f"  Grokked mul: {data_05['grokked_mul']} at step {data_05['grok_step_mul']}")
    print(f"  Final λ_min(total) = {hess_05['total'][-1, 0]:.4f}")
    print(f"  Final λ_min(add)   = {hess_05['add'][-1, 0]:.4f}")
    print(f"  Final λ_min(mul)   = {hess_05['mul'][-1, 0]:.4f}")

    if hess_10 is not None:
        print(f"\n  Comparison at final checkpoint:")
        print(f"  {'':>12s}  {'wd=1.0':>10s}  {'wd=0.5':>10s}  {'wd=0.0':>10s}")
        for mode in ["total", "add", "mul"]:
            v10 = hess_10[mode][-1, 0] if hess_10 is not None else float("nan")
            v05 = hess_05[mode][-1, 0]
            v00 = hess_00[mode][-1, 0] if hess_00 is not None else float("nan")
            print(f"  λ_min({mode:>5s})  {v10:10.4f}  {v05:10.4f}  {v00:10.4f}")

    # Save
    save_path = PLOT_DIR / "hessian_wd05_results.pt"
    torch.save({
        "hess_05": hess_05,
        "data_05_meta": {
            "grokked_add": data_05["grokked_add"],
            "grokked_mul": data_05["grokked_mul"],
            "grok_step_add": data_05["grok_step_add"],
            "grok_step_mul": data_05["grok_step_mul"],
        },
    }, save_path)
    print(f"\n  Saved → {save_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
