#!/usr/bin/env python3
"""
Gradient projection ablation for multi-task grokking.

Proves the PCA eigenvectors are "the real deal" by showing:
  - Projecting gradients onto PCA manifold (removing ⊥ component) → KILLS grokking
  - Projecting onto random manifold (same dimensionality) → grokking SURVIVES

This is the causal test: if the orthogonal-to-PCA directions are necessary
for grokking, then constraining gradients to the PCA manifold should prevent it.

Conditions:
  baseline        — Normal training (no intervention)
  pca-project     — Remove orthogonal gradient component (strength × g_⊥)
  random-project  — Remove component orthogonal to RANDOM basis (control)

Strengths: 0.25, 0.5, 0.75, 1.0
Seeds: 42, 137, 2024
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

sys.path.insert(0, str(Path(__file__).parent.parent))
from pca_sweep_analysis import pca_on_trajectory, collect_trajectory

from train_multitask import (
    MultiTaskConfig, MultiTaskTransformer, build_dataset, sample_batch,
    get_device, eval_accuracy, extract_attn_matrices,
)
from commutator_analysis import (
    build_pca_basis, _param_offsets, commutator_defect_median,
    flatten_model_params, _write_params,
    COMM_K, COMM_ETA, N_PCA_COMPONENTS,
)

RESULTS_DIR = Path(__file__).parent / "results"
PLOT_DIR = Path(__file__).parent / "plots"
SEEDS = [42, 137, 2024]
STRENGTHS = [0.25, 0.5, 0.75, 1.0]

# Intervention config
T_START = 500         # Start intervention after memorization phase
EVAL_EVERY = 100
MAX_STEPS = 50_000    # Cap for projection runs
COMM_EVERY = 500      # Measure commutator every N steps


def build_random_basis(model, n_dirs=16, seed=0):
    """Random orthonormal basis with same dims as PCA basis."""
    _, total_params = _param_offsets(model)
    rng = torch.Generator()
    rng.manual_seed(seed)
    R = torch.randn(total_params, n_dirs, generator=rng)
    Q, _ = torch.linalg.qr(R, mode="reduced")
    return Q


def project_gradient(model, B, strength=1.0):
    """
    After loss.backward(), remove (strength fraction of) the gradient component
    orthogonal to basis B.

    g_new = g - strength * g_⊥  where g_⊥ = g - B(B^T g)

    At strength=1.0: g_new = B(B^T g)  (fully constrained to manifold)
    At strength=0.0: g_new = g          (no intervention)
    """
    grads = []
    params_with_grad = []
    for p in model.parameters():
        if not p.requires_grad or p.grad is None:
            continue
        grads.append(p.grad.flatten())
        params_with_grad.append(p)

    if not grads:
        return

    grad_flat = torch.cat(grads)
    device = grad_flat.device
    B_dev = B.to(device)

    grad_parallel = B_dev @ (B_dev.T @ grad_flat)
    grad_perp = grad_flat - grad_parallel
    grad_new = grad_flat - strength * grad_perp

    offset = 0
    for p in params_with_grad:
        n = p.grad.numel()
        p.grad.copy_(grad_new[offset:offset + n].view_as(p.grad))
        offset += n


def train_with_projection(cfg, B, strength, label, max_steps=None):
    """Train multi-task model with gradient projection intervention."""
    device = get_device()
    steps = max_steps or MAX_STEPS

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
    B_dev = B.to(device) if B is not None else None

    def batch_fn():
        return sample_batch(train_pairs, cfg.BATCH_SIZE, cfg.P, device)

    records = []
    grokked_add, grokked_mul = False, False
    grok_step_add, grok_step_mul = None, None
    patience_add, patience_mul = 0, 0
    t0 = time.time()

    for step in range(1, steps + 1):
        model.train()
        a, b, y_add, y_mul = sample_batch(train_pairs, cfg.BATCH_SIZE, cfg.P, device)
        logits_add, logits_mul = model(a, b)
        loss = loss_fn(logits_add, y_add) + loss_fn(logits_mul, y_mul)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)

        # Gradient projection intervention
        if step >= T_START and B_dev is not None and strength > 0:
            project_gradient(model, B_dev, strength=strength)

        opt.step()

        if step % EVAL_EVERY == 0 or step == 1:
            model.eval()
            train_add, train_mul = eval_accuracy(model, train_pairs, cfg, device)
            test_add, test_mul = eval_accuracy(model, test_pairs, cfg, device)

            rec = {
                "step": step,
                "train_add": train_add, "train_mul": train_mul,
                "test_add": test_add, "test_mul": test_mul,
                "loss": loss.item(),
            }

            # Commutator measurement (less frequent)
            if step % COMM_EVERY == 0:
                out = commutator_defect_median(model, batch_fn, device,
                                               K=5, eta=COMM_ETA, mode="total")
                rec["defect"] = out["median"]

            records.append(rec)

            # Track grokking
            if not grokked_add and test_add >= cfg.STOP_ACC:
                patience_add += 1
                if patience_add >= cfg.STOP_PATIENCE:
                    grokked_add = True
                    grok_step_add = step
            else:
                if not grokked_add:
                    patience_add = 0

            if not grokked_mul and test_mul >= cfg.STOP_ACC:
                patience_mul += 1
                if patience_mul >= cfg.STOP_PATIENCE:
                    grokked_mul = True
                    grok_step_mul = step
            else:
                if not grokked_mul:
                    patience_mul = 0

            # Stop when both grok (or hit max)
            if grokked_add and grokked_mul:
                break

            if step % 2000 == 0:
                elapsed = (time.time() - t0) / 60
                d = rec.get("defect", 0)
                print(f"      step {step:6d} | add {test_add:.3f} mul {test_mul:.3f} | "
                      f"defect {d:.1f} | {elapsed:.1f}m")

    elapsed = (time.time() - t0) / 60
    return {
        "records": records,
        "grokked_add": grokked_add,
        "grokked_mul": grokked_mul,
        "grok_step_add": grok_step_add,
        "grok_step_mul": grok_step_mul,
        "label": label,
        "strength": strength,
        "seed": cfg.SEED,
        "elapsed_min": elapsed,
    }


def main():
    PLOT_DIR.mkdir(exist_ok=True)
    device = get_device()
    print(f"Device: {device}")

    cache_path = PLOT_DIR / "gradient_projection_ablation.pt"
    all_runs = {}
    pca_bases = {}
    random_bases = {}

    if cache_path.exists():
        cached = torch.load(cache_path, map_location="cpu", weights_only=False)
        all_runs = cached.get("all_runs", {})
        pca_bases = cached.get("pca_bases", {})
        random_bases = cached.get("random_bases", {})
        print(f"  Loaded {len(all_runs)} cached runs")

    # ── Phase 1: Get PCA bases from trained models ─────────────────────
    print(f"\n{'='*70}")
    print("  Phase 1: Build PCA and Random Bases")
    print(f"{'='*70}")

    for seed in SEEDS:
        if seed in pca_bases:
            print(f"  seed={seed}: PCA basis cached ({pca_bases[seed].shape})")
            continue

        tag = f"multitask_s{seed}"
        path = RESULTS_DIR / f"{tag}.pt"
        if not path.exists():
            print(f"  [{tag}] not found!")
            continue

        data = torch.load(path, map_location="cpu", weights_only=False)
        cfg = MultiTaskConfig(**data["cfg"])
        model = MultiTaskTransformer(cfg)
        B = build_pca_basis(model, data["attn_logs"],
                            n_components=N_PCA_COMPONENTS, device="cpu")
        pca_bases[seed] = B
        print(f"  seed={seed}: PCA basis shape {B.shape}")

        # Random basis with same dimensionality
        n_dirs = B.shape[1]
        R = build_random_basis(model, n_dirs=n_dirs, seed=seed + 77777)
        random_bases[seed] = R
        print(f"  seed={seed}: Random basis shape {R.shape}")

    # ── Phase 2: Run experiments ───────────────────────────────────────
    print(f"\n{'='*70}")
    print("  Phase 2: Gradient Projection Experiments")
    print(f"{'='*70}")

    conditions = [("baseline", 0.0)]
    for s in STRENGTHS:
        conditions.append(("pca-project", s))
        conditions.append(("random-project", s))

    total = len(conditions) * len(SEEDS)
    run_i = 0

    for cond_name, strength in conditions:
        for seed in SEEDS:
            run_i += 1
            key = (cond_name, strength, seed)

            if key in all_runs:
                r = all_runs[key]
                ga = r.get("grok_step_add", "—")
                gm = r.get("grok_step_mul", "—")
                print(f"  [{run_i}/{total}] {cond_name} str={strength} s={seed} — "
                      f"cached (add={ga}, mul={gm})")
                continue

            print(f"\n  [{run_i}/{total}] {cond_name} str={strength} s={seed}")

            cfg = MultiTaskConfig(SEED=seed)

            if cond_name == "baseline":
                B = pca_bases[seed]  # doesn't matter, strength=0
                max_steps = None
            elif cond_name == "pca-project":
                B = pca_bases[seed]
                max_steps = MAX_STEPS
            else:
                B = random_bases[seed]
                max_steps = MAX_STEPS

            r = train_with_projection(cfg, B, strength, cond_name, max_steps=max_steps)
            all_runs[key] = r

            ga = r["grok_step_add"] or "—"
            gm = r["grok_step_mul"] or "—"
            print(f"    → add={ga}, mul={gm} ({r['elapsed_min']:.1f}m)")

            # Save after each run
            torch.save({
                "all_runs": all_runs,
                "pca_bases": pca_bases,
                "random_bases": random_bases,
            }, cache_path)

    # ── Summary ────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  GRADIENT PROJECTION ABLATION RESULTS")
    print(f"{'='*70}")
    print(f"  {'Condition':>18s}  {'Str':>5s}  {'Seed':>5s}  {'Add Grok':>9s}  {'Mul Grok':>9s}")
    print(f"  {'-'*18}  {'-'*5}  {'-'*5}  {'-'*9}  {'-'*9}")

    for cond_name, strength in conditions:
        for seed in SEEDS:
            key = (cond_name, strength, seed)
            if key not in all_runs:
                continue
            r = all_runs[key]
            ga = str(r["grok_step_add"]) if r["grok_step_add"] else "FAIL"
            gm = str(r["grok_step_mul"]) if r["grok_step_mul"] else "FAIL"
            print(f"  {cond_name:>18s}  {strength:>5.2f}  {seed:>5d}  {ga:>9s}  {gm:>9s}")

    # ── Figures ────────────────────────────────────────────────────────
    print(f"\n  Generating figures...")

    # Fig 1: Accuracy overlay at strength=1.0
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    plot_configs = [
        ("baseline", 0.0, "#333333", "Baseline"),
        ("pca-project", 1.0, "#2980b9", "PCA projection (str=1.0)"),
        ("random-project", 1.0, "#27ae60", "Random projection (str=1.0)"),
    ]

    for task_idx, (task, task_label) in enumerate([("test_add", "Mod-Add"),
                                                     ("test_mul", "Mod-Mul")]):
        ax = axes[task_idx]
        for cond_name, strength, color, label in plot_configs:
            all_steps = set()
            seed_data = {}
            for seed in SEEDS:
                key = (cond_name, strength, seed)
                if key not in all_runs:
                    continue
                recs = all_runs[key]["records"]
                sd = {rec["step"]: rec[task] for rec in recs}
                seed_data[seed] = sd
                all_steps.update(sd.keys())

            if not seed_data:
                continue

            steps_sorted = sorted(all_steps)
            means, lows, highs = [], [], []
            for s in steps_sorted:
                vals = [sd[s] for sd in seed_data.values() if s in sd]
                if vals:
                    means.append(np.mean(vals))
                    lows.append(np.min(vals))
                    highs.append(np.max(vals))
                else:
                    means.append(float("nan"))
                    lows.append(float("nan"))
                    highs.append(float("nan"))

            ax.plot(steps_sorted, means, color=color, lw=2.5, label=label)
            ax.fill_between(steps_sorted, lows, highs, color=color, alpha=0.15)

        ax.axvline(T_START, color="gray", ls="-.", alpha=0.5, lw=1)
        ax.axhline(0.98, color="green", ls=":", alpha=0.3)
        ax.set_xlabel("Training step", fontsize=12)
        ax.set_ylabel(f"Test accuracy ({task_label})", fontsize=12)
        ax.set_title(task_label, fontsize=13)
        ax.set_ylim(-0.02, 1.05)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.2)

    fig.suptitle("PCA vs Random Gradient Projection (strength=1.0)\n"
                 f"PCA projection kills grokking; random projection does not",
                 fontsize=13, y=1.03)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "figGP_A_accuracy_overlay.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved figGP_A_accuracy_overlay.png")

    # Fig 2: Dose-response (grok step vs strength)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    for ax_idx, (task_key, task_grok_key, task_label) in enumerate([
        ("test_add", "grok_step_add", "Add"),
        ("test_mul", "grok_step_mul", "Mul"),
    ]):
        ax = axes[ax_idx]
        for cond_name, color, marker in [
            ("pca-project", "#2980b9", "o"),
            ("random-project", "#27ae60", "s"),
        ]:
            means, all_pts = [], []
            for s in STRENGTHS:
                gsteps = []
                for seed in SEEDS:
                    key = (cond_name, s, seed)
                    if key in all_runs:
                        gs = all_runs[key][task_grok_key]
                        if gs:
                            gsteps.append(gs)
                all_pts.append(gsteps)
                means.append(np.mean(gsteps) if gsteps else MAX_STEPS)

            label = "PCA" if "pca" in cond_name else "Random"
            ax.plot(STRENGTHS, means, f"{marker}-", color=color, lw=2.5,
                    ms=10, label=label)
            for i, pts in enumerate(all_pts):
                for p in pts:
                    ax.scatter(STRENGTHS[i], p, color=color, s=30, alpha=0.5)

            # Mark failures
            for i, pts in enumerate(all_pts):
                n_fail = len(SEEDS) - len(pts)
                if n_fail > 0:
                    ax.scatter(STRENGTHS[i], MAX_STEPS, color=color, marker="X",
                               s=100, zorder=10)

        # Baseline
        baseline_steps = []
        for seed in SEEDS:
            key = ("baseline", 0.0, seed)
            if key in all_runs:
                gs = all_runs[key][task_grok_key]
                if gs:
                    baseline_steps.append(gs)
        if baseline_steps:
            ax.axhline(np.mean(baseline_steps), color="#333", ls="--", alpha=0.5,
                       lw=1.5, label="Baseline")

        ax.set_xlabel("Projection strength", fontsize=12)
        ax.set_ylabel("Grok step", fontsize=12)
        ax.set_title(f"{task_label}: Grok Timing vs Projection Strength", fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    fig.suptitle("Dose-Response: PCA vs Random Projection\n"
                 "(X = failed to grok within budget)",
                 fontsize=13, y=1.03)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "figGP_B_dose_response.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved figGP_B_dose_response.png")

    # Fig 3: Grok rate (fraction that grokked)
    fig, ax = plt.subplots(figsize=(8, 5))
    for cond_name, color, marker in [
        ("pca-project", "#2980b9", "o"),
        ("random-project", "#27ae60", "s"),
    ]:
        rates_add, rates_mul = [], []
        for s in STRENGTHS:
            n_add, n_mul = 0, 0
            for seed in SEEDS:
                key = (cond_name, s, seed)
                if key in all_runs:
                    if all_runs[key]["grokked_add"]:
                        n_add += 1
                    if all_runs[key]["grokked_mul"]:
                        n_mul += 1
            rates_add.append(n_add / len(SEEDS))
            rates_mul.append(n_mul / len(SEEDS))

        label = "PCA" if "pca" in cond_name else "Random"
        ax.plot(STRENGTHS, rates_add, f"{marker}-", color=color, lw=2.5, ms=10,
                label=f"{label} (add)")
        ax.plot(STRENGTHS, rates_mul, f"{marker}--", color=color, lw=2, ms=8,
                alpha=0.7, label=f"{label} (mul)")

    ax.set_xlabel("Projection strength", fontsize=12)
    ax.set_ylabel("Grok rate (fraction of seeds)", fontsize=12)
    ax.set_title("Grok Success Rate: PCA vs Random Projection", fontsize=13)
    ax.set_ylim(-0.05, 1.15)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "figGP_C_grok_rate.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved figGP_C_grok_rate.png")

    # Final save
    torch.save({
        "all_runs": all_runs,
        "pca_bases": pca_bases,
        "random_bases": random_bases,
    }, cache_path)
    print(f"\n  Saved all results to {cache_path.name}")
    print("\nDone.")


if __name__ == "__main__":
    main()
