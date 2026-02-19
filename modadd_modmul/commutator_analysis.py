#!/usr/bin/env python3
"""
Commutator / curvature analysis for multi-task grokking.

Loads checkpoints from train_multitask.py and computes:
  1. Commutator defect over training (both tasks use shared trunk)
  2. PCA basis from attention weight trajectory
  3. Project commutator onto PCA manifold → integrability test
  4. Per-task commutator: separate batches from add vs mul loss
  5. Cross-task commutator: batch A = add loss, batch B = mul loss

Produces:
  figMT_J — Commutator defect (total) over training
  figMT_K — Integrability: resid/full ratio over training
  figMT_L — Cross-task commutator (add-grad vs mul-grad)
  figMT_M — Defect × integrability combined
  figMT_N — Attention weight fraction of commutator
"""

import math, sys, random
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

# Local import
from train_multitask import (
    MultiTaskConfig, MultiTaskTransformer, build_dataset, sample_batch,
    get_device, extract_attn_matrices,
)

RESULTS_DIR = Path(__file__).parent / "results"
PLOT_DIR = Path(__file__).parent / "plots"
WEIGHT_KEYS = ["WQ", "WK", "WV", "WO"]
COMM_K = 9
COMM_ETA = 1e-3
N_PCA_COMPONENTS = 2


# ═══════════════════════════════════════════════════════════════════════════
# Commutator functions (adapted for multi-task)
# ═══════════════════════════════════════════════════════════════════════════

def flatten_model_params(model):
    return torch.cat([p.detach().flatten() for p in model.parameters() if p.requires_grad])


def _param_offsets(model):
    offsets = {}
    cursor = 0
    for p in model.parameters():
        if not p.requires_grad:
            continue
        offsets[id(p)] = cursor
        cursor += p.numel()
    return offsets, cursor


def _write_params(model, theta):
    with torch.no_grad():
        offset = 0
        for p in model.parameters():
            if not p.requires_grad:
                continue
            n = p.numel()
            p.copy_(theta[offset:offset+n].view_as(p))
            offset += n


def _batch_grad_total(model, a, b, y_add, y_mul):
    """Gradient of combined loss = CE(add) + CE(mul)."""
    model.zero_grad(set_to_none=True)
    logits_add, logits_mul = model(a, b)
    loss = F.cross_entropy(logits_add, y_add) + F.cross_entropy(logits_mul, y_mul)
    loss.backward()
    return torch.cat([
        (p.grad if p.grad is not None else torch.zeros_like(p)).flatten()
        for p in model.parameters() if p.requires_grad
    ])


def _batch_grad_single(model, a, b, target, task="add"):
    """Gradient of single-task loss."""
    model.zero_grad(set_to_none=True)
    logits_add, logits_mul = model(a, b)
    if task == "add":
        loss = F.cross_entropy(logits_add, target)
    else:
        loss = F.cross_entropy(logits_mul, target)
    loss.backward()
    return torch.cat([
        (p.grad if p.grad is not None else torch.zeros_like(p)).flatten()
        for p in model.parameters() if p.requires_grad
    ])


def commutator_defect(model, batch_fn, device, eta=1e-3, eps=1e-12, mode="total"):
    """
    Commutator defect. mode can be:
      - "total": both batches use combined (add+mul) loss
      - "cross": batch A uses add loss, batch B uses mul loss
    """
    was_training = model.training
    model.train()

    a1, b1, ya1, ym1 = batch_fn()
    a2, b2, ya2, ym2 = batch_fn()

    theta0 = flatten_model_params(model)

    if mode == "total":
        gA = _batch_grad_total(model, a1, b1, ya1, ym1)
        gB = _batch_grad_total(model, a2, b2, ya2, ym2)
    elif mode == "cross":
        gA = _batch_grad_single(model, a1, b1, ya1, task="add")
        gB = _batch_grad_single(model, a2, b2, ym2, task="mul")
    else:
        raise ValueError(f"Unknown mode: {mode}")

    grad_cos = (gA @ gB) / (gA.norm() * gB.norm() + eps)

    # Path AB
    _write_params(model, theta0 - eta * gA)
    if mode == "total":
        gB1 = _batch_grad_total(model, a2, b2, ya2, ym2)
    else:
        gB1 = _batch_grad_single(model, a2, b2, ym2, task="mul")
    thetaAB = theta0 - eta * gA - eta * gB1

    # Path BA
    _write_params(model, theta0 - eta * gB)
    if mode == "total":
        gA1 = _batch_grad_total(model, a1, b1, ya1, ym1)
    else:
        gA1 = _batch_grad_single(model, a1, b1, ya1, task="add")
    thetaBA = theta0 - eta * gB - eta * gA1

    _write_params(model, theta0)
    if not was_training:
        model.eval()

    normA = (eta * gA).norm()
    normB = (eta * gB).norm()
    delta = thetaAB - thetaBA
    defect = (delta.norm() / (normA * normB + eps)).item()

    return defect, delta.detach(), grad_cos.item(), normA.detach(), normB.detach()


def commutator_defect_median(model, batch_fn, device, K=9, eta=1e-3, mode="total"):
    Ds, deltas, gcoss = [], [], []
    for _ in range(K):
        D, delta, gcos, nA, nB = commutator_defect(model, batch_fn, device, eta=eta, mode=mode)
        Ds.append(D)
        deltas.append(delta)
        gcoss.append(gcos)

    Ds_t = torch.tensor(Ds)
    med_idx = Ds_t.argsort()[len(Ds_t)//2]
    return {
        "median": Ds_t.median().item(),
        "p90": Ds_t.quantile(0.9).item(),
        "raw": Ds,
        "median_delta": deltas[med_idx],
        "gcoss": gcoss,
    }


def projected_commutator(delta, B, normA, normB, eps=1e-12):
    delta = delta.reshape(-1)
    if B is None or delta.numel() != B.shape[0]:
        full_val = (delta.norm() / (normA * normB + eps)).item()
        return {"proj": float("nan"), "resid": float("nan"), "full": full_val}
    coeffs = B.T @ delta
    proj = B @ coeffs
    resid = delta - proj
    scale = normA * normB + eps
    return {
        "proj": (proj.norm() / scale).item(),
        "resid": (resid.norm() / scale).item(),
        "full": (delta.norm() / scale).item(),
    }


# ═══════════════════════════════════════════════════════════════════════════
# PCA basis construction (same as grok_commutator_analysis.py)
# ═══════════════════════════════════════════════════════════════════════════

def build_pca_basis(model, attn_logs, n_components=2, device="cpu"):
    offsets, total_params = _param_offsets(model)
    basis_vecs = []

    for layer_idx, layer in enumerate(model.encoder.layers):
        attn_mod = layer.self_attn
        d = attn_mod.embed_dim
        in_proj_id = id(attn_mod.in_proj_weight)
        out_proj_id = id(attn_mod.out_proj.weight)

        if in_proj_id not in offsets or out_proj_id not in offsets:
            continue

        in_proj_offset = offsets[in_proj_id]
        out_proj_offset = offsets[out_proj_id]

        for wkey, local_start in [("WQ", 0), ("WK", d*d), ("WV", 2*d*d)]:
            _, mats = collect_trajectory(attn_logs, layer_idx, wkey)
            pca = pca_on_trajectory(mats, top_k=n_components)
            if pca is None:
                continue
            n_avail = min(n_components, len(pca["components"]))
            for ci in range(n_avail):
                direction = torch.from_numpy(pca["components"][ci]).float()
                gv = torch.zeros(total_params, device=device)
                start = in_proj_offset + local_start
                gv[start:start + d*d] = direction.to(device)
                basis_vecs.append(gv)

        _, mats = collect_trajectory(attn_logs, layer_idx, "WO")
        pca = pca_on_trajectory(mats, top_k=n_components)
        if pca is None:
            continue
        n_avail = min(n_components, len(pca["components"]))
        for ci in range(n_avail):
            direction = torch.from_numpy(pca["components"][ci]).float()
            gv = torch.zeros(total_params, device=device)
            gv[out_proj_offset:out_proj_offset + d*d] = direction.to(device)
            basis_vecs.append(gv)

    if not basis_vecs:
        return None
    B = torch.stack(basis_vecs, dim=1)
    B_ortho, _ = torch.linalg.qr(B.cpu(), mode="reduced")
    return B_ortho.to(device)


def attn_weight_mask(model):
    offsets, total_params = _param_offsets(model)
    mask = torch.zeros(total_params, dtype=torch.bool)
    for layer in model.encoder.layers:
        attn_mod = layer.self_attn
        for p in [attn_mod.in_proj_weight, attn_mod.out_proj.weight]:
            if p.requires_grad and id(p) in offsets:
                start = offsets[id(p)]
                mask[start:start + p.numel()] = True
        for p in [attn_mod.in_proj_bias, attn_mod.out_proj.bias]:
            if p is not None and p.requires_grad and id(p) in offsets:
                start = offsets[id(p)]
                mask[start:start + p.numel()] = True
    return mask


# ═══════════════════════════════════════════════════════════════════════════
# Per-checkpoint measurement
# ═══════════════════════════════════════════════════════════════════════════

def measure_at_checkpoints(cfg, checkpoints, attn_logs, train_pairs, B,
                           K=9, eta=1e-3, modes=("total", "cross")):
    device = get_device()
    p = cfg.P

    model = MultiTaskTransformer(cfg).to(device)
    amask = attn_weight_mask(model)

    def batch_fn():
        return sample_batch(train_pairs, cfg.BATCH_SIZE, p, device)

    results = {mode: [] for mode in modes}
    total_ckpts = len(checkpoints)

    for ci, (step, sd) in enumerate(checkpoints):
        model.load_state_dict(sd)
        model.to(device)

        for mode in modes:
            out = commutator_defect_median(model, batch_fn, device, K=K, eta=eta, mode=mode)

            D, delta, gcos, normA, normB = commutator_defect(
                model, batch_fn, device, eta=eta, mode=mode
            )
            delta_cpu = delta.cpu()
            normA_cpu = normA.cpu()
            normB_cpu = normB.cpu()

            pc = projected_commutator(delta_cpu, B.cpu() if B is not None else None,
                                      normA_cpu, normB_cpu)

            delta_full_norm = delta_cpu.norm().item()
            attn_frac = (delta_cpu[amask].norm().item() / (delta_full_norm + 1e-15))

            results[mode].append({
                "step": step,
                "defect_median": out["median"],
                "defect_p90": out["p90"],
                "grad_cos": np.mean(out["gcoss"]),
                "proj": pc["proj"],
                "resid": pc["resid"],
                "full": pc["full"],
                "attn_frac": attn_frac,
            })

        if (ci+1) % 10 == 0 or ci == total_ckpts - 1:
            r_total = results["total"][-1]
            rf = r_total["resid"] / (r_total["full"] + 1e-15)
            r_cross = results["cross"][-1] if "cross" in results else r_total
            print(f"    ckpt {ci+1}/{total_ckpts}: step={step}, "
                  f"defect={r_total['defect_median']:.4f}, "
                  f"resid/full={rf:.1%}, "
                  f"cross_defect={r_cross['defect_median']:.4f}")

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Figures
# ═══════════════════════════════════════════════════════════════════════════

def plot_figures(comm_results, data, seed):
    # ── Figure MT_J: Commutator defect over training ──────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))

    for mode, color, label in [("total", "#1a5276", "Total (add+mul) loss"),
                                ("cross", "#c0392b", "Cross-task (add vs mul)")]:
        if mode not in comm_results:
            continue
        comm = comm_results[mode]
        steps = [c["step"] for c in comm]
        defs = [c["defect_median"] for c in comm]
        ax.plot(steps, defs, label=label, color=color, lw=2)

    if data["grok_step_add"]:
        ax.axvline(data["grok_step_add"], color="#1f77b4", ls=":", alpha=0.7, label=f"add groks @{data['grok_step_add']}")
    if data["grok_step_mul"]:
        ax.axvline(data["grok_step_mul"], color="#d62728", ls=":", alpha=0.7, label=f"mul groks @{data['grok_step_mul']}")

    ax.set_xlabel("Training step", fontsize=12)
    ax.set_ylabel("Commutator defect (median K=9)", fontsize=12)
    ax.set_title(f"Multi-Task Commutator Defect (seed={seed})", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / f"figMT_J_defect_s{seed}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved figMT_J_defect_s{seed}.png")

    # ── Figure MT_K: Integrability ──────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for idx, (mode, title) in enumerate([("total", "Total Loss Commutator"),
                                          ("cross", "Cross-Task Commutator")]):
        ax = axes[idx]
        if mode not in comm_results:
            continue
        comm = comm_results[mode]
        steps = [c["step"] for c in comm]
        resid_fracs = [c["resid"] / (c["full"] + 1e-15) for c in comm]
        proj_fracs = [c["proj"] / (c["full"] + 1e-15) for c in comm]

        ax.plot(steps, resid_fracs, label="Residual (perp PCA)", lw=2.5, color="#e74c3c")
        ax.plot(steps, proj_fracs, label="Projected (par PCA)", lw=2, color="#27ae60", ls="--")
        ax.axhline(1.0, color="gray", ls=":", alpha=0.3)
        ax.axhline(0.0, color="gray", ls=":", alpha=0.3)
        if data["grok_step_add"]:
            ax.axvline(data["grok_step_add"], color="#1f77b4", ls=":", alpha=0.5)
        if data["grok_step_mul"]:
            ax.axvline(data["grok_step_mul"], color="#d62728", ls=":", alpha=0.5)
        ax.set_xlabel("Training step")
        ax.set_ylabel("Fraction of ||commutator||")
        ax.set_title(title)
        ax.set_ylim(-0.05, 1.15)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    fig.suptitle(f"Integrability of PCA Manifold (seed={seed})\n"
                 f"(resid/full near 100% means curvature is orthogonal to execution manifold)",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / f"figMT_K_integrability_s{seed}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved figMT_K_integrability_s{seed}.png")

    # ── Figure MT_L: Cross-task defect vs total defect ────────────────────
    if "total" in comm_results and "cross" in comm_results:
        fig, ax = plt.subplots(figsize=(10, 5))
        comm_t = comm_results["total"]
        comm_c = comm_results["cross"]
        steps_t = [c["step"] for c in comm_t]
        steps_c = [c["step"] for c in comm_c]

        ratio = []
        for ct, cc in zip(comm_t, comm_c):
            if ct["defect_median"] > 1e-15:
                ratio.append(cc["defect_median"] / ct["defect_median"])
            else:
                ratio.append(float("nan"))

        ax.plot(steps_t, ratio, color="#8e44ad", lw=2, label="cross / total defect ratio")
        ax.axhline(1.0, color="gray", ls="--", alpha=0.5)
        if data["grok_step_add"]:
            ax.axvline(data["grok_step_add"], color="#1f77b4", ls=":", alpha=0.7, label=f"add groks")
        if data["grok_step_mul"]:
            ax.axvline(data["grok_step_mul"], color="#d62728", ls=":", alpha=0.7, label=f"mul groks")
        ax.set_xlabel("Training step")
        ax.set_ylabel("Cross-task / Total defect ratio")
        ax.set_title(f"Cross-Task vs Total Commutator Defect (seed={seed})\n"
                     f"(ratio > 1 means task gradients interfere more than random batches)")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(PLOT_DIR / f"figMT_L_cross_vs_total_s{seed}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved figMT_L_cross_vs_total_s{seed}.png")

    # ── Figure MT_M: Defect × integrability combined ──────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for idx, (mode, title) in enumerate([("total", "Total"), ("cross", "Cross-task")]):
        ax = axes[idx]
        if mode not in comm_results:
            continue
        comm = comm_results[mode]
        steps = [c["step"] for c in comm]
        defs = [c["defect_median"] for c in comm]
        resid_fracs = [c["resid"] / (c["full"] + 1e-15) * 100 for c in comm]

        color1, color2 = "#1a5276", "#e74c3c"
        ax.plot(steps, defs, label="Defect (left)", lw=2, color=color1)
        ax.set_ylabel("Commutator defect", color=color1)
        ax.tick_params(axis="y", labelcolor=color1)
        ax.set_yscale("log")

        ax2 = ax.twinx()
        ax2.plot(steps, resid_fracs, label="Resid % (right)", lw=2, color=color2, ls="--")
        ax2.set_ylabel("Residual fraction (%)", color=color2)
        ax2.tick_params(axis="y", labelcolor=color2)
        ax2.set_ylim(80, 101)

        ax.set_title(f"{title} Commutator")
        ax.set_xlabel("Training step")
        ax.grid(alpha=0.3)
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper left")

    fig.suptitle(f"Defect + Integrability (seed={seed})", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / f"figMT_M_combined_s{seed}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved figMT_M_combined_s{seed}.png")

    # ── Figure MT_N: Grad cosine similarity ──────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    for mode, color, label in [("total", "#1a5276", "Total (same-loss batches)"),
                                ("cross", "#c0392b", "Cross-task (add vs mul grad)")]:
        if mode not in comm_results:
            continue
        comm = comm_results[mode]
        steps = [c["step"] for c in comm]
        gcoss = [c["grad_cos"] for c in comm]
        ax.plot(steps, gcoss, label=label, color=color, lw=2)

    ax.axhline(0, color="gray", ls="--", alpha=0.3)
    if data["grok_step_add"]:
        ax.axvline(data["grok_step_add"], color="#1f77b4", ls=":", alpha=0.5)
    if data["grok_step_mul"]:
        ax.axvline(data["grok_step_mul"], color="#d62728", ls=":", alpha=0.5)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Gradient cosine similarity")
    ax.set_title(f"Gradient Alignment Over Training (seed={seed})")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / f"figMT_N_grad_cos_s{seed}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved figMT_N_grad_cos_s{seed}.png")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    PLOT_DIR.mkdir(exist_ok=True)
    device = get_device()
    print(f"Device: {device}")

    seeds = [42, 137, 2024]
    available = [s for s in seeds if (RESULTS_DIR / f"multitask_s{s}.pt").exists()]
    if not available:
        print("No results found. Run train_multitask.py first.")
        sys.exit(1)

    all_comm_results = {}

    for seed in available:
        print(f"\n{'='*70}")
        print(f"  Commutator analysis: seed={seed}")
        print(f"{'='*70}")

        data = torch.load(RESULTS_DIR / f"multitask_s{seed}.pt",
                          map_location="cpu", weights_only=False)
        cfg_dict = data["cfg"]
        cfg = MultiTaskConfig(**cfg_dict)

        checkpoints = data["checkpoints"]
        attn_logs = data["attn_logs"]
        train_pairs, _ = build_dataset(cfg.P, cfg.TRAIN_FRACTION, cfg.SEED)

        # Subsample checkpoints for speed (every 5th)
        if len(checkpoints) > 100:
            idx = np.linspace(0, len(checkpoints)-1, 100, dtype=int)
            checkpoints = [checkpoints[i] for i in idx]
        print(f"  Using {len(checkpoints)} checkpoints, {len(attn_logs)} attn snaps")

        # Build PCA basis
        print(f"  Building PCA basis (top-{N_PCA_COMPONENTS})...")
        model = MultiTaskTransformer(cfg)
        B = build_pca_basis(model, attn_logs, n_components=N_PCA_COMPONENTS, device="cpu")
        if B is not None:
            print(f"  Basis: {B.shape} ({B.shape[1]} directions)")
        else:
            print(f"  WARNING: no PCA basis")

        # Measure commutators
        print(f"  Measuring commutators...")
        comm_results = measure_at_checkpoints(
            cfg, checkpoints, attn_logs, train_pairs, B,
            K=COMM_K, eta=COMM_ETA, modes=("total", "cross")
        )

        all_comm_results[seed] = comm_results

        # Plot
        plot_figures(comm_results, data, seed)

    # Save results
    save_path = PLOT_DIR / "commutator_results.pt"
    torch.save(all_comm_results, save_path)
    print(f"\n  Saved all results to {save_path}")

    # Summary
    print(f"\n{'='*70}")
    print("  COMMUTATOR SUMMARY")
    print(f"{'='*70}")
    for seed, comm in sorted(all_comm_results.items()):
        for mode in ["total", "cross"]:
            if mode not in comm or not comm[mode]:
                continue
            last = comm[mode][-1]
            rf = last["resid"] / (last["full"] + 1e-15)
            print(f"  seed={seed} {mode:>6s}: defect={last['defect_median']:.4f}, "
                  f"resid/full={rf:.1%}, grad_cos={last['grad_cos']:.3f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
