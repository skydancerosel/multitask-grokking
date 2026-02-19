#!/usr/bin/env python3
"""
Commutator / curvature analysis for tri-task grokking.

Adapts modadd_modmul/commutator_analysis.py for the 3-task setting:
  (x+y) mod p, (x*y) mod p, (x^2+y^2) mod p

Modes:
  - "total": both batches use combined loss (add+mul+sq)
  - "cross_add_mul": batch A = add loss, batch B = mul loss
  - "cross_add_sq":  batch A = add loss, batch B = sq loss
  - "cross_mul_sq":  batch A = mul loss, batch B = sq loss

Produces:
  figTT_J — Commutator defect (total + cross) over training
  figTT_K — Integrability: resid/full ratio over training
  figTT_L — Cross-task commutator comparison (3 pairwise)
  figTT_M — Defect + integrability combined
  figTT_N — Gradient cosine similarity
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

sys.path.insert(0, str(Path(__file__).parent))
from train_tritask import (
    TriTaskConfig, TriTaskTransformer, build_dataset, sample_batch,
    get_device, extract_attn_matrices,
)

RESULTS_DIR = Path(__file__).parent / "results"
PLOT_DIR = Path(__file__).parent / "plots"
WEIGHT_KEYS = ["WQ", "WK", "WV", "WO"]
COMM_K = 9
COMM_ETA = 1e-3
N_PCA_COMPONENTS = 3    # 3 tasks → rank-3 manifold expected
TASK_NAMES = ["add", "mul", "sq"]
TASK_COLORS = {"add": "#1f77b4", "mul": "#d62728", "sq": "#2ca02c"}
SEEDS = [42, 137, 2024]
WD_VALUES = [1.0, 0.5, 0.1, 0.0]
WD_COLORS = {1.0: "#2ca02c", 0.5: "#1f77b4", 0.1: "#ff7f0e", 0.0: "#d62728"}

CROSS_MODES = [
    ("cross_add_mul", "add", "mul", "#9467bd"),
    ("cross_add_sq",  "add", "sq",  "#ff7f0e"),
    ("cross_mul_sq",  "mul", "sq",  "#8c564b"),
]


# ═══════════════════════════════════════════════════════════════════════════
# Commutator functions
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


def _batch_grad_total(model, a, b, y_add, y_mul, y_sq):
    """Gradient of combined loss = CE(add) + CE(mul) + CE(sq)."""
    model.zero_grad(set_to_none=True)
    logits_add, logits_mul, logits_sq = model(a, b)
    loss = (F.cross_entropy(logits_add, y_add) +
            F.cross_entropy(logits_mul, y_mul) +
            F.cross_entropy(logits_sq, y_sq))
    loss.backward()
    return torch.cat([
        (p.grad if p.grad is not None else torch.zeros_like(p)).flatten()
        for p in model.parameters() if p.requires_grad
    ])


def _batch_grad_single(model, a, b, y_add, y_mul, y_sq, task="add"):
    """Gradient of single-task loss."""
    model.zero_grad(set_to_none=True)
    logits_add, logits_mul, logits_sq = model(a, b)
    if task == "add":
        loss = F.cross_entropy(logits_add, y_add)
    elif task == "mul":
        loss = F.cross_entropy(logits_mul, y_mul)
    elif task == "sq":
        loss = F.cross_entropy(logits_sq, y_sq)
    else:
        raise ValueError(f"Unknown task: {task}")
    loss.backward()
    return torch.cat([
        (p.grad if p.grad is not None else torch.zeros_like(p)).flatten()
        for p in model.parameters() if p.requires_grad
    ])


def _sample_batch_with_targets(train_pairs, bs, p, device):
    """Sample batch, return (a, b, y_add, y_mul, y_sq)."""
    idx = np.random.randint(0, len(train_pairs), size=bs)
    ab = np.array([train_pairs[i] for i in idx], dtype=np.int64)
    a = torch.tensor(ab[:, 0], device=device)
    b = torch.tensor(ab[:, 1], device=device)
    y_add = (a + b) % p
    y_mul = (a * b) % p
    y_sq  = (a * a + b * b) % p
    return a, b, y_add, y_mul, y_sq


def commutator_defect(model, batch_fn, device, eta=1e-3, eps=1e-12,
                       mode="total", taskA="add", taskB="mul"):
    """
    Commutator defect.
    mode: "total" or "cross" (cross uses taskA for batch A, taskB for batch B)
    """
    was_training = model.training
    model.train()

    a1, b1, ya1, ym1, ys1 = batch_fn()
    a2, b2, ya2, ym2, ys2 = batch_fn()

    theta0 = flatten_model_params(model)

    if mode == "total":
        gA = _batch_grad_total(model, a1, b1, ya1, ym1, ys1)
        gB = _batch_grad_total(model, a2, b2, ya2, ym2, ys2)
    elif mode == "cross":
        gA = _batch_grad_single(model, a1, b1, ya1, ym1, ys1, task=taskA)
        gB = _batch_grad_single(model, a2, b2, ya2, ym2, ys2, task=taskB)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    grad_cos = (gA @ gB) / (gA.norm() * gB.norm() + eps)

    # Path AB
    _write_params(model, theta0 - eta * gA)
    if mode == "total":
        gB1 = _batch_grad_total(model, a2, b2, ya2, ym2, ys2)
    else:
        gB1 = _batch_grad_single(model, a2, b2, ya2, ym2, ys2, task=taskB)
    thetaAB = theta0 - eta * gA - eta * gB1

    # Path BA
    _write_params(model, theta0 - eta * gB)
    if mode == "total":
        gA1 = _batch_grad_total(model, a1, b1, ya1, ym1, ys1)
    else:
        gA1 = _batch_grad_single(model, a1, b1, ya1, ym1, ys1, task=taskA)
    thetaBA = theta0 - eta * gB - eta * gA1

    _write_params(model, theta0)
    if not was_training:
        model.eval()

    normA = (eta * gA).norm()
    normB = (eta * gB).norm()
    delta = thetaAB - thetaBA
    defect = (delta.norm() / (normA * normB + eps)).item()

    return defect, delta.detach(), grad_cos.item(), normA.detach(), normB.detach()


def commutator_defect_median(model, batch_fn, device, K=9, eta=1e-3,
                              mode="total", taskA="add", taskB="mul"):
    Ds, deltas, gcoss = [], [], []
    for _ in range(K):
        D, delta, gcos, nA, nB = commutator_defect(
            model, batch_fn, device, eta=eta, mode=mode, taskA=taskA, taskB=taskB
        )
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
# PCA basis construction
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
                           K=9, eta=1e-3):
    device = get_device()
    p = cfg.P

    model = TriTaskTransformer(cfg).to(device)
    amask = attn_weight_mask(model)

    def batch_fn():
        return _sample_batch_with_targets(train_pairs, cfg.BATCH_SIZE, p, device)

    # All modes to compute
    all_modes = [("total", "total", None, None)]
    for cross_name, taskA, taskB, _ in CROSS_MODES:
        all_modes.append((cross_name, "cross", taskA, taskB))

    results = {m[0]: [] for m in all_modes}
    total_ckpts = len(checkpoints)

    for ci, (step, sd) in enumerate(checkpoints):
        model.load_state_dict(sd)
        model.to(device)

        for mode_name, mode_type, taskA, taskB in all_modes:
            out = commutator_defect_median(
                model, batch_fn, device, K=K, eta=eta,
                mode=mode_type, taskA=taskA or "add", taskB=taskB or "mul"
            )

            D, delta, gcos, normA, normB = commutator_defect(
                model, batch_fn, device, eta=eta,
                mode=mode_type, taskA=taskA or "add", taskB=taskB or "mul"
            )
            delta_cpu = delta.cpu()
            normA_cpu = normA.cpu()
            normB_cpu = normB.cpu()

            pc = projected_commutator(delta_cpu, B.cpu() if B is not None else None,
                                      normA_cpu, normB_cpu)

            delta_full_norm = delta_cpu.norm().item()
            attn_frac = (delta_cpu[amask].norm().item() / (delta_full_norm + 1e-15))

            results[mode_name].append({
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
            print(f"    ckpt {ci+1}/{total_ckpts}: step={step}, "
                  f"defect={r_total['defect_median']:.4f}, "
                  f"resid/full={rf:.1%}")

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Figures
# ═══════════════════════════════════════════════════════════════════════════

def plot_figures(comm_results, data, seed):
    grok_step = data.get("grok_step", {})

    def add_grok_lines(ax):
        for t in TASK_NAMES:
            gs = grok_step.get(t)
            if gs:
                ax.axvline(gs, color=TASK_COLORS[t], ls=":", alpha=0.5,
                           label=f"{t} groks @{gs}")

    # ── Figure TT_J: Commutator defect over training ─────────────────────
    fig, ax = plt.subplots(figsize=(12, 5))

    # Total
    comm = comm_results["total"]
    steps = [c["step"] for c in comm]
    defs = [c["defect_median"] for c in comm]
    ax.plot(steps, defs, label="Total (add+mul+sq) loss", color="#1a5276", lw=2.5)

    # Cross-task pairs
    for cross_name, taskA, taskB, color in CROSS_MODES:
        if cross_name not in comm_results:
            continue
        comm_c = comm_results[cross_name]
        defs_c = [c["defect_median"] for c in comm_c]
        ax.plot(steps, defs_c, label=f"Cross: {taskA} vs {taskB}", color=color, lw=2)

    add_grok_lines(ax)
    ax.set_xlabel("Training step", fontsize=12)
    ax.set_ylabel("Commutator defect (median K=9)", fontsize=12)
    ax.set_title(f"Tri-Task Commutator Defect (seed={seed})", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / f"figTT_J_defect_s{seed}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved figTT_J_defect_s{seed}.png")

    # ── Figure TT_K: Integrability ───────────────────────────────────────
    n_panels = 1 + len(CROSS_MODES)
    fig, axes = plt.subplots(1, n_panels, figsize=(6*n_panels, 5))

    panel_specs = [("total", "Total Loss Commutator")]
    for cross_name, taskA, taskB, _ in CROSS_MODES:
        panel_specs.append((cross_name, f"Cross: {taskA} vs {taskB}"))

    for idx, (mode_name, title) in enumerate(panel_specs):
        ax = axes[idx]
        if mode_name not in comm_results:
            continue
        comm = comm_results[mode_name]
        steps = [c["step"] for c in comm]
        resid_fracs = [c["resid"] / (c["full"] + 1e-15) for c in comm]
        proj_fracs = [c["proj"] / (c["full"] + 1e-15) for c in comm]

        ax.plot(steps, resid_fracs, label="Residual (perp PCA)", lw=2.5, color="#e74c3c")
        ax.plot(steps, proj_fracs, label="Projected (par PCA)", lw=2, color="#27ae60", ls="--")
        ax.axhline(1.0, color="gray", ls=":", alpha=0.3)
        ax.axhline(0.0, color="gray", ls=":", alpha=0.3)
        add_grok_lines(ax)
        ax.set_xlabel("Training step")
        ax.set_ylabel("Fraction of ||commutator||")
        ax.set_title(title)
        ax.set_ylim(-0.05, 1.15)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    fig.suptitle(f"Integrability of PCA Manifold (seed={seed})\n"
                 f"(resid/full near 100% = curvature orthogonal to execution manifold)",
                 fontsize=12, y=1.04)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / f"figTT_K_integrability_s{seed}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved figTT_K_integrability_s{seed}.png")

    # ── Figure TT_L: Cross-task defect ratios ────────────────────────────
    if "total" in comm_results:
        fig, ax = plt.subplots(figsize=(12, 5))
        comm_t = comm_results["total"]
        steps = [c["step"] for c in comm_t]

        for cross_name, taskA, taskB, color in CROSS_MODES:
            if cross_name not in comm_results:
                continue
            comm_c = comm_results[cross_name]
            ratio = []
            for ct, cc in zip(comm_t, comm_c):
                if ct["defect_median"] > 1e-15:
                    ratio.append(cc["defect_median"] / ct["defect_median"])
                else:
                    ratio.append(float("nan"))
            ax.plot(steps, ratio, color=color, lw=2,
                    label=f"{taskA} vs {taskB} / total")

        ax.axhline(1.0, color="gray", ls="--", alpha=0.5)
        add_grok_lines(ax)
        ax.set_xlabel("Training step")
        ax.set_ylabel("Cross-task / Total defect ratio")
        ax.set_title(f"Cross-Task vs Total Commutator Defect (seed={seed})\n"
                     f"(ratio > 1 means task gradients interfere more than random batches)")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(PLOT_DIR / f"figTT_L_cross_vs_total_s{seed}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved figTT_L_cross_vs_total_s{seed}.png")

    # ── Figure TT_M: Defect + integrability combined ─────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for idx, (mode_name, title) in enumerate([("total", "Total"),
                                               ("cross_add_mul", "Cross: add vs mul")]):
        ax = axes[idx]
        if mode_name not in comm_results:
            continue
        comm = comm_results[mode_name]
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
    fig.savefig(PLOT_DIR / f"figTT_M_combined_s{seed}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved figTT_M_combined_s{seed}.png")

    # ── Figure TT_N: Gradient cosine similarity ──────────────────────────
    fig, ax = plt.subplots(figsize=(12, 5))

    comm = comm_results["total"]
    steps = [c["step"] for c in comm]
    gcoss = [c["grad_cos"] for c in comm]
    ax.plot(steps, gcoss, label="Total (same-loss batches)", color="#1a5276", lw=2)

    for cross_name, taskA, taskB, color in CROSS_MODES:
        if cross_name not in comm_results:
            continue
        comm_c = comm_results[cross_name]
        gcoss_c = [c["grad_cos"] for c in comm_c]
        ax.plot(steps, gcoss_c, label=f"Cross: {taskA} vs {taskB}", color=color, lw=2)

    ax.axhline(0, color="gray", ls="--", alpha=0.3)
    add_grok_lines(ax)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Gradient cosine similarity")
    ax.set_title(f"Gradient Alignment Over Training (seed={seed})")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / f"figTT_N_grad_cos_s{seed}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved figTT_N_grad_cos_s{seed}.png")


# ═══════════════════════════════════════════════════════════════════════════
# Cross-WD comparison plots
# ═══════════════════════════════════════════════════════════════════════════

def wd_tag(wd):
    return f"wd{wd:.0f}" if wd == int(wd) else f"wd{wd}"


def plot_cross_wd_defect(all_wd_comm):
    """
    Cross-WD comparison of commutator defect.
    """
    for seed in SEEDS:
        seed_data = {wd: all_wd_comm.get((wd, seed)) for wd in WD_VALUES}
        seed_data = {wd: d for wd, d in seed_data.items() if d is not None}
        if len(seed_data) < 2:
            continue

        # ── Figure P1: Defect (total) across WD values ────────────────────
        fig, axes = plt.subplots(1, 2, figsize=(16, 5))

        # Panel 1: defect over training
        ax = axes[0]
        for wd in sorted(seed_data.keys(), reverse=True):
            comm = seed_data[wd].get("total", [])
            if not comm:
                continue
            steps = [c["step"] for c in comm]
            defs = [c["defect_median"] for c in comm]
            ax.plot(steps, defs, color=WD_COLORS[wd], lw=2, label=f"wd={wd}")
        ax.set_xlabel("Training step")
        ax.set_ylabel("Commutator defect (median)")
        ax.set_title("Total Loss Defect")
        ax.set_yscale("log")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

        # Panel 2: resid/full ratio across WD values
        ax = axes[1]
        for wd in sorted(seed_data.keys(), reverse=True):
            comm = seed_data[wd].get("total", [])
            if not comm:
                continue
            steps = [c["step"] for c in comm]
            rf = [c["resid"] / (c["full"] + 1e-15) for c in comm]
            ax.plot(steps, rf, color=WD_COLORS[wd], lw=2, label=f"wd={wd}")
        ax.axhline(1.0, color="gray", ls="--", alpha=0.5)
        ax.set_xlabel("Training step")
        ax.set_ylabel("Residual / Full")
        ax.set_title("Integrability (resid/full)")
        ax.set_ylim(0.9, 1.02)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

        fig.suptitle(f"Commutator Defect Across Weight Decay (seed={seed})",
                     fontsize=13, y=1.02)
        fig.tight_layout()
        fig.savefig(PLOT_DIR / f"figTT_P1_cross_wd_defect_s{seed}.png",
                    dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved figTT_P1_cross_wd_defect_s{seed}.png")

    # ── Figure P2: Summary bar chart — final defect & grad_cos vs WD ─────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: final total defect
    ax = axes[0]
    x = np.arange(len(SEEDS))
    width = 0.18
    for i, wd in enumerate(WD_VALUES):
        vals = []
        for seed in SEEDS:
            comm = all_wd_comm.get((wd, seed), {}).get("total", [])
            vals.append(comm[-1]["defect_median"] if comm else 0)
        ax.bar(x + i * width, vals, width, label=f"wd={wd}",
               color=WD_COLORS[wd], alpha=0.85)
    ax.set_xlabel("Seed")
    ax.set_ylabel("Final total defect")
    ax.set_title("Final Commutator Defect")
    ax.set_xticks(x + 1.5 * width)
    ax.set_xticklabels([f"s{s}" for s in SEEDS])
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # Panel 2: final grad_cos (total)
    ax = axes[1]
    for i, wd in enumerate(WD_VALUES):
        vals = []
        for seed in SEEDS:
            comm = all_wd_comm.get((wd, seed), {}).get("total", [])
            vals.append(comm[-1]["grad_cos"] if comm else 0)
        ax.bar(x + i * width, vals, width, label=f"wd={wd}",
               color=WD_COLORS[wd], alpha=0.85)
    ax.set_xlabel("Seed")
    ax.set_ylabel("Final gradient cosine")
    ax.set_title("Final Gradient Alignment (total loss)")
    ax.set_xticks(x + 1.5 * width)
    ax.set_xticklabels([f"s{s}" for s in SEEDS])
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Commutator Summary Across Weight Decay", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "figTT_P2_wd_summary.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved figTT_P2_wd_summary.png")

    # ── Figure P3: Cross-task defect across WD (one seed) ────────────────
    for seed in SEEDS:
        seed_data = {wd: all_wd_comm.get((wd, seed)) for wd in WD_VALUES}
        seed_data = {wd: d for wd, d in seed_data.items() if d is not None}
        if len(seed_data) < 2:
            continue

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        for pi, (cross_name, taskA, taskB, _) in enumerate(CROSS_MODES):
            ax = axes[pi]
            for wd in sorted(seed_data.keys(), reverse=True):
                comm = seed_data[wd].get(cross_name, [])
                if not comm:
                    continue
                steps = [c["step"] for c in comm]
                defs = [c["defect_median"] for c in comm]
                ax.plot(steps, defs, color=WD_COLORS[wd], lw=2, label=f"wd={wd}")
            ax.set_xlabel("Training step")
            ax.set_ylabel("Defect (median)")
            ax.set_title(f"Cross: {taskA} vs {taskB}")
            ax.set_yscale("log")
            ax.legend(fontsize=9)
            ax.grid(alpha=0.3)

        fig.suptitle(f"Cross-Task Defect Across WD (seed={seed})", fontsize=13, y=1.02)
        fig.tight_layout()
        fig.savefig(PLOT_DIR / f"figTT_P3_cross_task_wd_s{seed}.png",
                    dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved figTT_P3_cross_task_wd_s{seed}.png")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    PLOT_DIR.mkdir(exist_ok=True)
    device = get_device()
    print(f"Device: {device}")

    all_wd_comm = {}   # keyed by (wd, seed)

    for wd in WD_VALUES:
        for seed in SEEDS:
            tag = wd_tag(wd)
            fpath = RESULTS_DIR / f"tritask_{tag}_s{seed}.pt"

            if not fpath.exists():
                print(f"[skip] {fpath.name} not found")
                continue

            print(f"\n{'='*70}")
            print(f"  Commutator analysis: wd={wd}, seed={seed}")
            print(f"{'='*70}")

            data = torch.load(fpath, map_location="cpu", weights_only=False)
            cfg_dict = data["cfg"]
            cfg = TriTaskConfig(**cfg_dict)

            checkpoints = data["checkpoints"]
            attn_logs = data["attn_logs"]
            train_pairs, _ = build_dataset(cfg.P, cfg.TRAIN_FRACTION, cfg.SEED)

            # Subsample checkpoints for speed
            if len(checkpoints) > 80:
                idx = np.linspace(0, len(checkpoints)-1, 80, dtype=int)
                checkpoints = [checkpoints[i] for i in idx]
            print(f"  Using {len(checkpoints)} checkpoints, {len(attn_logs)} attn snaps")

            # Build PCA basis from this run's own attn logs
            print(f"  Building PCA basis (top-{N_PCA_COMPONENTS})...")
            model = TriTaskTransformer(cfg)
            B = build_pca_basis(model, attn_logs, n_components=N_PCA_COMPONENTS, device="cpu")
            if B is not None:
                print(f"  Basis: {B.shape} ({B.shape[1]} directions)")
            else:
                print(f"  WARNING: no PCA basis")

            # Measure commutators
            print(f"  Measuring commutators...")
            comm_results = measure_at_checkpoints(
                cfg, checkpoints, attn_logs, train_pairs, B,
                K=COMM_K, eta=COMM_ETA,
            )

            all_wd_comm[(wd, seed)] = comm_results

            # Per-run plots (same style as before)
            plot_figures(comm_results, data, seed)

    # Cross-WD plots
    print(f"\n  Generating cross-WD comparison figures...")
    plot_cross_wd_defect(all_wd_comm)

    # Save results
    save_path = PLOT_DIR / "commutator_results.pt"
    torch.save(all_wd_comm, save_path)
    print(f"\n  Saved all results to {save_path}")

    # Summary
    print(f"\n{'='*70}")
    print("  COMMUTATOR SUMMARY (ALL WD VALUES)")
    print(f"{'='*70}")
    header = (f"  {'wd':>5s}  {'seed':>5s}  {'mode':>15s}  {'defect':>10s}  "
              f"{'resid/full':>10s}  {'grad_cos':>10s}")
    print(header)
    for wd in WD_VALUES:
        for seed in SEEDS:
            comm = all_wd_comm.get((wd, seed))
            if comm is None:
                continue
            for mode_name in ["total", "cross_add_mul", "cross_add_sq", "cross_mul_sq"]:
                if mode_name not in comm or not comm[mode_name]:
                    continue
                last = comm[mode_name][-1]
                rf = last["resid"] / (last["full"] + 1e-15)
                print(f"  {wd:5.1f}  {seed:5d}  {mode_name:>15s}  "
                      f"{last['defect_median']:10.4f}  {rf:10.1%}  "
                      f"{last['grad_cos']:10.3f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
