#!/usr/bin/env python3
"""
Eigenvector reconstruction ablation v3.

Strategy: keep non-attention weights at final values. For attention weights,
project the delta (W_final - W_init) onto the PCA basis and reconstruct.

The PCA basis is built from attention weight trajectories, lives in the
attention-weight subspace of parameter space. We:
  1. Extract the attention-weight subspace of the full delta
  2. Project that subspace delta onto top-k PCA directions
  3. Reconstruct: W_attn = W_attn_init + PCA_proj(delta_attn)
  4. Keep everything else (FFN, embeddings, heads, layernorms) at FINAL values

Sweep k = 1..20 PCA components.
Controls: random rank-k in the attention subspace, init-only attention, full model.
"""

import sys, random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))
from pca_sweep_analysis import pca_on_trajectory, collect_trajectory

from train_multitask import (
    MultiTaskConfig, MultiTaskTransformer, build_dataset,
    get_device, eval_accuracy,
)
from commutator_analysis import _param_offsets

RESULTS_DIR = Path(__file__).parent / "results"
PLOT_DIR = Path(__file__).parent / "plots"
SEEDS = [42, 137, 2024]
WEIGHT_KEYS = ["WQ", "WK", "WV", "WO"]


def get_attn_mask_and_offsets(model):
    """
    Return a boolean mask [P] that is True for attention weight parameters,
    and the total parameter count P.
    """
    offsets, P = _param_offsets(model)
    mask = torch.zeros(P, dtype=torch.bool)
    for layer in model.encoder.layers:
        attn = layer.self_attn
        for p in [attn.in_proj_weight, attn.out_proj.weight]:
            if p.requires_grad and id(p) in offsets:
                start = offsets[id(p)]
                mask[start:start + p.numel()] = True
        # Include biases too
        for p in [attn.in_proj_bias, attn.out_proj.bias]:
            if p is not None and p.requires_grad and id(p) in offsets:
                start = offsets[id(p)]
                mask[start:start + p.numel()] = True
    return mask, P


def build_attn_pca_basis(model, attn_logs, n_components=8):
    """
    Build PCA basis from attention weight trajectories.
    Each basis vector is in the attention-weight subspace (masked to attn params only).
    Returns B_attn: [A, K] where A = number of attention params.
    Also returns the mask for mapping back to full parameter space.
    """
    offsets, P = _param_offsets(model)
    mask, _ = get_attn_mask_and_offsets(model)
    A = mask.sum().item()

    # Build full-space basis vectors first, then extract attention subspace
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
                gv = torch.zeros(P)
                start = in_proj_offset + local_start
                gv[start:start + d*d] = direction
                basis_vecs.append(gv[mask])  # Extract attention subspace

        _, mats = collect_trajectory(attn_logs, layer_idx, "WO")
        pca = pca_on_trajectory(mats, top_k=n_components)
        if pca is None:
            continue
        n_avail = min(n_components, len(pca["components"]))
        for ci in range(n_avail):
            direction = torch.from_numpy(pca["components"][ci]).float()
            gv = torch.zeros(P)
            gv[out_proj_offset:out_proj_offset + d*d] = direction
            basis_vecs.append(gv[mask])

    if not basis_vecs:
        return None, mask
    B = torch.stack(basis_vecs, dim=1)  # [A, K]
    B_ortho, _ = torch.linalg.qr(B, mode="reduced")
    return B_ortho, mask


def flatten_params(model):
    return torch.cat([p.detach().flatten() for p in model.parameters() if p.requires_grad])


def write_params(model, theta):
    with torch.no_grad():
        offset = 0
        for p in model.parameters():
            if not p.requires_grad:
                continue
            n = p.numel()
            p.copy_(theta[offset:offset+n].view_as(p))
            offset += n


def reconstruct_attn_only(theta_init, theta_final, mask, B_attn, keep_k):
    """
    Keep non-attention weights at final.
    Reconstruct attention weights: attn = init + proj_k(final_attn - init_attn)
    """
    theta_recon = theta_final.clone()

    delta_attn = theta_final[mask] - theta_init[mask]  # attention-only delta
    B_k = B_attn[:, :keep_k]
    coeffs = B_k.T @ delta_attn
    delta_recon = B_k @ coeffs

    theta_recon[mask] = theta_init[mask] + delta_recon
    return theta_recon


def measure_accuracy(model, theta, cfg, device):
    write_params(model, theta.to(device))
    model.to(device)
    model.eval()
    train_pairs, test_pairs = build_dataset(cfg.P, cfg.TRAIN_FRACTION, cfg.SEED)
    train_add, train_mul = eval_accuracy(model, train_pairs, cfg, device)
    test_add, test_mul = eval_accuracy(model, test_pairs, cfg, device)
    return {
        "train_add": train_add, "train_mul": train_mul,
        "test_add": test_add, "test_mul": test_mul,
    }


def run_sweep(seed, tag_prefix="multitask"):
    tag = f"{tag_prefix}_s{seed}"
    path = RESULTS_DIR / f"{tag}.pt"
    if not path.exists():
        print(f"  [{tag}] not found")
        return None

    print(f"\n  Loading {tag}...")
    data = torch.load(path, map_location="cpu", weights_only=False)
    cfg = MultiTaskConfig(**data["cfg"])
    device = get_device()
    attn_logs = data["attn_logs"]

    model = MultiTaskTransformer(cfg).to(device)

    # Get theta_init and theta_final
    model.load_state_dict(data["init_state"])
    theta_init = flatten_params(model).cpu()

    model.load_state_dict(data["checkpoints"][-1][1])
    theta_final = flatten_params(model).cpu()

    P = theta_init.numel()

    # Build PCA basis in attention subspace
    model_cpu = MultiTaskTransformer(cfg)
    max_pc = 10
    B_attn, mask = build_attn_pca_basis(model_cpu, attn_logs, n_components=max_pc)
    if B_attn is None:
        return None

    A = mask.sum().item()
    K = B_attn.shape[1]
    print(f"  P={P}, A={A} attn params ({A/P:.1%}), K={K} basis dirs")

    delta_attn = theta_final[mask] - theta_init[mask]
    delta_full = theta_final - theta_init
    print(f"  ||delta_full||={delta_full.norm():.2f}, ||delta_attn||={delta_attn.norm():.2f} ({delta_attn.norm()/delta_full.norm():.1%})")

    # Variance captured within attention subspace
    for k in [1, 2, 3, 5, 8, 16, min(K, 32), K]:
        if k > K:
            continue
        Bk = B_attn[:, :k]
        proj = Bk @ (Bk.T @ delta_attn)
        v = (proj.norm() ** 2) / (delta_attn.norm() ** 2)
        print(f"    top-{k:2d} captures {v:.1%} of attn delta variance")

    # ─── Sweep ───
    k_values = list(range(1, min(K+1, 21)))
    results = []

    # Baselines
    acc = measure_accuracy(model, theta_final, cfg, device)
    results.append({"condition": "Full model", "k": K+1, **acc})
    print(f"  {'Full model':>25s}: add={acc['test_add']:.3f} mul={acc['test_mul']:.3f}")

    acc = measure_accuracy(model, theta_init, cfg, device)
    results.append({"condition": "Init model", "k": 0, **acc})
    print(f"  {'Init model':>25s}: add={acc['test_add']:.3f} mul={acc['test_mul']:.3f}")

    # Attn-at-init, rest at final
    theta_attn_init = theta_final.clone()
    theta_attn_init[mask] = theta_init[mask]
    acc = measure_accuracy(model, theta_attn_init, cfg, device)
    results.append({"condition": "Attn=init, rest=final", "k": 0, **acc})
    print(f"  {'Attn=init, rest=final':>25s}: add={acc['test_add']:.3f} mul={acc['test_mul']:.3f}")

    # PCA sweep
    for k in k_values:
        theta_recon = reconstruct_attn_only(theta_init, theta_final, mask, B_attn, k)
        Bk = B_attn[:, :k]
        var_k = ((Bk.T @ delta_attn).norm() ** 2 / (delta_attn.norm() ** 2)).item()
        acc = measure_accuracy(model, theta_recon, cfg, device)
        results.append({
            "condition": f"PCA top-{k}",
            "k": k, "var_captured": var_k, **acc,
        })
        print(f"  {'PCA top-'+str(k):>25s}: add={acc['test_add']:.3f} mul={acc['test_mul']:.3f}  (var={var_k:.1%})")

    # Random controls in attention subspace
    for k in [1, 2, 5, 16]:
        rng = torch.Generator()
        rng.manual_seed(seed + 77777)
        R = torch.randn(A, k, generator=rng)
        Q, _ = torch.linalg.qr(R, mode="reduced")
        coeffs = Q.T @ delta_attn
        delta_rand = Q @ coeffs
        theta_rand = theta_final.clone()
        theta_rand[mask] = theta_init[mask] + delta_rand
        var_k = (coeffs.norm() ** 2 / (delta_attn.norm() ** 2)).item()
        acc = measure_accuracy(model, theta_rand, cfg, device)
        results.append({
            "condition": f"Random rank-{k}",
            "k": k, "var_captured": var_k, **acc,
        })
        print(f"  {'Random rank-'+str(k):>25s}: add={acc['test_add']:.3f} mul={acc['test_mul']:.3f}  (var={var_k:.1%})")

    return {
        "seed": seed, "results": results, "K": K, "A": A, "P": P,
        "grok_step_add": data["grok_step_add"],
        "grok_step_mul": data["grok_step_mul"],
    }


def plot_accuracy_curve(all_results):
    """figABL3_A: Test accuracy vs k (line plot)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for tidx, (task, label) in enumerate([("test_add", "Mod-Add"), ("test_mul", "Mod-Mul")]):
        ax = axes[tidx]

        # Per-seed PCA curves
        for r in all_results:
            pca = [res for res in r["results"] if res["condition"].startswith("PCA")]
            ks = [res["k"] for res in pca]
            accs = [res[task] for res in pca]
            ax.plot(ks, accs, "o-", alpha=0.4, lw=1.5, ms=4, label=f"PCA s{r['seed']}")

        # Mean PCA curve
        pca_by_k = {}
        for r in all_results:
            for res in r["results"]:
                if res["condition"].startswith("PCA"):
                    pca_by_k.setdefault(res["k"], []).append(res[task])
        ks = sorted(pca_by_k.keys())
        means = [np.mean(pca_by_k[k]) for k in ks]
        ax.plot(ks, means, "k-D", lw=3, ms=6, label="PCA mean", zorder=10)

        # Random controls
        for r in all_results:
            for res in r["results"]:
                if res["condition"].startswith("Random"):
                    ax.scatter(res["k"], res[task], marker="x", color="red",
                               s=80, zorder=8, alpha=0.6)
        ax.scatter([], [], marker="x", color="red", s=80, label="Random control")

        # Baselines
        full = np.mean([res[task] for r in all_results for res in r["results"]
                        if res["condition"] == "Full model"])
        attn_init = np.mean([res[task] for r in all_results for res in r["results"]
                             if res["condition"] == "Attn=init, rest=final"])
        ax.axhline(full, color="blue", ls="--", alpha=0.5, label=f"Full model ({full:.2f})")
        ax.axhline(attn_init, color="orange", ls="--", alpha=0.5,
                   label=f"Attn=init ({attn_init:.2f})")
        ax.axhline(0.98, color="green", ls=":", alpha=0.3)

        ax.set_xlabel("Number of PCA components (k)", fontsize=12)
        ax.set_ylabel(f"Test accuracy ({label})", fontsize=12)
        ax.set_title(label, fontsize=13)
        ax.set_ylim(-0.02, 1.05)
        ax.legend(fontsize=7, loc="center right")
        ax.grid(alpha=0.3)

    fig.suptitle("Attention-Weight Reconstruction Ablation\n"
                 "(non-attention weights kept at final; attention = init + PCA_k(Δ))",
                 fontsize=13, y=1.03)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / f"figABL3_A_accuracy_vs_k.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved figABL3_A_accuracy_vs_k.png")


def plot_bar_summary(all_results):
    """figABL3_B: Bar chart for key conditions."""
    key_conds = ["Init model", "Attn=init, rest=final",
                 "Random rank-1", "PCA top-1", "PCA top-2", "PCA top-3",
                 "PCA top-5", "PCA top-8", "PCA top-10", "PCA top-16", "PCA top-20",
                 "Random rank-16", "Full model"]

    conds_present = []
    add_m, add_s, mul_m, mul_s = [], [], [], []
    for c in key_conds:
        adds = [res["test_add"] for r in all_results for res in r["results"] if res["condition"] == c]
        muls = [res["test_mul"] for r in all_results for res in r["results"] if res["condition"] == c]
        if adds:
            conds_present.append(c)
            add_m.append(np.mean(adds)); add_s.append(np.std(adds))
            mul_m.append(np.mean(muls)); mul_s.append(np.std(muls))

    x = np.arange(len(conds_present))
    w = 0.35
    fig, ax = plt.subplots(figsize=(16, 6))
    b1 = ax.bar(x - w/2, add_m, w, yerr=add_s, capsize=3, label="Add",
                color="#3498db", edgecolor="k", alpha=0.85)
    b2 = ax.bar(x + w/2, mul_m, w, yerr=mul_s, capsize=3, label="Mul",
                color="#e74c3c", edgecolor="k", alpha=0.85)

    for i, c in enumerate(conds_present):
        if "Random" in c or c in ["Init model", "Attn=init, rest=final"]:
            b1[i].set_facecolor("#bdc3c7"); b1[i].set_alpha(0.5)
            b2[i].set_facecolor("#bdc3c7"); b2[i].set_alpha(0.5)

    ax.axhline(0.98, color="green", ls="--", alpha=0.4)
    for bar in list(b1) + list(b2):
        h = bar.get_height()
        if h > 0.03:
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.015,
                    f"{h:.2f}", ha="center", fontsize=7, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(conds_present, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Test accuracy", fontsize=12)
    ax.set_title("Eigenvector Reconstruction Ablation (attn weights only, rest=final)",
                 fontsize=12)
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / f"figABL3_B_bar_summary.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved figABL3_B_bar_summary.png")


def main():
    PLOT_DIR.mkdir(exist_ok=True)
    device = get_device()
    print(f"Device: {device}")

    all_results = []
    for seed in SEEDS:
        r = run_sweep(seed)
        if r is not None:
            all_results.append(r)

    if not all_results:
        print("No results.")
        sys.exit(1)

    save_path = PLOT_DIR / "eigenvector_ablation_v3_results.pt"
    torch.save(all_results, save_path)
    print(f"\nSaved to {save_path}")

    print("\n" + "=" * 70)
    print("  PLOTTING")
    print("=" * 70)
    plot_accuracy_curve(all_results)
    plot_bar_summary(all_results)

    # Summary
    print(f"\n{'='*70}")
    print("  RECONSTRUCTION ABLATION SUMMARY (v3: attn-only PCA, rest=final)")
    print(f"{'='*70}")
    cond_order = ["Init model", "Attn=init, rest=final",
                  "Random rank-1", "PCA top-1", "PCA top-2", "PCA top-3",
                  "PCA top-5", "PCA top-8", "PCA top-10",
                  "PCA top-16", "PCA top-20",
                  "Random rank-5", "Random rank-16", "Full model"]
    print(f"{'Condition':>28s}  {'Add':>12s}  {'Mul':>12s}  {'Var%':>6s}")
    print("-" * 68)
    for cond in cond_order:
        adds, muls, vcs = [], [], []
        for r in all_results:
            for res in r["results"]:
                if res["condition"] == cond:
                    adds.append(res["test_add"])
                    muls.append(res["test_mul"])
                    if "var_captured" in res:
                        vcs.append(res["var_captured"])
        if not adds:
            continue
        vc = f"{np.mean(vcs)*100:.1f}%" if vcs else "—"
        print(f"{cond:>28s}  {np.mean(adds):.3f}±{np.std(adds):.3f}  "
              f"{np.mean(muls):.3f}±{np.std(muls):.3f}  {vc:>6s}")

    print("\nDone.")


if __name__ == "__main__":
    main()
