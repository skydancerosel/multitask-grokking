#!/usr/bin/env python3
"""
Single-snapshot model decomposition at grok time.

Given ONLY θ_init and θ_grok (no training trajectory), decompose Δθ
and test which top-k components are necessary for generalization.

Decomposition methods:
  1. Per-layer SVD: reshape each weight matrix's Δ, keep top-k singular vectors
  2. Block-wise PCA: group parameters by module, do PCA within each block
  3. Spectral decomposition of attention weight matrices (QKV, O)
  4. Random subspace control

For each: reconstruct θ = θ_init + project_k(Δθ), measure test accuracy.
"""

import sys, random
from pathlib import Path
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

from train_multitask import (
    MultiTaskConfig, MultiTaskTransformer, build_dataset,
    get_device, eval_accuracy,
)

RESULTS_DIR = Path(__file__).parent / "results"
PLOT_DIR = Path(__file__).parent / "plots"
SEEDS = [42, 137, 2024]


def measure_accuracy(model, cfg, device):
    """Evaluate model on train/test for both tasks."""
    model.to(device)
    model.eval()
    train_pairs, test_pairs = build_dataset(cfg.P, cfg.TRAIN_FRACTION, cfg.SEED)
    train_add, train_mul = eval_accuracy(model, train_pairs, cfg, device)
    test_add, test_mul = eval_accuracy(model, test_pairs, cfg, device)
    return {
        "train_add": train_add, "train_mul": train_mul,
        "test_add": test_add, "test_mul": test_mul,
    }


def apply_sd(model, sd):
    """Load state dict and return model."""
    model.load_state_dict(sd)
    return model


# ═══════════════════════════════════════════════════════════════════
# Method 1: Per-layer SVD — for 2D weight matrices
# ═══════════════════════════════════════════════════════════════════
def perlayer_svd_reconstruct(init_sd, final_sd, rank):
    """
    For each 2D weight matrix: Δ_W = U S V^T, keep top-rank.
    Non-2D params (biases, layernorms, embeddings): keep at final.
    """
    recon_sd = {}
    total_params = 0
    svd_params = 0

    for key in final_sd:
        delta = final_sd[key] - init_sd[key]
        if delta.dim() == 2 and min(delta.shape) > 1:
            r = min(rank, min(delta.shape))
            U, S, Vh = torch.linalg.svd(delta.float(), full_matrices=False)
            delta_r = (U[:, :r] * S[:r]) @ Vh[:r, :]
            recon_sd[key] = init_sd[key] + delta_r.to(init_sd[key].dtype)
            svd_params += delta.numel()
        else:
            recon_sd[key] = final_sd[key].clone()
        total_params += final_sd[key].numel()

    return recon_sd, svd_params, total_params


# ═══════════════════════════════════════════════════════════════════
# Method 2: Per-layer SVD — ALL params (1D reshaped to 2D)
# ═══════════════════════════════════════════════════════════════════
def perlayer_svd_all_reconstruct(init_sd, final_sd, rank):
    """
    For each parameter tensor, reshape to 2D and do SVD.
    1D params are reshaped to [n, 1], so SVD is trivial.
    Embeddings: [vocab, d] → already 2D.
    """
    recon_sd = {}

    for key in final_sd:
        delta = (final_sd[key] - init_sd[key]).float()
        shape = delta.shape

        if delta.dim() >= 2:
            # Reshape to [first_dim, rest]
            mat = delta.view(shape[0], -1)
        elif delta.dim() == 1:
            mat = delta.view(-1, 1)
        else:
            recon_sd[key] = final_sd[key].clone()
            continue

        r = min(rank, min(mat.shape))
        U, S, Vh = torch.linalg.svd(mat, full_matrices=False)
        mat_r = (U[:, :r] * S[:r]) @ Vh[:r, :]
        delta_r = mat_r.view(shape)
        recon_sd[key] = (init_sd[key].float() + delta_r).to(init_sd[key].dtype)

    return recon_sd


# ═══════════════════════════════════════════════════════════════════
# Method 3: Global Δθ decomposition via random sampling
# ═══════════════════════════════════════════════════════════════════
def global_topk_magnitude(init_sd, final_sd, keep_frac):
    """
    Keep only the top keep_frac of Δθ by absolute magnitude.
    """
    # Flatten all deltas
    deltas = {}
    all_vals = []
    for key in final_sd:
        d = (final_sd[key] - init_sd[key]).float()
        deltas[key] = d
        all_vals.append(d.flatten())

    all_vals = torch.cat(all_vals)
    threshold = torch.quantile(all_vals.abs(), 1.0 - keep_frac)

    recon_sd = {}
    n_kept = 0
    for key in final_sd:
        mask = deltas[key].abs() >= threshold
        delta_pruned = deltas[key] * mask.float()
        recon_sd[key] = (init_sd[key].float() + delta_pruned).to(init_sd[key].dtype)
        n_kept += mask.sum().item()

    return recon_sd, n_kept, all_vals.numel()


# ═══════════════════════════════════════════════════════════════════
# Method 4: Spectral decomposition of attention heads
# ═══════════════════════════════════════════════════════════════════
def attention_spectral_reconstruct(init_sd, final_sd, rank, model):
    """
    Decompose Δ for Q, K, V, O projection matrices via SVD.
    Keep top-rank singular values.
    All other params: keep at final.
    """
    recon_sd = {}
    attn_keys = []

    for key in final_sd:
        is_attn = any(kw in key for kw in [
            'in_proj_weight', 'out_proj.weight',
            'self_attn.in_proj', 'self_attn.out_proj'
        ])

        if is_attn and final_sd[key].dim() == 2:
            delta = (final_sd[key] - init_sd[key]).float()
            r = min(rank, min(delta.shape))
            U, S, Vh = torch.linalg.svd(delta, full_matrices=False)
            delta_r = (U[:, :r] * S[:r]) @ Vh[:r, :]
            recon_sd[key] = (init_sd[key].float() + delta_r).to(init_sd[key].dtype)
            attn_keys.append(key)
        else:
            recon_sd[key] = final_sd[key].clone()

    return recon_sd, attn_keys


# ═══════════════════════════════════════════════════════════════════
# Method 5: Module-level importance (zero out entire modules)
# ═══════════════════════════════════════════════════════════════════
def module_ablation(init_sd, final_sd):
    """
    For each module group, test: keep Δ only for that module,
    everything else at init. Returns importance scores.
    """
    # Group parameters by module
    modules = OrderedDict()
    for key in final_sd:
        parts = key.split('.')
        if 'layers' in parts:
            idx = parts.index('layers')
            module = '.'.join(parts[:idx+2])  # e.g., "encoder.layers.0"
        elif 'embed' in key.lower() or 'token' in key.lower():
            module = "embeddings"
        elif 'head' in key.lower():
            module = "heads"
        elif 'norm' in key.lower():
            module = "norms"
        else:
            module = "other"
        modules.setdefault(module, []).append(key)

    return modules


# ═══════════════════════════════════════════════════════════════════
# Method 6: Layer-wise scaling of Δθ
# ═══════════════════════════════════════════════════════════════════
def layerwise_scaling_reconstruct(init_sd, final_sd, scales):
    """
    Scale Δθ per-module by given factors.
    scales: dict of module_prefix → scale_factor
    """
    recon_sd = {}
    for key in final_sd:
        delta = final_sd[key] - init_sd[key]
        scale = 1.0
        for prefix, s in scales.items():
            if prefix in key:
                scale = s
                break
        recon_sd[key] = init_sd[key] + scale * delta
    return recon_sd


def run_analysis(seed):
    tag = f"multitask_s{seed}"
    path = RESULTS_DIR / f"{tag}.pt"
    if not path.exists():
        print(f"  [{tag}] not found")
        return None

    print(f"\n{'='*65}")
    print(f"  Seed {seed}: {tag}")
    print(f"{'='*65}")

    data = torch.load(path, map_location="cpu", weights_only=False)
    cfg = MultiTaskConfig(**data["cfg"])
    device = get_device()

    model = MultiTaskTransformer(cfg)

    # Get init and final state dicts
    init_sd = data["init_state"]
    final_sd = data["checkpoints"][-1][1]

    # Reference measurements
    apply_sd(model, final_sd)
    ref_final = measure_accuracy(model, cfg, device)
    print(f"  Full model:  add={ref_final['test_add']:.3f}  mul={ref_final['test_mul']:.3f}")

    apply_sd(model, init_sd)
    ref_init = measure_accuracy(model, cfg, device)
    print(f"  Init model:  add={ref_init['test_add']:.3f}  mul={ref_init['test_mul']:.3f}")

    # Compute delta norms by parameter
    print(f"\n  Parameter delta norms:")
    param_info = []
    total_delta_sq = 0
    for key in final_sd:
        d = (final_sd[key] - init_sd[key]).float()
        norm = d.norm().item()
        frac = d.norm().item()**2
        total_delta_sq += frac
        param_info.append((key, d.numel(), norm))

    param_info.sort(key=lambda x: x[2], reverse=True)
    for key, n, norm in param_info[:15]:
        frac = norm**2 / total_delta_sq * 100
        print(f"    {key:50s}  {n:7d} params  ||Δ||={norm:8.3f}  ({frac:5.1f}%)")

    results = {"seed": seed, "ref_final": ref_final, "ref_init": ref_init}

    # ─── Method 1: Per-layer SVD (2D only) ───────────────────────
    print(f"\n  ─── Per-layer SVD (2D weights only, rest=final) ───")
    results["svd_2d"] = {}
    for rank in [1, 2, 3, 5, 10, 20, 32, 64, 128]:
        recon_sd, svd_p, total_p = perlayer_svd_reconstruct(init_sd, final_sd, rank)
        apply_sd(model, recon_sd)
        acc = measure_accuracy(model, cfg, device)
        results["svd_2d"][rank] = acc
        star = " ★" if acc["test_add"] > 0.5 or acc["test_mul"] > 0.5 else ""
        print(f"    rank-{rank:3d}: add={acc['test_add']:.3f}  mul={acc['test_mul']:.3f}{star}")

    # ─── Method 2: Per-layer SVD (all params) ────────────────────
    print(f"\n  ─── Per-layer SVD (ALL params reshaped to 2D) ───")
    results["svd_all"] = {}
    for rank in [1, 2, 3, 5, 10, 20, 32, 64, 128]:
        recon_sd = perlayer_svd_all_reconstruct(init_sd, final_sd, rank)
        apply_sd(model, recon_sd)
        acc = measure_accuracy(model, cfg, device)
        results["svd_all"][rank] = acc
        star = " ★" if acc["test_add"] > 0.5 or acc["test_mul"] > 0.5 else ""
        print(f"    rank-{rank:3d}: add={acc['test_add']:.3f}  mul={acc['test_mul']:.3f}{star}")

    # ─── Method 3: Magnitude pruning ─────────────────────────────
    print(f"\n  ─── Magnitude pruning of Δθ ───")
    results["pruning"] = {}
    for keep in [0.001, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.50, 0.80, 1.0]:
        recon_sd, n_kept, n_total = global_topk_magnitude(init_sd, final_sd, keep)
        apply_sd(model, recon_sd)
        acc = measure_accuracy(model, cfg, device)
        results["pruning"][keep] = {"n_kept": n_kept, **acc}
        star = " ★" if acc["test_add"] > 0.5 or acc["test_mul"] > 0.5 else ""
        print(f"    keep {keep*100:5.1f}% ({n_kept:6d}/{n_total}): "
              f"add={acc['test_add']:.3f}  mul={acc['test_mul']:.3f}{star}")

    # ─── Method 4: Attention spectral ────────────────────────────
    print(f"\n  ─── Attention-only spectral decomposition ───")
    results["attn_spectral"] = {}
    for rank in [1, 2, 3, 5, 10, 20, 32, 64, 128]:
        recon_sd, attn_keys = attention_spectral_reconstruct(init_sd, final_sd, rank, model)
        apply_sd(model, recon_sd)
        acc = measure_accuracy(model, cfg, device)
        results["attn_spectral"][rank] = acc
        star = " ★" if acc["test_add"] > 0.5 or acc["test_mul"] > 0.5 else ""
        print(f"    rank-{rank:3d}: add={acc['test_add']:.3f}  mul={acc['test_mul']:.3f}{star}")
    if attn_keys:
        print(f"    Attention keys: {attn_keys}")

    # ─── Method 5: Module ablation ───────────────────────────────
    print(f"\n  ─── Module ablation (keep one module, rest=init) ───")
    modules = module_ablation(init_sd, final_sd)
    results["module_ablation"] = {}

    for mod_name, keys in modules.items():
        # Everything at init except this module at final
        recon_sd = {k: init_sd[k].clone() for k in init_sd}
        for k in keys:
            recon_sd[k] = final_sd[k].clone()
        apply_sd(model, recon_sd)
        acc = measure_accuracy(model, cfg, device)
        n_params = sum(final_sd[k].numel() for k in keys)
        results["module_ablation"][mod_name] = {"n_params": n_params, **acc}
        star = " ★" if acc["test_add"] > 0.5 or acc["test_mul"] > 0.5 else ""
        print(f"    only={mod_name:30s} ({n_params:7d} params): "
              f"add={acc['test_add']:.3f}  mul={acc['test_mul']:.3f}{star}")

    # Reverse: everything at final except one module at init
    print(f"\n  ─── Module ablation (remove one module, rest=final) ───")
    results["module_remove"] = {}

    for mod_name, keys in modules.items():
        recon_sd = {k: final_sd[k].clone() for k in final_sd}
        for k in keys:
            recon_sd[k] = init_sd[k].clone()
        apply_sd(model, recon_sd)
        acc = measure_accuracy(model, cfg, device)
        n_params = sum(final_sd[k].numel() for k in keys)
        results["module_remove"][mod_name] = {"n_params": n_params, **acc}
        star = " ★" if acc["test_add"] > 0.5 or acc["test_mul"] > 0.5 else ""
        print(f"    rm={mod_name:30s} ({n_params:7d} params): "
              f"add={acc['test_add']:.3f}  mul={acc['test_mul']:.3f}{star}")

    # ─── Method 6: Scale Δθ uniformly ────────────────────────────
    print(f"\n  ─── Uniform scaling of Δθ ───")
    results["uniform_scale"] = {}
    for scale in [0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.5]:
        recon_sd = {}
        for key in final_sd:
            delta = final_sd[key] - init_sd[key]
            recon_sd[key] = init_sd[key] + scale * delta
        apply_sd(model, recon_sd)
        acc = measure_accuracy(model, cfg, device)
        results["uniform_scale"][scale] = acc
        star = " ★" if acc["test_add"] > 0.5 or acc["test_mul"] > 0.5 else ""
        print(f"    scale={scale:.2f}: add={acc['test_add']:.3f}  mul={acc['test_mul']:.3f}{star}")

    return results


def plot_results(all_results):
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    colors = {"42": "#2980b9", "137": "#e74c3c", "2024": "#27ae60"}

    for task_idx, (task, label) in enumerate([("test_add", "Mod-Add"), ("test_mul", "Mod-Mul")]):

        # ─── Plot 1: Per-layer SVD (2D) ───
        ax = axes[0, 0] if task_idx == 0 else None
        if ax:
            for r in all_results:
                s = str(r["seed"])
                ranks = sorted(r["svd_2d"].keys())
                accs = [r["svd_2d"][rk][task] for rk in ranks]
                ax.plot(ranks, accs, "o-", color=colors[s], alpha=0.5, lw=1, ms=3,
                        label=f"s{s} {label}")
            # Mean
            svd_by_r = {}
            for r in all_results:
                for rk, v in r["svd_2d"].items():
                    for t in ["test_add", "test_mul"]:
                        svd_by_r.setdefault((rk, t), []).append(v[t])

            ranks = sorted(set(rk for rk, _ in svd_by_r.keys()))
            for t, ls, lbl in [("test_add", "-", "Add"), ("test_mul", "--", "Mul")]:
                means = [np.mean(svd_by_r[(rk, t)]) for rk in ranks]
                ax.plot(ranks, means, f"D{ls}", color="black", lw=2.5, ms=6,
                        label=f"Mean {lbl}", zorder=10)

            ax.axhline(1/97, color="gray", ls=":", alpha=0.3)
            ax.set_xlabel("SVD rank", fontsize=11)
            ax.set_ylabel("Test accuracy", fontsize=11)
            ax.set_title("Per-layer SVD\n(2D weights only, rest=final)", fontsize=11)
            ax.set_ylim(-0.02, 1.05)
            ax.set_xscale("log", base=2)
            ax.legend(fontsize=7)
            ax.grid(alpha=0.3)

        # ─── Plot 2: Magnitude pruning ───
        ax = axes[0, 1] if task_idx == 0 else None
        if ax:
            for r in all_results:
                s = str(r["seed"])
                keeps = sorted(r["pruning"].keys())
                for t, ls, lbl in [("test_add", "-", "Add"), ("test_mul", "--", "Mul")]:
                    accs = [r["pruning"][k][t] for k in keeps]
                    ax.plot([k*100 for k in keeps], accs, f"o{ls}",
                            color=colors[s], alpha=0.4, lw=1, ms=3)

            prune_by_k = {}
            for r in all_results:
                for k, v in r["pruning"].items():
                    for t in ["test_add", "test_mul"]:
                        prune_by_k.setdefault((k, t), []).append(v[t])
            keeps = sorted(set(k for k, _ in prune_by_k.keys()))
            for t, ls, lbl in [("test_add", "-", "Add"), ("test_mul", "--", "Mul")]:
                means = [np.mean(prune_by_k[(k, t)]) for k in keeps]
                ax.plot([k*100 for k in keeps], means, f"D{ls}", color="black",
                        lw=2.5, ms=6, label=f"Mean {lbl}", zorder=10)

            ax.axhline(1/97, color="gray", ls=":", alpha=0.3)
            ax.set_xlabel("% parameters kept", fontsize=11)
            ax.set_ylabel("Test accuracy", fontsize=11)
            ax.set_title("Magnitude Pruning of Δθ", fontsize=11)
            ax.set_ylim(-0.02, 1.05)
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)

        # ─── Plot 3: Attention spectral ───
        ax = axes[0, 2] if task_idx == 0 else None
        if ax:
            for r in all_results:
                s = str(r["seed"])
                ranks = sorted(r["attn_spectral"].keys())
                for t, ls, lbl in [("test_add", "-", "Add"), ("test_mul", "--", "Mul")]:
                    accs = [r["attn_spectral"][rk][t] for rk in ranks]
                    ax.plot(ranks, accs, f"o{ls}", color=colors[s], alpha=0.4, lw=1, ms=3)

            attn_by_r = {}
            for r in all_results:
                for rk, v in r["attn_spectral"].items():
                    for t in ["test_add", "test_mul"]:
                        attn_by_r.setdefault((rk, t), []).append(v[t])
            ranks = sorted(set(rk for rk, _ in attn_by_r.keys()))
            for t, ls, lbl in [("test_add", "-", "Add"), ("test_mul", "--", "Mul")]:
                means = [np.mean(attn_by_r[(rk, t)]) for rk in ranks]
                ax.plot(ranks, means, f"D{ls}", color="black", lw=2.5, ms=6,
                        label=f"Mean {lbl}", zorder=10)

            ax.axhline(1/97, color="gray", ls=":", alpha=0.3)
            ax.set_xlabel("SVD rank", fontsize=11)
            ax.set_ylabel("Test accuracy", fontsize=11)
            ax.set_title("Attention-only Spectral\n(other params at final)", fontsize=11)
            ax.set_ylim(-0.02, 1.05)
            ax.set_xscale("log", base=2)
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)

    # ─── Plot 4: Module ablation ───
    ax = axes[1, 0]
    mod_names = None
    for r in all_results:
        if mod_names is None:
            mod_names = list(r["module_ablation"].keys())

    x = np.arange(len(mod_names))
    width = 0.12

    for i, r in enumerate(all_results):
        s = str(r["seed"])
        vals_add = [r["module_ablation"][m]["test_add"] for m in mod_names]
        vals_mul = [r["module_ablation"][m]["test_mul"] for m in mod_names]
        ax.bar(x + i*width, vals_add, width, color=colors[s], alpha=0.6)
        ax.bar(x + i*width, vals_mul, width, color=colors[s], alpha=0.3,
               hatch='//')

    ax.set_xticks(x + width)
    ax.set_xticklabels([m.split('.')[-1] if '.' in m else m for m in mod_names],
                       rotation=45, ha='right', fontsize=8)
    ax.set_ylabel("Test accuracy", fontsize=11)
    ax.set_title("Keep ONE module (rest=init)", fontsize=11)
    ax.axhline(1/97, color="gray", ls=":", alpha=0.3)
    ax.grid(alpha=0.3, axis='y')

    # ─── Plot 5: Module removal ───
    ax = axes[1, 1]
    for i, r in enumerate(all_results):
        s = str(r["seed"])
        vals_add = [r["module_remove"][m]["test_add"] for m in mod_names]
        vals_mul = [r["module_remove"][m]["test_mul"] for m in mod_names]
        ax.bar(x + i*width, vals_add, width, color=colors[s], alpha=0.6)
        ax.bar(x + i*width, vals_mul, width, color=colors[s], alpha=0.3,
               hatch='//')

    ax.set_xticks(x + width)
    ax.set_xticklabels([m.split('.')[-1] if '.' in m else m for m in mod_names],
                       rotation=45, ha='right', fontsize=8)
    ax.set_ylabel("Test accuracy", fontsize=11)
    ax.set_title("Remove ONE module (rest=final)", fontsize=11)
    ax.axhline(1/97, color="gray", ls=":", alpha=0.3)
    ax.grid(alpha=0.3, axis='y')

    # ─── Plot 6: Uniform scaling ───
    ax = axes[1, 2]
    for r in all_results:
        s = str(r["seed"])
        scales = sorted(r["uniform_scale"].keys())
        for t, ls, lbl in [("test_add", "-", "Add"), ("test_mul", "--", "Mul")]:
            accs = [r["uniform_scale"][sc][t] for sc in scales]
            ax.plot(scales, accs, f"o{ls}", color=colors[s], alpha=0.4, lw=1, ms=3)

    scale_by_s = {}
    for r in all_results:
        for sc, v in r["uniform_scale"].items():
            for t in ["test_add", "test_mul"]:
                scale_by_s.setdefault((sc, t), []).append(v[t])
    scales = sorted(set(sc for sc, _ in scale_by_s.keys()))
    for t, ls, lbl in [("test_add", "-", "Add"), ("test_mul", "--", "Mul")]:
        means = [np.mean(scale_by_s[(sc, t)]) for sc in scales]
        ax.plot(scales, means, f"D{ls}", color="black", lw=2.5, ms=6,
                label=f"Mean {lbl}", zorder=10)

    ax.axhline(1/97, color="gray", ls=":", alpha=0.3)
    ax.axvline(1.0, color="green", ls="--", alpha=0.3)
    ax.set_xlabel("Scale factor", fontsize=11)
    ax.set_ylabel("Test accuracy", fontsize=11)
    ax.set_title("Uniform Scaling of Δθ\n(θ = θ_init + s·Δθ)", fontsize=11)
    ax.set_ylim(-0.02, 1.05)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.suptitle("Single-Snapshot Decomposition of Grokked Model\n"
                 "(No trajectory information — only θ_init and θ_grok)",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "figSNAP_A_decomposition.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved figSNAP_A_decomposition.png")


def main():
    PLOT_DIR.mkdir(exist_ok=True)
    device = get_device()
    print(f"Device: {device}")

    all_results = []
    for seed in SEEDS:
        r = run_analysis(seed)
        if r is not None:
            all_results.append(r)

    if not all_results:
        print("No results.")
        sys.exit(1)

    # Save
    save_path = PLOT_DIR / "snapshot_decomposition_results.pt"
    torch.save(all_results, save_path)
    print(f"\nSaved to {save_path}")

    # Plot
    print(f"\n{'='*65}")
    print("  PLOTTING")
    print(f"{'='*65}")
    plot_results(all_results)

    # Summary
    print(f"\n{'='*65}")
    print("  SINGLE-SNAPSHOT DECOMPOSITION SUMMARY")
    print(f"{'='*65}")

    print(f"\n  Per-layer SVD (2D weights):")
    print(f"  {'Rank':>6s}  {'Add':>12s}  {'Mul':>12s}")
    for rank in [1, 2, 3, 5, 10, 20, 32, 64, 128]:
        adds = [r["svd_2d"][rank]["test_add"] for r in all_results if rank in r["svd_2d"]]
        muls = [r["svd_2d"][rank]["test_mul"] for r in all_results if rank in r["svd_2d"]]
        if adds:
            print(f"  {rank:>6d}  {np.mean(adds):.3f}±{np.std(adds):.3f}  "
                  f"{np.mean(muls):.3f}±{np.std(muls):.3f}")

    print(f"\n  Attention-only spectral:")
    for rank in [1, 2, 3, 5, 10, 20, 32, 64, 128]:
        adds = [r["attn_spectral"][rank]["test_add"] for r in all_results if rank in r["attn_spectral"]]
        muls = [r["attn_spectral"][rank]["test_mul"] for r in all_results if rank in r["attn_spectral"]]
        if adds:
            print(f"  {rank:>6d}  {np.mean(adds):.3f}±{np.std(adds):.3f}  "
                  f"{np.mean(muls):.3f}±{np.std(muls):.3f}")

    print(f"\n  Magnitude pruning:")
    for keep in [0.001, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.50, 0.80, 1.0]:
        adds = [r["pruning"][keep]["test_add"] for r in all_results if keep in r["pruning"]]
        muls = [r["pruning"][keep]["test_mul"] for r in all_results if keep in r["pruning"]]
        if adds:
            print(f"  {keep*100:>5.1f}%  {np.mean(adds):.3f}±{np.std(adds):.3f}  "
                  f"{np.mean(muls):.3f}±{np.std(muls):.3f}")

    print(f"\n  Uniform Δθ scaling:")
    for scale in [0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.5]:
        adds = [r["uniform_scale"][scale]["test_add"] for r in all_results]
        muls = [r["uniform_scale"][scale]["test_mul"] for r in all_results]
        print(f"  {scale:>5.2f}  {np.mean(adds):.3f}±{np.std(adds):.3f}  "
              f"{np.mean(muls):.3f}±{np.std(muls):.3f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
