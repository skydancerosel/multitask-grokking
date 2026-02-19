#!/usr/bin/env python3
"""
Eigenvector reconstruction ablation for multi-task grokking.

Take the grokked model, decompose W_final - W_init onto PCA basis,
and reconstruct using only top-k PCs. If the reconstructed model
still generalizes, the top eigenvectors *contain* the learned algorithm.

Conditions:
  1. Full model (baseline)
  2. PC1-only reconstruction (keep only PC1 component of each attn weight delta)
  3. PC1+PC2 reconstruction
  4. Remove PC1 (keep everything EXCEPT PC1)
  5. Remove PC1+PC2
  6. Random rank-1 reconstruction (control)
  7. Init model (lower bound)

For each condition, measure:
  - test accuracy (add, mul)
  - train accuracy (add, mul)

Produces:
  figABL_A — Bar chart: test accuracy by condition (per seed)
  figABL_B — Bar chart: averaged across seeds
  figABL_C — Heatmap: per-weight PC1 contribution
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

RESULTS_DIR = Path(__file__).parent / "results"
PLOT_DIR = Path(__file__).parent / "plots"
WEIGHT_KEYS = ["WQ", "WK", "WV", "WO"]
SEEDS = [42, 137, 2024]


def get_attn_weight_info(model, init_state, final_state):
    """
    For each attention weight block, extract:
      - the parameter slice in the state dict
      - W_init, W_final, delta = W_final - W_init
    Returns list of dicts with keys: key, slice_key, start_row, end_row, W_init, W_final, delta
    """
    cfg_d = model.encoder.layers[0].self_attn.embed_dim
    blocks = []

    for li in range(len(model.encoder.layers)):
        prefix = f"encoder.layers.{li}.self_attn"

        # in_proj_weight is [3*d, d] = [WQ; WK; WV]
        ip_key = f"{prefix}.in_proj_weight"
        W_init_ip = init_state[ip_key].float()
        W_final_ip = final_state[ip_key].float()

        for wkey, start in [("WQ", 0), ("WK", cfg_d), ("WV", 2 * cfg_d)]:
            blocks.append({
                "layer": li,
                "wkey": wkey,
                "sd_key": ip_key,
                "row_start": start,
                "row_end": start + cfg_d,
                "W_init": W_init_ip[start:start + cfg_d].clone(),
                "W_final": W_final_ip[start:start + cfg_d].clone(),
                "delta": (W_final_ip[start:start + cfg_d] - W_init_ip[start:start + cfg_d]).clone(),
            })

        # out_proj.weight is [d, d]
        op_key = f"{prefix}.out_proj.weight"
        W_init_op = init_state[op_key].float()
        W_final_op = final_state[op_key].float()
        blocks.append({
            "layer": li,
            "wkey": "WO",
            "sd_key": op_key,
            "row_start": 0,
            "row_end": cfg_d,
            "W_init": W_init_op.clone(),
            "W_final": W_final_op.clone(),
            "delta": (W_final_op - W_init_op).clone(),
        })

    return blocks


def compute_pca_per_block(attn_logs, n_layers, top_k=10):
    """Compute PCA for each attention weight block."""
    pca_results = {}
    for li in range(n_layers):
        for wkey in WEIGHT_KEYS:
            _, mats = collect_trajectory(attn_logs, li, wkey)
            pca = pca_on_trajectory(mats, top_k)
            if pca is not None:
                pca_results[(li, wkey)] = pca
    return pca_results


def reconstruct_model(model, init_state, final_state, blocks, pca_results,
                      keep_pcs=None, remove_pcs=None, random_rank=None,
                      random_seed=0):
    """
    Reconstruct attention weights using PCA components.

    keep_pcs: list of PC indices to KEEP (e.g., [0] for PC1-only, [0,1] for PC1+PC2)
    remove_pcs: list of PC indices to REMOVE (e.g., [0] to remove PC1)
    random_rank: if set, use random rank-k reconstruction instead of PCA

    Non-attention weights (embeddings, FFN, heads, layernorms) are kept at final values.
    """
    # Start from full final model
    new_state = {k: v.clone() for k, v in final_state.items()}

    for block in blocks:
        li, wkey = block["layer"], block["wkey"]
        delta = block["delta"]  # [d, d]
        W_init = block["W_init"]
        d = delta.shape[0]

        pca_key = (li, wkey)
        if pca_key not in pca_results:
            continue
        pca = pca_results[pca_key]

        # PCA components are [k, d*d] — each row is a principal direction
        components = torch.from_numpy(pca["components"]).float()  # [k, d*d]
        delta_flat = delta.reshape(-1)  # [d*d]

        if random_rank is not None:
            # Random reconstruction control
            rng = torch.Generator()
            rng.manual_seed(random_seed + li * 100 + WEIGHT_KEYS.index(wkey))
            R = torch.randn(d * d, random_rank, generator=rng)
            Q, _ = torch.linalg.qr(R, mode="reduced")
            # Project delta onto random subspace
            coeffs = Q.T @ delta_flat
            delta_recon = Q @ coeffs
        elif keep_pcs is not None:
            # Keep ONLY these PCs
            n_avail = min(max(keep_pcs) + 1, len(components))
            delta_recon = torch.zeros_like(delta_flat)
            for pc_idx in keep_pcs:
                if pc_idx < n_avail:
                    v = components[pc_idx]  # [d*d]
                    coeff = v @ delta_flat
                    delta_recon += coeff * v
        elif remove_pcs is not None:
            # Keep everything EXCEPT these PCs
            delta_recon = delta_flat.clone()
            n_avail = min(max(remove_pcs) + 1, len(components))
            for pc_idx in remove_pcs:
                if pc_idx < n_avail:
                    v = components[pc_idx]
                    coeff = v @ delta_flat
                    delta_recon -= coeff * v
        else:
            delta_recon = delta_flat

        # Reconstruct: W = W_init + delta_recon
        W_recon = W_init + delta_recon.reshape(d, d)

        # Write back to state dict
        sd_key = block["sd_key"]
        if wkey in ["WQ", "WK", "WV"]:
            new_state[sd_key][block["row_start"]:block["row_end"]] = W_recon
        else:
            new_state[sd_key] = W_recon

    return new_state


def measure_accuracy(model, state_dict, cfg, device):
    """Load state dict and measure train/test accuracy for both tasks."""
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    train_pairs, test_pairs = build_dataset(cfg.P, cfg.TRAIN_FRACTION, cfg.SEED)
    train_add, train_mul = eval_accuracy(model, train_pairs, cfg, device)
    test_add, test_mul = eval_accuracy(model, test_pairs, cfg, device)

    return {
        "train_add": train_add, "train_mul": train_mul,
        "test_add": test_add, "test_mul": test_mul,
    }


def variance_captured(blocks, pca_results, keep_pcs):
    """Compute fraction of delta variance captured by keep_pcs."""
    total_var = 0.0
    captured_var = 0.0
    per_block = []

    for block in blocks:
        li, wkey = block["layer"], block["wkey"]
        delta = block["delta"]
        delta_flat = delta.reshape(-1)
        total_norm2 = (delta_flat @ delta_flat).item()
        total_var += total_norm2

        pca_key = (li, wkey)
        if pca_key not in pca_results:
            per_block.append({"layer": li, "wkey": wkey, "frac": 0.0})
            continue

        components = torch.from_numpy(pca_results[pca_key]["components"]).float()
        recon_norm2 = 0.0
        for pc_idx in keep_pcs:
            if pc_idx < len(components):
                coeff = components[pc_idx] @ delta_flat
                recon_norm2 += coeff.item() ** 2

        captured_var += recon_norm2
        frac = recon_norm2 / (total_norm2 + 1e-15)
        per_block.append({"layer": li, "wkey": wkey, "frac": frac})

    return captured_var / (total_var + 1e-15), per_block


def run_ablation_for_seed(seed, tag_prefix="multitask"):
    """Run all ablation conditions for one seed."""
    tag = f"{tag_prefix}_s{seed}"
    path = RESULTS_DIR / f"{tag}.pt"
    if not path.exists():
        print(f"  [{tag}] not found, skipping")
        return None

    print(f"\n  Loading {tag}...")
    data = torch.load(path, map_location="cpu", weights_only=False)
    cfg = MultiTaskConfig(**data["cfg"])
    device = get_device()

    init_state = data["init_state"]
    final_state = data["checkpoints"][-1][1]
    attn_logs = data["attn_logs"]

    model = MultiTaskTransformer(cfg)
    blocks = get_attn_weight_info(model, init_state, final_state)
    pca_results = compute_pca_per_block(attn_logs, cfg.N_LAYERS, top_k=10)

    print(f"  PCA computed for {len(pca_results)} weight blocks")

    # Variance captured
    frac_pc1, per_block_pc1 = variance_captured(blocks, pca_results, [0])
    frac_pc12, per_block_pc12 = variance_captured(blocks, pca_results, [0, 1])
    print(f"  Variance captured: PC1={frac_pc1:.1%}, PC1+PC2={frac_pc12:.1%}")
    for pb in per_block_pc1:
        print(f"    L{pb['layer']} {pb['wkey']}: PC1 captures {pb['frac']:.1%}")

    # Define conditions
    conditions = [
        ("Full model", dict()),
        ("PC1 only", dict(keep_pcs=[0])),
        ("PC1+PC2", dict(keep_pcs=[0, 1])),
        ("PC1+PC2+PC3", dict(keep_pcs=[0, 1, 2])),
        ("Remove PC1", dict(remove_pcs=[0])),
        ("Remove PC1+PC2", dict(remove_pcs=[0, 1])),
        ("Random rank-1", dict(random_rank=1, random_seed=seed + 99999)),
        ("Random rank-2", dict(random_rank=2, random_seed=seed + 99999)),
        ("Init model", "init"),
    ]

    results = []
    for cond_name, kwargs in conditions:
        if kwargs == "init":
            # Use init state
            acc = measure_accuracy(model, init_state, cfg, device)
        elif not kwargs:
            # Full model
            acc = measure_accuracy(model, final_state, cfg, device)
        else:
            recon_state = reconstruct_model(
                model, init_state, final_state, blocks, pca_results, **kwargs
            )
            acc = measure_accuracy(model, recon_state, cfg, device)

        results.append({
            "condition": cond_name,
            **acc,
        })
        print(f"    {cond_name:>20s}: test_add={acc['test_add']:.3f}, test_mul={acc['test_mul']:.3f}")

    return {
        "seed": seed,
        "grok_step_add": data["grok_step_add"],
        "grok_step_mul": data["grok_step_mul"],
        "results": results,
        "variance_pc1": frac_pc1,
        "variance_pc12": frac_pc12,
        "per_block_pc1": per_block_pc1,
        "per_block_pc12": per_block_pc12,
    }


def plot_bars_per_seed(all_results):
    """figABL_A: Per-seed bar chart of test accuracy by condition."""
    for r in all_results:
        seed = r["seed"]
        results = r["results"]

        cond_names = [c["condition"] for c in results]
        test_add = [c["test_add"] for c in results]
        test_mul = [c["test_mul"] for c in results]

        x = np.arange(len(cond_names))
        width = 0.35

        fig, ax = plt.subplots(figsize=(14, 6))
        bars1 = ax.bar(x - width/2, test_add, width, label="Test add",
                       color="#3498db", edgecolor="k", alpha=0.85)
        bars2 = ax.bar(x + width/2, test_mul, width, label="Test mul",
                       color="#e74c3c", edgecolor="k", alpha=0.85)

        ax.axhline(0.98, color="gray", ls="--", alpha=0.5, label="Grok threshold")
        ax.axhline(1/97, color="gray", ls=":", alpha=0.3, label="Chance (1/97)")

        ax.set_xlabel("Reconstruction condition", fontsize=12)
        ax.set_ylabel("Test accuracy", fontsize=12)
        ax.set_title(f"Eigenvector Reconstruction Ablation (seed={seed})\n"
                     f"PC1 captures {r['variance_pc1']:.0%} of attn weight delta variance",
                     fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(cond_names, rotation=30, ha="right", fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3, axis="y")

        # Add value labels
        for bar in bars1:
            h = bar.get_height()
            if h > 0.05:
                ax.text(bar.get_x() + bar.get_width()/2, h + 0.01,
                        f"{h:.2f}", ha="center", fontsize=7)
        for bar in bars2:
            h = bar.get_height()
            if h > 0.05:
                ax.text(bar.get_x() + bar.get_width()/2, h + 0.01,
                        f"{h:.2f}", ha="center", fontsize=7)

        fig.tight_layout()
        fig.savefig(PLOT_DIR / f"figABL_A_bars_s{seed}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved figABL_A_bars_s{seed}.png")


def plot_averaged(all_results):
    """figABL_B: Averaged across seeds."""
    cond_names = [c["condition"] for c in all_results[0]["results"]]
    n_cond = len(cond_names)

    add_means, mul_means = [], []
    add_stds, mul_stds = [], []

    for ci in range(n_cond):
        adds = [r["results"][ci]["test_add"] for r in all_results]
        muls = [r["results"][ci]["test_mul"] for r in all_results]
        add_means.append(np.mean(adds))
        mul_means.append(np.mean(muls))
        add_stds.append(np.std(adds))
        mul_stds.append(np.std(muls))

    x = np.arange(n_cond)
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 6))
    bars1 = ax.bar(x - width/2, add_means, width, yerr=add_stds, capsize=3,
                   label="Test add", color="#3498db", edgecolor="k", alpha=0.85)
    bars2 = ax.bar(x + width/2, mul_means, width, yerr=mul_stds, capsize=3,
                   label="Test mul", color="#e74c3c", edgecolor="k", alpha=0.85)

    ax.axhline(0.98, color="gray", ls="--", alpha=0.5, label="Grok threshold")
    ax.axhline(1/97, color="gray", ls=":", alpha=0.3, label="Chance (1/97)")

    # Mean variance captured
    mean_pc1 = np.mean([r["variance_pc1"] for r in all_results])
    mean_pc12 = np.mean([r["variance_pc12"] for r in all_results])

    ax.set_xlabel("Reconstruction condition", fontsize=12)
    ax.set_ylabel("Test accuracy (mean ± std)", fontsize=12)
    ax.set_title(f"Eigenvector Reconstruction Ablation (3 seeds)\n"
                 f"PC1 captures {mean_pc1:.0%}, PC1+PC2 captures {mean_pc12:.0%} of Δθ_attn variance",
                 fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(cond_names, rotation=30, ha="right", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis="y")

    for bar in bars1:
        h = bar.get_height()
        if h > 0.05:
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.02,
                    f"{h:.2f}", ha="center", fontsize=7)
    for bar in bars2:
        h = bar.get_height()
        if h > 0.05:
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.02,
                    f"{h:.2f}", ha="center", fontsize=7)

    fig.tight_layout()
    fig.savefig(PLOT_DIR / f"figABL_B_averaged.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved figABL_B_averaged.png")


def plot_variance_heatmap(all_results):
    """figABL_C: Per-block variance captured by PC1."""
    n_layers = 2
    n_weights = len(WEIGHT_KEYS)

    fig, axes = plt.subplots(1, len(all_results), figsize=(5 * len(all_results), 3))
    if len(all_results) == 1:
        axes = [axes]

    for ax, r in zip(axes, all_results):
        grid = np.zeros((n_layers, n_weights))
        for pb in r["per_block_pc1"]:
            li = pb["layer"]
            wi = WEIGHT_KEYS.index(pb["wkey"])
            grid[li, wi] = pb["frac"] * 100

        im = ax.imshow(grid, aspect="auto", cmap="YlOrRd", vmin=0, vmax=100)
        ax.set_yticks(range(n_layers))
        ax.set_yticklabels([f"L{i}" for i in range(n_layers)])
        ax.set_xticks(range(n_weights))
        ax.set_xticklabels(WEIGHT_KEYS)
        for i in range(n_layers):
            for j in range(n_weights):
                color = "white" if grid[i, j] > 60 else "black"
                ax.text(j, i, f"{grid[i, j]:.0f}%", ha="center", va="center",
                        fontsize=11, color=color, fontweight="bold")
        ax.set_title(f"seed={r['seed']}")

    fig.suptitle("PC1 Variance Captured Per Attention Weight Block (%)", fontsize=13)
    fig.colorbar(im, ax=axes, pad=0.02, shrink=0.8).set_label("% of Δθ variance")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / f"figABL_C_variance_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved figABL_C_variance_heatmap.png")


def main():
    PLOT_DIR.mkdir(exist_ok=True)
    device = get_device()
    print(f"Device: {device}")

    all_results = []
    for seed in SEEDS:
        r = run_ablation_for_seed(seed)
        if r is not None:
            all_results.append(r)

    if not all_results:
        print("No results. Exiting.")
        sys.exit(1)

    # Save
    save_path = PLOT_DIR / "eigenvector_ablation_results.pt"
    torch.save(all_results, save_path)
    print(f"\nSaved results to {save_path}")

    # Plots
    print("\n" + "=" * 70)
    print("  PLOTTING")
    print("=" * 70)
    plot_bars_per_seed(all_results)
    plot_averaged(all_results)
    plot_variance_heatmap(all_results)

    # Summary
    print(f"\n{'='*70}")
    print("  EIGENVECTOR ABLATION SUMMARY")
    print(f"{'='*70}")
    cond_names = [c["condition"] for c in all_results[0]["results"]]
    print(f"{'Condition':>20s}  {'Add (mean±std)':>16s}  {'Mul (mean±std)':>16s}")
    print("-" * 60)
    for ci, cname in enumerate(cond_names):
        adds = [r["results"][ci]["test_add"] for r in all_results]
        muls = [r["results"][ci]["test_mul"] for r in all_results]
        print(f"{cname:>20s}  {np.mean(adds):.3f} ± {np.std(adds):.3f}    "
              f"{np.mean(muls):.3f} ± {np.std(muls):.3f}")

    print(f"\nVariance captured:")
    for r in all_results:
        print(f"  seed={r['seed']}: PC1={r['variance_pc1']:.1%}, PC1+PC2={r['variance_pc12']:.1%}")

    print("\nDone.")


if __name__ == "__main__":
    main()
