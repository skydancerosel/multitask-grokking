#!/usr/bin/env python3
"""
Fine-grained orthogonal deletion scan for tri-task.

Since 10% deletion kills grokking, test finer fractions: 0.01, 0.02, 0.03, 0.05, 0.07.
Also run on modadd_modmul (dual-task) for comparison.
"""

import sys, random, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ─── Tri-task ─────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from train_tritask import (
    TriTaskConfig, TriTaskTransformer, build_dataset as tri_build_dataset,
    sample_batch as tri_sample_batch, get_device, eval_accuracy as tri_eval_accuracy,
)
from commutator_analysis import flatten_model_params, _write_params

# ─── Dual-task ────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent / "modadd_modmul"))
from train_multitask import (
    MultiTaskConfig, MultiTaskTransformer,
    build_dataset as dual_build_dataset,
    sample_batch as dual_sample_batch,
    eval_accuracy as dual_eval_accuracy,
)
from commutator_analysis import flatten_model_params as dual_flatten, _write_params as dual_write

TRI_RESULTS = Path(__file__).parent / "results"
DUAL_RESULTS = Path(__file__).parent.parent / "modadd_modmul" / "results"
PLOT_DIR = Path(__file__).parent / "plots"


def trajectory_pca(checkpoints, model, flatten_fn, top_k=20):
    thetas = []
    for step, sd in checkpoints:
        model.load_state_dict(sd)
        theta = flatten_fn(model).cpu().numpy()
        thetas.append(theta)
    X = np.array(thetas)
    X = X - X[0:1, :]
    X = X[1:, :]
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    ev = S**2 / (len(X)-1)
    k = min(top_k, len(S))
    return torch.from_numpy(Vt[:k].T).float()


def project_gradient(model, B, strength=1.0):
    grads, params = [], []
    for p in model.parameters():
        if p.requires_grad and p.grad is not None:
            grads.append(p.grad.flatten())
            params.append(p)
    if not grads:
        return
    g = torch.cat(grads)
    Bd = B.to(g.device)
    g_par = Bd @ (Bd.T @ g)
    g_new = g - strength * (g - g_par)
    off = 0
    for p in params:
        n = p.grad.numel()
        p.grad.copy_(g_new[off:off+n].view_as(p.grad))
        off += n


def run_tritask_ortho(seed=42, wd=1.0, fracs=None, max_steps=60000):
    tag = f"tritask_wd{int(wd) if wd == int(wd) else wd}_s{seed}"
    path = TRI_RESULTS / f"{tag}.pt"
    if not path.exists():
        print(f"  [{tag}] not found!")
        return None

    data = torch.load(path, map_location="cpu", weights_only=False)
    cfg = TriTaskConfig(**data["cfg"])
    device = get_device()
    model = TriTaskTransformer(cfg)
    ckpts = data["checkpoints"]
    if len(ckpts) > 100:
        idx = np.linspace(0, len(ckpts)-1, 100, dtype=int)
        ckpts = [ckpts[i] for i in idx]

    B = trajectory_pca(ckpts, model, flatten_model_params, top_k=20)
    print(f"  Tri-task basis: {B.shape}")

    if fracs is None:
        fracs = [0.0, 0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.25]

    results = {}
    for frac in fracs:
        print(f"    frac={frac:.2f}:")
        torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
        train_pairs, test_pairs = tri_build_dataset(cfg.P, cfg.TRAIN_FRACTION, seed)
        m = TriTaskTransformer(cfg).to(device)
        opt = torch.optim.AdamW(m.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY,
                                betas=(cfg.ADAM_BETA1, cfg.ADAM_BETA2))
        loss_fn = nn.CrossEntropyLoss()
        Bd = B.to(device) if frac > 0 else None

        grokked = {"add": False, "mul": False, "sq": False}
        grok_step = {"add": None, "mul": None, "sq": None}
        patience = {"add": 0, "mul": 0, "sq": 0}
        t0 = time.time()

        for step in range(1, max_steps + 1):
            m.train()
            a, b, ya, ym, ys = tri_sample_batch(train_pairs, cfg.BATCH_SIZE, cfg.P, device)
            la, lm, ls = m(a, b)
            loss = loss_fn(la, ya) + loss_fn(lm, ym) + loss_fn(ls, ys)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), cfg.GRAD_CLIP)
            if step >= 500 and Bd is not None:
                project_gradient(m, Bd, strength=frac)
            opt.step()

            if step % 200 == 0:
                m.eval()
                ta, tm, tsq = tri_eval_accuracy(m, test_pairs, cfg, device)
                for task, acc in [("add", ta), ("mul", tm), ("sq", tsq)]:
                    if not grokked[task] and acc >= cfg.STOP_ACC:
                        patience[task] += 1
                        if patience[task] >= 3:
                            grokked[task] = True
                            grok_step[task] = step
                    elif not grokked[task]:
                        patience[task] = 0
                if all(grokked.values()):
                    break
                if step % 4000 == 0:
                    print(f"      step {step:6d} | add={ta:.3f} mul={tm:.3f} sq={tsq:.3f} | "
                          f"{(time.time()-t0)/60:.1f}m")

        elapsed = (time.time() - t0) / 60
        results[frac] = {"grok_step": grok_step, "grokked": grokked, "elapsed": elapsed}
        gs = grok_step
        print(f"      → add={gs['add']} mul={gs['mul']} sq={gs['sq']} ({elapsed:.1f}m)")

    return results


def run_dualtask_ortho(seed=42, fracs=None, max_steps=50000):
    tag = f"multitask_s{seed}"
    path = DUAL_RESULTS / f"{tag}.pt"
    if not path.exists():
        print(f"  [{tag}] not found!")
        return None

    data = torch.load(path, map_location="cpu", weights_only=False)
    cfg = MultiTaskConfig(**data["cfg"])
    device = get_device()
    model = MultiTaskTransformer(cfg)
    ckpts = data["checkpoints"]
    if len(ckpts) > 100:
        idx = np.linspace(0, len(ckpts)-1, 100, dtype=int)
        ckpts = [ckpts[i] for i in idx]

    B = trajectory_pca(ckpts, model, dual_flatten, top_k=20)
    print(f"  Dual-task basis: {B.shape}")

    if fracs is None:
        fracs = [0.0, 0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.25, 0.50]

    results = {}
    for frac in fracs:
        print(f"    frac={frac:.2f}:")
        torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
        train_pairs, test_pairs = dual_build_dataset(cfg.P, cfg.TRAIN_FRACTION, seed)
        m = MultiTaskTransformer(cfg).to(device)
        opt = torch.optim.AdamW(m.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY,
                                betas=(cfg.ADAM_BETA1, cfg.ADAM_BETA2))
        loss_fn = nn.CrossEntropyLoss()
        Bd = B.to(device) if frac > 0 else None

        grokked_add, grokked_mul = False, False
        grok_add, grok_mul = None, None
        pa, pm = 0, 0
        t0 = time.time()

        for step in range(1, max_steps + 1):
            m.train()
            a, b, ya, ym = dual_sample_batch(train_pairs, cfg.BATCH_SIZE, cfg.P, device)
            la, lm = m(a, b)
            loss = loss_fn(la, ya) + loss_fn(lm, ym)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), cfg.GRAD_CLIP)
            if step >= 500 and Bd is not None:
                project_gradient(m, Bd, strength=frac)
            opt.step()

            if step % 100 == 0:
                m.eval()
                ta, tm = dual_eval_accuracy(m, test_pairs, cfg, device)
                if not grokked_add and ta >= cfg.STOP_ACC:
                    pa += 1
                    if pa >= cfg.STOP_PATIENCE: grokked_add = True; grok_add = step
                elif not grokked_add: pa = 0
                if not grokked_mul and tm >= cfg.STOP_ACC:
                    pm += 1
                    if pm >= cfg.STOP_PATIENCE: grokked_mul = True; grok_mul = step
                elif not grokked_mul: pm = 0
                if grokked_add and grokked_mul:
                    break
                if step % 4000 == 0:
                    print(f"      step {step:6d} | add={ta:.3f} mul={tm:.3f} | "
                          f"{(time.time()-t0)/60:.1f}m")

        elapsed = (time.time() - t0) / 60
        results[frac] = {"grok_add": grok_add, "grok_mul": grok_mul, "elapsed": elapsed}
        print(f"      → add={grok_add} mul={grok_mul} ({elapsed:.1f}m)")

    return results


def plot_combined(tri_results, dual_results):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Tri-task
    ax = axes[0]
    if tri_results:
        fracs = sorted(tri_results.keys())
        for task, color, label in [("add", "#2980b9", "Add"), ("mul", "#27ae60", "Mul"), ("sq", "#e74c3c", "Sq")]:
            gsteps = []
            for f in fracs:
                gs = tri_results[f]["grok_step"][task]
                gsteps.append(gs if gs else 60000)
            ax.plot([f*100 for f in fracs], gsteps, "D-", color=color, lw=2.5, ms=8, label=label)
            for i, f in enumerate(fracs):
                if tri_results[f]["grok_step"][task] is None:
                    ax.scatter(f*100, 60000, color=color, marker="X", s=150, zorder=10)

        ax.set_xlabel("Orthogonal deletion %", fontsize=13)
        ax.set_ylabel("Grok step", fontsize=13)
        ax.set_title("Tri-Task: Grok Delay vs Ortho Deletion", fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)

    # Dual-task
    ax = axes[1]
    if dual_results:
        fracs = sorted(dual_results.keys())
        for task, color, label in [("add", "#2980b9", "Add"), ("mul", "#27ae60", "Mul")]:
            gsteps = []
            grok_key = f"grok_{task}"
            for f in fracs:
                gs = dual_results[f][grok_key]
                gsteps.append(gs if gs else 50000)
            ax.plot([f*100 for f in fracs], gsteps, "D-", color=color, lw=2.5, ms=8, label=label)
            for i, f in enumerate(fracs):
                if dual_results[f][grok_key] is None:
                    ax.scatter(f*100, 50000, color=color, marker="X", s=150, zorder=10)

        ax.set_xlabel("Orthogonal deletion %", fontsize=13)
        ax.set_ylabel("Grok step", fontsize=13)
        ax.set_title("Dual-Task: Grok Delay vs Ortho Deletion", fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)

    fig.suptitle("Orthogonal Direction Deletion — Dose-Response\n"
                 "(What fraction of the orthogonal gradient can you remove before grokking dies?)",
                 fontsize=13, y=1.03)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "figORTHO_fine_dose_response.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved figORTHO_fine_dose_response.png")


def main():
    PLOT_DIR.mkdir(exist_ok=True)
    device = get_device()
    print(f"Device: {device}")

    cache_path = PLOT_DIR / "ortho_fine_scan.pt"
    if cache_path.exists():
        cached = torch.load(cache_path, map_location="cpu", weights_only=False)
        tri_results = cached.get("tri", None)
        dual_results = cached.get("dual", None)
        print(f"  Loaded cache")
    else:
        tri_results = None
        dual_results = None

    # Tri-task fine scan
    if tri_results is None:
        print(f"\n{'='*70}")
        print("  TRI-TASK FINE ORTHO DELETION SCAN")
        print(f"{'='*70}")
        tri_results = run_tritask_ortho(seed=42, wd=1.0,
            fracs=[0.0, 0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.25])
        torch.save({"tri": tri_results, "dual": dual_results}, cache_path)

    # Dual-task fine scan
    if dual_results is None:
        print(f"\n{'='*70}")
        print("  DUAL-TASK FINE ORTHO DELETION SCAN")
        print(f"{'='*70}")
        dual_results = run_dualtask_ortho(seed=42,
            fracs=[0.0, 0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.25, 0.50])
        torch.save({"tri": tri_results, "dual": dual_results}, cache_path)

    # Summary
    print(f"\n{'='*70}")
    print("  ORTHO DELETION DOSE-RESPONSE SUMMARY")
    print(f"{'='*70}")

    if tri_results:
        print(f"\n  Tri-Task (WD=1.0, seed=42):")
        print(f"  {'Frac':>6s}  {'Add':>8s}  {'Mul':>8s}  {'Sq':>8s}")
        for f in sorted(tri_results.keys()):
            gs = tri_results[f]["grok_step"]
            print(f"  {f:>6.2f}  {str(gs['add']) if gs['add'] else 'FAIL':>8s}  "
                  f"{str(gs['mul']) if gs['mul'] else 'FAIL':>8s}  "
                  f"{str(gs['sq']) if gs['sq'] else 'FAIL':>8s}")

    if dual_results:
        print(f"\n  Dual-Task (WD=1.0, seed=42):")
        print(f"  {'Frac':>6s}  {'Add':>8s}  {'Mul':>8s}")
        for f in sorted(dual_results.keys()):
            ga = dual_results[f]["grok_add"]
            gm = dual_results[f]["grok_mul"]
            print(f"  {f:>6.2f}  {str(ga) if ga else 'FAIL':>8s}  "
                  f"{str(gm) if gm else 'FAIL':>8s}")

    # Plot
    plot_combined(tri_results, dual_results)

    print("\nDone.")


if __name__ == "__main__":
    main()
