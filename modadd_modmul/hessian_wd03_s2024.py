#!/usr/bin/env python3
"""WD=0.3, seed=2024: train + Hessian."""

import sys, random, time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).parent))
from train_multitask import (
    MultiTaskConfig, MultiTaskTransformer, build_dataset, sample_batch,
    get_device, extract_attn_matrices, eval_accuracy,
)
from hessian_analysis import analyze_hessian

RESULTS_DIR = Path(__file__).parent / "results"
PLOT_DIR = Path(__file__).parent / "plots"

SEED = 2024
WD = 0.3

def train():
    tag = f"multitask_wd03_s{SEED}"
    out_path = RESULTS_DIR / f"{tag}.pt"
    if out_path.exists():
        print(f"  [{tag}] already exists, loading...")
        return torch.load(out_path, map_location="cpu", weights_only=False)

    device = get_device()
    cfg = MultiTaskConfig(
        SEED=SEED, WEIGHT_DECAY=WD, STEPS=200_000,
        CHECKPOINT_EVERY=200, MODEL_LOG_EVERY=200,
    )

    torch.manual_seed(cfg.SEED); np.random.seed(cfg.SEED); random.seed(cfg.SEED)
    train_pairs, test_pairs = build_dataset(cfg.P, cfg.TRAIN_FRACTION, cfg.SEED)
    model = MultiTaskTransformer(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model: {n_params:,} params, wd={WD}, seed={SEED}")

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
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP); opt.step()

        if step % cfg.MODEL_LOG_EVERY == 0:
            attn_logs.append({"step": step, "layers": extract_attn_matrices(model)})
        if step % cfg.CHECKPOINT_EVERY == 0:
            checkpoints.append((step, {k: v.cpu().clone() for k, v in model.state_dict().items()}))

        if step % cfg.EVAL_EVERY == 0 or step == 1:
            train_add, train_mul = eval_accuracy(model, train_pairs, cfg, device)
            test_add, test_mul = eval_accuracy(model, test_pairs, cfg, device)
            metrics.append({"step": step, "train_add": train_add, "train_mul": train_mul,
                            "test_add": test_add, "test_mul": test_mul, "loss": loss.item()})
            if step % 2000 == 0:
                elapsed = (time.time() - t0) / 60
                print(f"    [wd={WD} s{SEED}] step {step:6d} | add: {train_add:.3f}/{test_add:.3f} | "
                      f"mul: {train_mul:.3f}/{test_mul:.3f} | {elapsed:.1f}m")

            if not grokked_add:
                if test_add >= cfg.STOP_ACC: patience_add += 1
                else: patience_add = 0
                if patience_add >= cfg.STOP_PATIENCE:
                    grokked_add = True; grok_step_add = step
                    print(f"    >>> ADD GROKKED at step {step}")
            if not grokked_mul:
                if test_mul >= cfg.STOP_ACC: patience_mul += 1
                else: patience_mul = 0
                if patience_mul >= cfg.STOP_PATIENCE:
                    grokked_mul = True; grok_step_mul = step
                    print(f"    >>> MUL GROKKED at step {step}")
            if grokked_add and grokked_mul:
                print(f"    >>> BOTH GROKKED — stopping at step {step}"); break

    elapsed = (time.time() - t0) / 60
    print(f"  Training done in {elapsed:.1f}m — add={grokked_add}@{grok_step_add}, mul={grokked_mul}@{grok_step_mul}")

    result = {"cfg": asdict(cfg), "attn_logs": attn_logs, "checkpoints": checkpoints,
              "metrics": metrics, "grokked_add": grokked_add, "grokked_mul": grokked_mul,
              "grok_step_add": grok_step_add, "grok_step_mul": grok_step_mul,
              "final_step": step, "init_state": init_state}
    torch.save(result, out_path)
    print(f"  saved → {out_path.name} ({len(checkpoints)} ckpts)")
    return result


def main():
    PLOT_DIR.mkdir(exist_ok=True); RESULTS_DIR.mkdir(exist_ok=True)

    print(f"\n{'='*70}\n  Training wd={WD} (seed={SEED})\n{'='*70}")
    data = train()

    print(f"\n{'='*70}\n  Hessian analysis: wd={WD} (seed={SEED})\n{'='*70}")
    hess = analyze_hessian(data, f"wd03_s{SEED}", SEED, k=5, n_iter=25)

    save_path = PLOT_DIR / f"hessian_wd03_s{SEED}.pt"
    torch.save({"hess": hess, "meta": {
        "grokked_add": data["grokked_add"], "grokked_mul": data["grokked_mul"],
        "grok_step_add": data["grok_step_add"], "grok_step_mul": data["grok_step_mul"],
        "seed": SEED, "wd": WD,
    }}, save_path)
    print(f"\n  Saved → {save_path}")
    print("Done.")


if __name__ == "__main__":
    main()
