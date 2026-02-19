#!/usr/bin/env python3
"""Train tri-task models for WD=0.2 and WD=0.3."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from train_tritask import TriTaskConfig, train
import torch

OUT_DIR = Path(__file__).parent / "results"
OUT_DIR.mkdir(exist_ok=True)

seeds = [42, 137, 2024]

for wd in [0.2, 0.3]:
    for seed in seeds:
        tag = f"tritask_wd{wd}_s{seed}"
        out_path = OUT_DIR / f"{tag}.pt"

        if out_path.exists():
            print(f"[{tag}] already exists, skipping")
            continue

        print(f"\n{'='*70}")
        print(f"  {tag}")
        print(f"{'='*70}")

        cfg = TriTaskConfig(SEED=seed, WEIGHT_DECAY=wd)
        result = train(cfg)

        torch.save(result, out_path)
        print(f"  saved → {out_path.name}")

print("\nAll done.")
