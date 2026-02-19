#!/usr/bin/env python3
"""
Multi-task grokking: (x+y) mod p AND (x*y) mod p jointly.

Shared transformer trunk, two classification heads.
loss = CE(out_add, target_add) + CE(out_mul, target_mul)

Saves:
  - Full state_dict checkpoints every CHECKPOINT_EVERY steps
  - Attention weight snapshots every MODEL_LOG_EVERY steps
  - Per-task metrics (train/test acc for add and mul separately)
  - SVD of weight deltas logged periodically
"""

import math, time, random, json, sys
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# ═══════════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class MultiTaskConfig:
    P: int = 97
    TRAIN_FRACTION: float = 0.5
    D_MODEL: int = 128
    N_LAYERS: int = 2
    N_HEADS: int = 4
    D_FF: int = 256
    DROPOUT: float = 0.0
    LR: float = 1e-3
    BATCH_SIZE: int = 512
    STEPS: int = 300_000       # longer budget — two tasks to learn
    EVAL_EVERY: int = 100
    MODEL_LOG_EVERY: int = 100
    CHECKPOINT_EVERY: int = 200
    GRAD_CLIP: float = 1.0
    ACC_BS: int = 2048
    STOP_ACC: float = 0.98     # both tasks must reach this
    STOP_PATIENCE: int = 3
    ADAM_BETA1: float = 0.9
    ADAM_BETA2: float = 0.98
    WEIGHT_DECAY: float = 1.0
    SEED: int = 42

OUT_DIR = Path(__file__).parent / "results"


# ═══════════════════════════════════════════════════════════════════════════
# Device
# ═══════════════════════════════════════════════════════════════════════════

def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# ═══════════════════════════════════════════════════════════════════════════
# Data — joint (a,b) → ((a+b)%p, (a*b)%p)
# ═══════════════════════════════════════════════════════════════════════════

def build_dataset(p, frac, seed):
    """
    All (a, b) pairs with a,b in 1..p-1 (nonzero for mul invertibility).
    Returns train/test splits as lists of (a, b) tuples.
    """
    pairs = [(a, b) for a in range(1, p) for b in range(1, p)]
    rng = random.Random(seed)
    rng.shuffle(pairs)
    n = int(frac * len(pairs))
    return pairs[:n], pairs[n:]


def sample_batch(pairs, bs, p, device):
    """Sample batch, return (a, b, y_add, y_mul)."""
    idx = np.random.randint(0, len(pairs), size=bs)
    ab = np.array([pairs[i] for i in idx], dtype=np.int64)
    a = torch.tensor(ab[:, 0], device=device)
    b = torch.tensor(ab[:, 1], device=device)
    y_add = (a + b) % p
    y_mul = (a * b) % p
    return a, b, y_add, y_mul


# ═══════════════════════════════════════════════════════════════════════════
# Model — shared trunk, two heads
# ═══════════════════════════════════════════════════════════════════════════

class MultiTaskTransformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg.P, cfg.D_MODEL)
        self.pos_emb = nn.Parameter(torch.randn(2, cfg.D_MODEL) / math.sqrt(cfg.D_MODEL))
        enc = nn.TransformerEncoderLayer(
            d_model=cfg.D_MODEL, nhead=cfg.N_HEADS,
            dim_feedforward=cfg.D_FF, dropout=cfg.DROPOUT,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc, num_layers=cfg.N_LAYERS)
        self.ln = nn.LayerNorm(cfg.D_MODEL)
        self.head_add = nn.Linear(cfg.D_MODEL, cfg.P)
        self.head_mul = nn.Linear(cfg.D_MODEL, cfg.P)

    def forward(self, a, b):
        x = torch.stack([a, b], dim=1)
        h = self.tok_emb(x) + self.pos_emb.unsqueeze(0)
        h = self.encoder(h)
        z = self.ln(h[:, 0, :])        # shared representation
        return self.head_add(z), self.head_mul(z)


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def extract_attn_matrices(model):
    logs = []
    for i, layer in enumerate(model.encoder.layers):
        attn = layer.self_attn
        d = attn.embed_dim
        if attn._qkv_same_embed_dim:
            Wq = attn.in_proj_weight[:d]
            Wk = attn.in_proj_weight[d:2*d]
            Wv = attn.in_proj_weight[2*d:]
        else:
            Wq = attn.q_proj_weight
            Wk = attn.k_proj_weight
            Wv = attn.v_proj_weight
        logs.append({
            "layer": i,
            "WQ": Wq.detach().cpu().clone(),
            "WK": Wk.detach().cpu().clone(),
            "WV": Wv.detach().cpu().clone(),
            "WO": attn.out_proj.weight.detach().cpu().clone(),
        })
    return logs


@torch.no_grad()
def eval_accuracy(model, pairs, cfg, device):
    """Evaluate both tasks. Returns (add_acc, mul_acc)."""
    model.eval()
    correct_add, correct_mul, total = 0, 0, 0
    for i in range(0, len(pairs), cfg.ACC_BS):
        chunk = pairs[i:i+cfg.ACC_BS]
        ab = torch.tensor(chunk, device=device)
        a, b = ab[:, 0], ab[:, 1]
        y_add = (a + b) % cfg.P
        y_mul = (a * b) % cfg.P
        logits_add, logits_mul = model(a, b)
        correct_add += (logits_add.argmax(dim=-1) == y_add).sum().item()
        correct_mul += (logits_mul.argmax(dim=-1) == y_mul).sum().item()
        total += a.numel()
    return correct_add / total, correct_mul / total


@torch.no_grad()
def compute_weight_svd(model, init_state):
    """
    SVD of weight deltas from initialization for each attention weight.
    Returns top-5 singular values per layer per weight.
    """
    svd_log = []
    for i, layer in enumerate(model.encoder.layers):
        attn = layer.self_attn
        d = attn.embed_dim
        layer_svd = {}
        for name, (start, end) in [
            ("WQ", (0, d)),
            ("WK", (d, 2*d)),
            ("WV", (2*d, 3*d)),
        ]:
            W_now = attn.in_proj_weight[start:end].detach().cpu().float()
            W_init = init_state[f"encoder.layers.{i}.self_attn.in_proj_weight"][start:end].float()
            delta = (W_now - W_init).reshape(-1)
            if delta.norm() < 1e-12:
                layer_svd[name] = [0.0] * 5
            else:
                U, S, V = torch.linalg.svd(W_now - W_init)
                layer_svd[name] = S[:5].tolist()

        W_now = attn.out_proj.weight.detach().cpu().float()
        W_init = init_state[f"encoder.layers.{i}.self_attn.out_proj.weight"].float()
        delta = W_now - W_init
        if delta.norm() < 1e-12:
            layer_svd["WO"] = [0.0] * 5
        else:
            U, S, V = torch.linalg.svd(delta)
            layer_svd["WO"] = S[:5].tolist()

        # Head deltas
        for head_name in ["head_add", "head_mul"]:
            W_now_h = getattr(model, head_name).weight.detach().cpu().float()
            W_init_h = init_state[f"{head_name}.weight"].float()
            delta_h = W_now_h - W_init_h
            if delta_h.norm() < 1e-12:
                layer_svd[head_name] = [0.0] * 5
            else:
                U, S, V = torch.linalg.svd(delta_h)
                layer_svd[head_name] = S[:5].tolist()

        svd_log.append(layer_svd)
    return svd_log


# ═══════════════════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════════════════

def train(cfg: MultiTaskConfig):
    device = get_device()
    print(f"Device: {device}")

    torch.manual_seed(cfg.SEED)
    np.random.seed(cfg.SEED)
    random.seed(cfg.SEED)

    train_pairs, test_pairs = build_dataset(cfg.P, cfg.TRAIN_FRACTION, cfg.SEED)
    print(f"Dataset: {len(train_pairs)} train, {len(test_pairs)} test "
          f"(p={cfg.P}, nonzero pairs)")

    model = MultiTaskTransformer(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {n_params:,} parameters")

    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY,
        betas=(cfg.ADAM_BETA1, cfg.ADAM_BETA2)
    )
    loss_fn = nn.CrossEntropyLoss()

    # Save init state for SVD deltas
    init_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    attn_logs = [{"step": 0, "layers": extract_attn_matrices(model)}]
    checkpoints = [(0, {k: v.cpu().clone() for k, v in model.state_dict().items()})]
    metrics = []
    svd_logs = []
    patience_add, patience_mul = 0, 0
    grokked_add, grokked_mul = False, False
    grok_step_add, grok_step_mul = None, None
    t0 = time.time()

    for step in range(1, cfg.STEPS + 1):
        model.train()
        a, b, y_add, y_mul = sample_batch(train_pairs, cfg.BATCH_SIZE, cfg.P, device)
        logits_add, logits_mul = model(a, b)
        loss_add = loss_fn(logits_add, y_add)
        loss_mul = loss_fn(logits_mul, y_mul)
        loss = loss_add + loss_mul

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
            m = {
                "step": step,
                "train_add": train_add, "train_mul": train_mul,
                "test_add": test_add, "test_mul": test_mul,
                "loss": loss.item(),
                "loss_add": loss_add.item(), "loss_mul": loss_mul.item(),
            }
            metrics.append(m)

            # SVD every 1000 steps
            if step % 1000 == 0:
                svd = compute_weight_svd(model, init_state)
                svd_logs.append({"step": step, "svd": svd})

            if step % (cfg.EVAL_EVERY * 10) == 0 or step == 1:
                elapsed = (time.time() - t0) / 60
                print(f"  step {step:6d} | add: {train_add:.3f}/{test_add:.3f} | "
                      f"mul: {train_mul:.3f}/{test_mul:.3f} | "
                      f"loss {loss.item():.4f} | {elapsed:.1f}m | "
                      f"snaps {len(attn_logs)} ckpts {len(checkpoints)}")

            # Track grokking per task
            if not grokked_add:
                if test_add >= cfg.STOP_ACC:
                    patience_add += 1
                    if patience_add >= cfg.STOP_PATIENCE:
                        grokked_add = True
                        grok_step_add = step
                        print(f"  >>> ADD GROKKED at step {step} (test={test_add:.3f})")
                else:
                    patience_add = 0

            if not grokked_mul:
                if test_mul >= cfg.STOP_ACC:
                    patience_mul += 1
                    if patience_mul >= cfg.STOP_PATIENCE:
                        grokked_mul = True
                        grok_step_mul = step
                        print(f"  >>> MUL GROKKED at step {step} (test={test_mul:.3f})")
                else:
                    patience_mul = 0

            # Stop only when BOTH grok
            if grokked_add and grokked_mul:
                print(f"  >>> BOTH GROKKED — stopping at step {step}")
                break

    elapsed = (time.time() - t0) / 60
    print(f"\nTraining complete in {elapsed:.1f}m")
    print(f"  add: grokked={grokked_add} at step {grok_step_add}")
    print(f"  mul: grokked={grokked_mul} at step {grok_step_mul}")

    result = {
        "cfg": asdict(cfg),
        "attn_logs": attn_logs,
        "checkpoints": checkpoints,
        "metrics": metrics,
        "svd_logs": svd_logs,
        "grokked_add": grokked_add,
        "grokked_mul": grokked_mul,
        "grok_step_add": grok_step_add,
        "grok_step_mul": grok_step_mul,
        "final_step": step,
        "init_state": init_state,
    }
    return result


def main():
    OUT_DIR.mkdir(exist_ok=True)

    seeds = [42, 137, 2024]
    for seed in seeds:
        tag = f"multitask_s{seed}"
        out_path = OUT_DIR / f"{tag}.pt"

        if out_path.exists():
            print(f"[{tag}] already exists, skipping")
            continue

        print(f"\n{'='*70}")
        print(f"  {tag}")
        print(f"{'='*70}")

        cfg = MultiTaskConfig(SEED=seed)
        result = train(cfg)

        torch.save(result, out_path)
        print(f"  saved → {out_path.name} "
              f"({len(result['attn_logs'])} attn snaps, "
              f"{len(result['checkpoints'])} ckpts)")

    print("\nAll seeds complete.")


if __name__ == "__main__":
    main()
