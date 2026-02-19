#!/usr/bin/env python3
"""
Tri-task grokking: (x+y) mod p, (x*y) mod p, AND (x²+y²) mod p jointly.

Shared transformer trunk, three classification heads.
loss = CE(out_add, y_add) + CE(out_mul, y_mul) + CE(out_sq, y_sq)

Saves:
  - Full state_dict checkpoints every CHECKPOINT_EVERY steps
  - Attention weight snapshots every MODEL_LOG_EVERY steps
  - Per-task metrics (train/test acc for add, mul, sq separately)
  - SVD of weight deltas logged periodically
"""

import math, time, random, sys
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# ═══════════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class TriTaskConfig:
    P: int = 97
    TRAIN_FRACTION: float = 0.5
    D_MODEL: int = 128
    N_LAYERS: int = 2
    N_HEADS: int = 4
    D_FF: int = 256
    DROPOUT: float = 0.0
    LR: float = 1e-3
    BATCH_SIZE: int = 512
    STEPS: int = 300_000       # generous budget — three tasks
    EVAL_EVERY: int = 100
    MODEL_LOG_EVERY: int = 100
    CHECKPOINT_EVERY: int = 200
    GRAD_CLIP: float = 1.0
    ACC_BS: int = 2048
    STOP_ACC: float = 0.98     # all three tasks must reach this
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
# Data — joint (a,b) → ((a+b)%p, (a*b)%p, (a²+b²)%p)
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
    """Sample batch, return (a, b, y_add, y_mul, y_sq)."""
    idx = np.random.randint(0, len(pairs), size=bs)
    ab = np.array([pairs[i] for i in idx], dtype=np.int64)
    a = torch.tensor(ab[:, 0], device=device)
    b = torch.tensor(ab[:, 1], device=device)
    y_add = (a + b) % p
    y_mul = (a * b) % p
    y_sq = (a * a + b * b) % p
    return a, b, y_add, y_mul, y_sq


# ═══════════════════════════════════════════════════════════════════════════
# Model — shared trunk, three heads
# ═══════════════════════════════════════════════════════════════════════════

class TriTaskTransformer(nn.Module):
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
        self.head_sq  = nn.Linear(cfg.D_MODEL, cfg.P)

    def forward(self, a, b):
        x = torch.stack([a, b], dim=1)
        h = self.tok_emb(x) + self.pos_emb.unsqueeze(0)
        h = self.encoder(h)
        z = self.ln(h[:, 0, :])        # shared representation
        return self.head_add(z), self.head_mul(z), self.head_sq(z)


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
    """Evaluate all three tasks. Returns (add_acc, mul_acc, sq_acc)."""
    model.eval()
    correct_add, correct_mul, correct_sq, total = 0, 0, 0, 0
    for i in range(0, len(pairs), cfg.ACC_BS):
        chunk = pairs[i:i+cfg.ACC_BS]
        ab = torch.tensor(chunk, device=device)
        a, b = ab[:, 0], ab[:, 1]
        y_add = (a + b) % cfg.P
        y_mul = (a * b) % cfg.P
        y_sq  = (a * a + b * b) % cfg.P
        logits_add, logits_mul, logits_sq = model(a, b)
        correct_add += (logits_add.argmax(dim=-1) == y_add).sum().item()
        correct_mul += (logits_mul.argmax(dim=-1) == y_mul).sum().item()
        correct_sq  += (logits_sq.argmax(dim=-1) == y_sq).sum().item()
        total += a.numel()
    return correct_add / total, correct_mul / total, correct_sq / total


@torch.no_grad()
def compute_weight_svd(model, init_state):
    """SVD of weight deltas from initialization."""
    svd_log = []
    for i, layer in enumerate(model.encoder.layers):
        attn = layer.self_attn
        d = attn.embed_dim
        layer_svd = {}
        for name, (start, end) in [
            ("WQ", (0, d)), ("WK", (d, 2*d)), ("WV", (2*d, 3*d)),
        ]:
            W_now = attn.in_proj_weight[start:end].detach().cpu().float()
            W_init = init_state[f"encoder.layers.{i}.self_attn.in_proj_weight"][start:end].float()
            delta = W_now - W_init
            if delta.norm() < 1e-12:
                layer_svd[name] = [0.0] * 5
            else:
                _, S, _ = torch.linalg.svd(delta)
                layer_svd[name] = S[:5].tolist()

        W_now = attn.out_proj.weight.detach().cpu().float()
        W_init = init_state[f"encoder.layers.{i}.self_attn.out_proj.weight"].float()
        delta = W_now - W_init
        if delta.norm() < 1e-12:
            layer_svd["WO"] = [0.0] * 5
        else:
            _, S, _ = torch.linalg.svd(delta)
            layer_svd["WO"] = S[:5].tolist()

        for head_name in ["head_add", "head_mul", "head_sq"]:
            W_now_h = getattr(model, head_name).weight.detach().cpu().float()
            W_init_h = init_state[f"{head_name}.weight"].float()
            delta_h = W_now_h - W_init_h
            if delta_h.norm() < 1e-12:
                layer_svd[head_name] = [0.0] * 5
            else:
                _, S, _ = torch.linalg.svd(delta_h)
                layer_svd[head_name] = S[:5].tolist()

        svd_log.append(layer_svd)
    return svd_log


# ═══════════════════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════════════════

TASK_NAMES = ["add", "mul", "sq"]

def train(cfg: TriTaskConfig):
    device = get_device()
    print(f"Device: {device}")

    torch.manual_seed(cfg.SEED)
    np.random.seed(cfg.SEED)
    random.seed(cfg.SEED)

    train_pairs, test_pairs = build_dataset(cfg.P, cfg.TRAIN_FRACTION, cfg.SEED)
    print(f"Dataset: {len(train_pairs)} train, {len(test_pairs)} test (p={cfg.P})")

    model = TriTaskTransformer(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {n_params:,} parameters")

    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY,
        betas=(cfg.ADAM_BETA1, cfg.ADAM_BETA2)
    )
    loss_fn = nn.CrossEntropyLoss()

    init_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    attn_logs = [{"step": 0, "layers": extract_attn_matrices(model)}]
    checkpoints = [(0, {k: v.cpu().clone() for k, v in model.state_dict().items()})]
    metrics = []
    svd_logs = []

    patience = {t: 0 for t in TASK_NAMES}
    grokked = {t: False for t in TASK_NAMES}
    grok_step = {t: None for t in TASK_NAMES}
    t0 = time.time()

    for step in range(1, cfg.STEPS + 1):
        model.train()
        a, b, y_add, y_mul, y_sq = sample_batch(train_pairs, cfg.BATCH_SIZE, cfg.P, device)
        logits_add, logits_mul, logits_sq = model(a, b)
        loss_add = loss_fn(logits_add, y_add)
        loss_mul = loss_fn(logits_mul, y_mul)
        loss_sq  = loss_fn(logits_sq, y_sq)
        loss = loss_add + loss_mul + loss_sq

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
            tr_add, tr_mul, tr_sq = eval_accuracy(model, train_pairs, cfg, device)
            te_add, te_mul, te_sq = eval_accuracy(model, test_pairs, cfg, device)
            m = {
                "step": step,
                "train_add": tr_add, "train_mul": tr_mul, "train_sq": tr_sq,
                "test_add": te_add, "test_mul": te_mul, "test_sq": te_sq,
                "loss": loss.item(),
                "loss_add": loss_add.item(), "loss_mul": loss_mul.item(),
                "loss_sq": loss_sq.item(),
            }
            metrics.append(m)

            if step % 1000 == 0:
                svd = compute_weight_svd(model, init_state)
                svd_logs.append({"step": step, "svd": svd})

            if step % (cfg.EVAL_EVERY * 10) == 0 or step == 1:
                elapsed = (time.time() - t0) / 60
                print(f"  step {step:6d} | add: {tr_add:.3f}/{te_add:.3f} | "
                      f"mul: {tr_mul:.3f}/{te_mul:.3f} | sq: {tr_sq:.3f}/{te_sq:.3f} | "
                      f"loss {loss.item():.4f} | {elapsed:.1f}m")

            # Track grokking per task
            test_accs = {"add": te_add, "mul": te_mul, "sq": te_sq}
            for t in TASK_NAMES:
                if not grokked[t]:
                    if test_accs[t] >= cfg.STOP_ACC:
                        patience[t] += 1
                        if patience[t] >= cfg.STOP_PATIENCE:
                            grokked[t] = True
                            grok_step[t] = step
                            print(f"  >>> {t.upper()} GROKKED at step {step} "
                                  f"(test={test_accs[t]:.3f})")
                    else:
                        patience[t] = 0

            if all(grokked.values()):
                print(f"  >>> ALL THREE GROKKED — stopping at step {step}")
                break

    elapsed = (time.time() - t0) / 60
    print(f"\nTraining complete in {elapsed:.1f}m")
    for t in TASK_NAMES:
        print(f"  {t}: grokked={grokked[t]} at step {grok_step[t]}")

    result = {
        "cfg": asdict(cfg),
        "attn_logs": attn_logs,
        "checkpoints": checkpoints,
        "metrics": metrics,
        "svd_logs": svd_logs,
        "grokked": grokked,
        "grok_step": grok_step,
        "final_step": step,
        "init_state": init_state,
    }
    return result


def main():
    OUT_DIR.mkdir(exist_ok=True)

    seeds = [42, 137, 2024]
    for wd in [1.0, 0.5, 0.1, 0.0]:
        for seed in seeds:
            wd_tag = f"wd{wd:.0f}" if wd == int(wd) else f"wd{wd}"
            tag = f"tritask_{wd_tag}_s{seed}"
            out_path = OUT_DIR / f"{tag}.pt"

            if out_path.exists():
                print(f"[{tag}] already exists, skipping")
                continue

            print(f"\n{'='*70}")
            print(f"  {tag}")
            print(f"{'='*70}")

            cfg = TriTaskConfig(SEED=seed, WEIGHT_DECAY=wd)
            if wd == 0.0:
                cfg.STEPS = 50_000  # no-WD control — won't grok, just baseline
            result = train(cfg)

            torch.save(result, out_path)
            print(f"  saved → {out_path.name} "
                  f"({len(result['attn_logs'])} attn snaps, "
                  f"{len(result['checkpoints'])} ckpts)")

    print("\nAll seeds complete.")


if __name__ == "__main__":
    main()
