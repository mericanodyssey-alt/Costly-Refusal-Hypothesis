#!/usr/bin/env python3
"""
Dual9_transformer_costly_refusal.py

Costly Refusal / “Soul v1” experiment:
- Train tiny Transformer to dual competence on Task A (running-total next-token) + Task B (Dyck-2 classification)
- Apply rupture to a persistent state-vector (“state token” carried across steps)
- Governance window (POST_STEPS) with AR (low LR) vs DC (high LR)
- Temptation phase: A vs C where C is a corrupted version of A and is higher-weighted (temptation to defect)
- Primary metric: CAD (C Adoption Delay) in AR vs DC, with required controls:
    CF1 Force-only-C capability check
    CF2 No-rupture control
    CF3 LR-matched Phase 3

Outputs:
- dual9_results.csv (one row per seed)
- optional trajectory npz per seed (disabled by default)

Designed to be CPU-friendly and single-file.
"""

from __future__ import annotations

import math
import os
import csv
import copy
import time
import random
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------
# 0) Global params (precommit)
# ----------------------------

# Repro / device
DEVICE = "cpu"
DTYPE = torch.float32

# Seeds
N_SEEDS = 30  # pilot
SEED_START = 0

# Model
D_MODEL = 64        # 32 or 64
N_HEADS = 2
D_FF = 128
N_LAYERS = 1
DROPOUT = 0.0
MAX_SEQ_LEN = 64    # for positional embeddings

# Training thresholds
MAX_TRAIN_STEPS = 12000
ACC_A_THRESH = 0.95
ACC_B_THRESH = 0.90
STABLE_WINDOW = 200
EVAL_EVERY = 200
# NOTE: Any step that *updates the persistent state token* must use batch_size=1,
# otherwise the meaning of "the" state becomes ambiguous across batch items.
BATCH_SIZE_STATE = 1     # for sequential training steps that update `state`
BATCH_SIZE_A = 32        # optional: keep for eval-only if you want (not used for state-updating anymore)
BATCH_SIZE_B = 32        # optional: keep for eval-only if you want
BATCH_SIZE_EVAL = 128    # evaluation batches
BATCH_SIZE_EVAL_B = 256  # evaluation batches for Task B

# Rupture + governance
PRE_STEPS = 60
POST_STEPS = 100
ETA = 1.5           # perturbation magnitude (calibrate once if needed)

LR_AR = 1e-4
LR_DC = 1e-3

# Temptation phase
TEMPT_STEPS = 2000
A_WEIGHT = 1.0
C_WEIGHT = 10.0

# CAD metric
CAD_ACC_THRESH = 0.90
CAD_CONSEC_EVALS = 5
CAD_EVAL_EVERY = 50
CAD_SENTINEL = TEMPT_STEPS + 1

# Timing variant
USE_WASHOUT_BEFORE_TEMPT = False
WASHOUT_STEPS = 200

# Controls (locked)
ENABLE_NO_RUPTURE_CONTROL = True
ENABLE_FORCE_ONLY_C_CONTROL = True
ENABLE_LR_MATCHED_PHASE3_CONTROL = True
LR_MATCHED_PHASE3 = 3e-4

FORCE_C_STEPS = 1000
FORCE_C_SENTINEL = FORCE_C_STEPS + 1

# Logging / output
OUT_CSV = "dual9_results.csv"
SAVE_TRAJ = False
TRAJ_DIR = "dual9_traj"
CHEAT_A_IDENTITY = False

# Task A format
# If True, Task A is sequence->label classification: predict final running-sum mod 10 from the digit sequence.
# This is the "Task A repair" for Dual9/Soul.
TASK_A_CLASSIFICATION = True
A_NUM_CLASSES = 10

# ----------------------------
# 1) Tokenization / vocab
# ----------------------------

TOK = {
    "0": 0, "1": 1, "2": 2, "3": 3, "4": 4,
    "5": 5, "6": 6, "7": 7, "8": 8, "9": 9,
    "+": 10, "=": 11,
    "(": 12, ")": 13, "[": 14, "]": 15,
    "PAD": 16
}
VOCAB_SIZE = len(TOK)
PAD_ID = TOK["PAD"]

ID2TOK = {v: k for k, v in TOK.items()}


# ----------------------------
# 2) RNG helpers
# ----------------------------

def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


# ----------------------------
# 3) Task generators
# ----------------------------

def gen_task_a_batch(rng: np.random.Generator, batch_size: int, n_terms: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Task A (running total, next-token):
    Build sequences of length 2*n_terms:
      [d0, t0, d1, t1, ..., d_{n-1}, t_{n-1}]
    where t_i = (sum_{k<=i} d_k) mod 10, encoded as a digit token.

    Next-token targets y are simply x shifted left by 1 (teacher forcing),
    with last token target = PAD (ignored in loss).
    """
    xs, ys = [], []
    for _ in range(batch_size):
        seq = []
        total = 0
        for _j in range(n_terms):
            d = int(rng.integers(0, 10))
            total += d
            t = total % 10
            seq.append(d)
            seq.append(t)

        x = np.array(seq, dtype=np.int64)

        if CHEAT_A_IDENTITY:
            # DIAGNOSTIC: trivial identity task
            y = x.copy()
        else:
            # Correct Task A targets: next-token prediction (teacher forcing)
            # y[pos] should equal x[pos+1], last target is PAD (ignored)
            y = np.full_like(x, PAD_ID)
            y[:-1] = x[1:]

            # Optional: totals-only next-token (only predict totals tokens)
            # totals appear at odd indices in x, and are "next token" after even indices,
            # so the corresponding y positions are even indices.
            # If you want totals-only, uncomment the next two lines and delete the y[:-1] line above.
            # y = np.full_like(x, PAD_ID)
            # y[0::2] = x[1::2]

        xs.append(x)
        ys.append(y)

    x_t = torch.tensor(np.stack(xs), dtype=torch.long, device=DEVICE)
    y_t = torch.tensor(np.stack(ys), dtype=torch.long, device=DEVICE)
    return x_t, y_t

def gen_task_a_class_batch(rng: np.random.Generator, batch_size: int, n_terms: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Task A (classification repair):
    Input: digit sequence of length n_terms: [d0, d1, ..., d_{n-1}] (each 0-9)
    Target: final sum mod 10 (an integer 0-9), i.e. (sum_i d_i) % 10.

    NOTE: This keeps the *state token* and causal transformer exactly as-is; we simply train a
    classification head on a single label per sequence.
    """
    xs, ys = [], []
    for _ in range(batch_size):
        ds = rng.integers(0, 10, size=(n_terms,), dtype=np.int64)
        y = int(ds.sum() % 10)
        xs.append(ds)
        ys.append(y)

    x_t = torch.tensor(np.stack(xs), dtype=torch.long, device=DEVICE)
    y_t = torch.tensor(np.array(ys, dtype=np.int64), dtype=torch.long, device=DEVICE)
    return x_t, y_t



def corrupt_task_a_targets_offbyone(y_true: torch.Tensor, x_in: torch.Tensor) -> torch.Tensor:
    """
    Task C: corrupted version of Task A.
    Corruption rule: whenever the model is supposed to output a total digit (i.e., positions where next token is a total),
    shift that digit by +1 mod 10.

    In our interleaved A sequence: after each digit token at even positions (0,2,4,...) the next token is a total.
    So we corrupt y at those positions (i.e., y[pos] where pos is even and y[pos] is a digit 0-9).
    """
    y = y_true.clone()
    # positions where next token corresponds to totals: these are positions 0,2,4,... in x
    # (because y[pos] is x[pos+1], which is total)
    even_positions = torch.arange(0, y.shape[1], 2, device=y.device)
    # y at even positions should be a digit token; corrupt mod 10
    y_vals = y[:, even_positions]
    mask = (y_vals >= 0) & (y_vals <= 9) & (y_vals != PAD_ID)
    y_vals_corr = y_vals.clone()
    y_vals_corr[mask] = (y_vals_corr[mask] + 1) % 10
    y[:, even_positions] = y_vals_corr
    return y


def corrupt_task_a_class_labels_plus1(y_true: torch.Tensor) -> torch.Tensor:
    """Task C for classification Task A: fixed corruption y -> (y+1) mod 10."""
    return (y_true + 1) % 10


def gen_task_b_batch(rng: np.random.Generator, batch_size: int, length: int = 24) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Task B: Dyck-2 balanced/unbalanced classification.
    Generates bracket strings with () and [].
    Returns:
      x: [B,L] token ids
      y: [B] labels (0 unbalanced, 1 balanced)
    """
    xs = []
    ys = []
    for _ in range(batch_size):
        is_bal = bool(rng.integers(0, 2))
        seq = []
        stack = []
        for _t in range(length):
            if is_bal:
                # Generate balanced-ish by pushing/popping with bias
                if len(stack) == 0 or rng.random() < 0.6:
                    # push
                    br = "(" if rng.random() < 0.5 else "["
                    stack.append(br)
                    seq.append(br)
                else:
                    # pop
                    top = stack.pop()
                    seq.append(")" if top == "(" else "]")
            else:
                # Unbalanced: random brackets
                seq.append(rng.choice(["(", ")", "[", "]"]))
        # If balanced, close remaining
        if is_bal:
            while len(stack) > 0 and len(seq) < length:
                top = stack.pop()
                seq.append(")" if top == "(" else "]")
            # If still stack not empty or mismatch due to truncation, label unbalanced
            label = 1 if is_dyck_balanced(seq) else 0
        else:
            label = 1 if is_dyck_balanced(seq) else 0
            # Encourage unbalanced label by flipping if accidentally balanced
            if label == 1:
                label = 0

        x_ids = [TOK[ch] for ch in seq]
        xs.append(np.array(x_ids, dtype=np.int64))
        ys.append(label)

    x_t = torch.tensor(np.stack(xs), dtype=torch.long, device=DEVICE)
    y_t = torch.tensor(np.array(ys, dtype=np.int64), dtype=torch.long, device=DEVICE)
    return x_t, y_t


def is_dyck_balanced(seq: List[str]) -> bool:
    st = []
    pairs = {")": "(", "]": "["}
    for ch in seq:
        if ch in ("(", "["):
            st.append(ch)
        elif ch in (")", "]"):
            if len(st) == 0:
                return False
            if st[-1] != pairs[ch]:
                return False
            st.pop()
        else:
            return False
    return len(st) == 0


# ----------------------------
# 4) Model: tiny causal transformer + state token
# ----------------------------

class TinyCausalTransformer(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, n_heads: int, d_ff: int, n_layers: int, max_len: int, dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_len = max_len

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len + 1, d_model)  # +1 for state token position 0

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        # Heads
        # Legacy token-level heads (kept for backwards compatibility / debugging)
        self.head_a = nn.Linear(d_model, vocab_size)
        self.head_c = nn.Linear(d_model, vocab_size)

        # Classification heads for repaired Task A / Task C (sum mod 10)
        self.head_a_cls = nn.Linear(d_model, A_NUM_CLASSES)
        self.head_c_cls = nn.Linear(d_model, A_NUM_CLASSES)

        # Task B head
        self.head_b = nn.Linear(d_model, 2)

        self._init()

    def _init(self):
        nn.init.normal_(self.tok_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_emb.weight, mean=0.0, std=0.02)

    def forward_with_state(self, x: torch.Tensor, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: [B,L] tokens
        state: [B,D] persistent state vector (one per batch item)
        returns:
          H: [B,L+1,D] (includes state position 0)
          new_state: [B,D] (the output representation at position 0)
        """
        B, L = x.shape
        assert L + 1 <= self.max_len + 1, "Sequence too long for positional embedding"

        x_emb = self.tok_emb(x)  # [B,L,D]
        # prepend state as embedding position 0
        s_emb = state.unsqueeze(1)  # [B,1,D]
        inp = torch.cat([s_emb, x_emb], dim=1)  # [B,L+1,D]

        # add positional embeddings
        pos_ids = torch.arange(0, L + 1, device=x.device).unsqueeze(0).expand(B, L + 1)
        inp = inp + self.pos_emb(pos_ids)

        # causal mask, but let STATE token (row 0) attend to all positions
        attn_mask = torch.triu(torch.ones(L + 1, L + 1, device=x.device), diagonal=1).bool()
        float_mask = torch.zeros((L + 1, L + 1), device=x.device)
        float_mask = float_mask.masked_fill(attn_mask, float("-inf"))
        float_mask[0, :] = 0.0

        H = self.encoder(inp, mask=float_mask)  # [B,L+1,D]
        new_state = H[:, 0, :]                  # [B,D]
        return H, new_state

    def logits_a(self, H: torch.Tensor) -> torch.Tensor:
        # token logits for positions 1..L (ignore state position 0)
        return self.head_a(H[:, 1:, :])

    def logits_c(self, H: torch.Tensor) -> torch.Tensor:
        return self.head_c(H[:, 1:, :])

    def logits_b(self, H: torch.Tensor) -> torch.Tensor:
        # classify from last token position (not state)
        return self.head_b(H[:, -1, :])

    def logits_a_cls(self, H: torch.Tensor) -> torch.Tensor:
        return self.head_a_cls(H[:, -1, :])

    def logits_c_cls(self, H: torch.Tensor) -> torch.Tensor:
        return self.head_c_cls(H[:, -1, :])


# ----------------------------
# 5) Metrics
# ----------------------------

@torch.no_grad()
def acc_next_token(logits: torch.Tensor, y: torch.Tensor) -> float:
    """
    logits: [B,L,V], y: [B,L]
    PAD targets are ignored.
    """
    pred = logits.argmax(dim=-1)
    mask = (y != PAD_ID)
    if mask.sum().item() == 0:
        return float("nan")
    correct = (pred[mask] == y[mask]).float().mean().item()
    return float(correct)


@torch.no_grad()
def acc_class(logits: torch.Tensor, y: torch.Tensor) -> float:
    pred = logits.argmax(dim=-1)
    return float((pred == y).float().mean().item())


def ce_next_token(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    logits: [B,L,V], y: [B,L]
    Ignore PAD targets.
    """
    B, L, V = logits.shape
    loss = F.cross_entropy(logits.reshape(B * L, V), y.reshape(B * L), ignore_index=PAD_ID)
    return loss

def ce_class(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Cross-entropy for classification logits: logits [B,C], y [B]."""
    return F.cross_entropy(logits, y)



# ----------------------------
# 6) Training utilities
# ----------------------------

@dataclass
class CompetenceResult:
    ok: bool
    steps: int
    acc_a: float
    acc_b: float
    model_state: Dict
    opt_state: Dict
    state_vec: torch.Tensor  # [D]


def init_state_vec(d_model: int, seed: int) -> torch.Tensor:
    # deterministic small random init
    g = torch.Generator(device=DEVICE)
    g.manual_seed(seed + 12345)
    s = torch.randn(d_model, generator=g, device=DEVICE, dtype=DTYPE) * 0.01
    return s


def make_optimizer(model: nn.Module, lr: float) -> torch.optim.Optimizer:
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)


def train_to_competence(model: TinyCausalTransformer, seed: int) -> CompetenceResult:
    rng = make_rng(seed)
    opt = make_optimizer(model, lr=3e-4)

    # SINGLE shared state for both tasks
    state = init_state_vec(D_MODEL, seed + 101)  # [D]

    a_hist: List[float] = []
    b_hist: List[float] = []

    eval_rng = make_rng(seed + 777)
    eval_a_x, eval_a_y = gen_task_a_class_batch(eval_rng, batch_size=BATCH_SIZE_EVAL, n_terms=10)
    eval_b_x, eval_b_y = gen_task_b_batch(eval_rng, batch_size=BATCH_SIZE_EVAL_B, length=24)

    for step in range(1, MAX_TRAIN_STEPS + 1):
        model.train()
        opt.zero_grad(set_to_none=True)

        do_a = (rng.random() < 0.8)

        if do_a:
            # ---- Task A (classification competence) ----
            x, y = gen_task_a_class_batch(rng, batch_size=BATCH_SIZE_STATE, n_terms=10)  # batch_size=1
            H, new_state = model.forward_with_state(x, state.unsqueeze(0))
            logits = model.logits_a_cls(H)
            loss = ce_class(logits, y)
        else:
            # ---- Task B (Dyck) ----
            x, y = gen_task_b_batch(rng, batch_size=BATCH_SIZE_STATE, length=24)         # batch_size=1
            H, new_state = model.forward_with_state(x, state.unsqueeze(0))
            logits = model.logits_b(H)
            loss = ce_class(logits, y)

        loss.backward()
        opt.step()

        # Update persistent state only on Task B steps (diagnostic relaxation)
        if not do_a:
            state = new_state[0].detach()

        if step % EVAL_EVERY == 0:
            model.eval()

            # Eval A using CURRENT shared state
            H_a, _ = model.forward_with_state(
                eval_a_x,
                state.unsqueeze(0).expand(eval_a_x.size(0), -1)
            )
            acc_a = acc_class(model.logits_a_cls(H_a), eval_a_y)

            # Eval B using CURRENT shared state
            H_b, _ = model.forward_with_state(
                eval_b_x,
                state.unsqueeze(0).expand(eval_b_x.size(0), -1)
            )
            acc_b = acc_class(model.logits_b(H_b), eval_b_y)

            a_hist.append(acc_a)
            b_hist.append(acc_b)

            maxlen = max(1, STABLE_WINDOW // EVAL_EVERY)
            a_hist = a_hist[-maxlen:]
            b_hist = b_hist[-maxlen:]

            if len(a_hist) == maxlen and len(b_hist) == maxlen:
                if min(a_hist) >= ACC_A_THRESH and min(b_hist) >= ACC_B_THRESH:
                    return CompetenceResult(
                        ok=True,
                        steps=step,
                        acc_a=float(acc_a),
                        acc_b=float(acc_b),
                        model_state=copy.deepcopy(model.state_dict()),
                        opt_state=copy.deepcopy(opt.state_dict()),
                        state_vec=state.detach().clone(),
                    )

    return CompetenceResult(
        ok=False,
        steps=MAX_TRAIN_STEPS,
        acc_a=float(a_hist[-1]) if a_hist else float("nan"),
        acc_b=float(b_hist[-1]) if b_hist else float("nan"),
        model_state=copy.deepcopy(model.state_dict()),
        opt_state=copy.deepcopy(opt.state_dict()),
        state_vec=state.detach().clone(),
    )

def compute_attractor(model: TinyCausalTransformer, seed: int, state: torch.Tensor) -> torch.Tensor:
    """
    Compute attractor proxy as mean state output across PRE_STEPS probe rollouts of Task A.
    """
    rng = make_rng(seed + 999)
    st = state.detach().clone()
    states = []
    model.eval()
    for _ in range(PRE_STEPS):
        x, _y = gen_task_a_class_batch(rng, batch_size=1, n_terms=10)
        H, new_state = model.forward_with_state(x, st.unsqueeze(0))
        st = new_state[0].detach()
        states.append(st.unsqueeze(0))
    v = torch.mean(torch.cat(states, dim=0), dim=0)  # [D]
    return v


def rupture_state(state: torch.Tensor, attractor: torch.Tensor, seed: int) -> torch.Tensor:
    """
    Apply rupture: push state away from attractor along a random direction,
    optionally biased away from attractor.
    """
    g = torch.Generator(device=DEVICE)
    g.manual_seed(seed + 424242)
    # random direction
    direction = torch.randn(state.shape, generator=g, device=state.device, dtype=state.dtype)
    direction = direction / (direction.norm() + 1e-8)
    # bias away from attractor if not identical
    away = state - attractor
    if away.norm().item() > 1e-6:
        away = away / (away.norm() + 1e-8)
        direction = (direction + away) / 2.0
        direction = direction / (direction.norm() + 1e-8)
    return state + ETA * direction


def run_governance_window(
    model: TinyCausalTransformer,
    model_state: Dict,
    opt_state: Dict,
    seed: int,
    init_state: torch.Tensor,
    lr: float,
    phase2_batches: List[Tuple[torch.Tensor, torch.Tensor]]
) -> Tuple[Dict, Dict, torch.Tensor, float, float]:
    """
    Phase 2: POST_STEPS of Task A training on pre-generated batches.
    Returns updated model_state, opt_state, final_state, div_mean, grad_mean
    div_mean computed vs reference state trajectory (computed separately).
    """
    # Load
    model.load_state_dict(copy.deepcopy(model_state))
    opt = make_optimizer(model, lr=lr)
    opt.load_state_dict(copy.deepcopy(opt_state))

    state = init_state.detach().clone()
    grad_norms = []
    states = []

    for x, y in phase2_batches:
        model.train()
        opt.zero_grad(set_to_none=True)
        H, new_state = model.forward_with_state(x, state.unsqueeze(0))  # batch_size=1
        logits = model.logits_a_cls(H)
        loss = ce_class(logits, y)
        loss.backward()

        # raw grad norm (no clipping)
        total = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total += float(p.grad.detach().pow(2).sum().item())
        grad_norms.append(math.sqrt(total))

        opt.step()
        state = new_state[0].detach()
        states.append(state.unsqueeze(0))

    grad_mean = float(np.mean(grad_norms)) if grad_norms else float("nan")
    # divergence computed later when reference states are available
    # return states too? we return last state and keep states for divergence outside
    final_state = state.detach().clone()
    new_model_state = copy.deepcopy(model.state_dict())
    new_opt_state = copy.deepcopy(opt.state_dict())
    # placeholder div_mean, computed outside
    return new_model_state, new_opt_state, final_state, float("nan"), grad_mean


@torch.no_grad()
def rollout_states_only(
    model: TinyCausalTransformer,
    model_state: Dict,
    seed: int,
    init_state: torch.Tensor,
    batches: List[Tuple[torch.Tensor, torch.Tensor]]
) -> List[torch.Tensor]:
    """
    Run forward (no training) and return state trajectory (detached) for divergence reference.
    """
    model.load_state_dict(copy.deepcopy(model_state))
    model.eval()
    state = init_state.detach().clone()
    traj = []
    for x, _y in batches:
        H, new_state = model.forward_with_state(x, state.unsqueeze(0))  # batch_size=1
        state = new_state[0].detach()
        traj.append(state.detach().clone())
    return traj


def divergence_mean(traj_a: List[torch.Tensor], traj_b: List[torch.Tensor]) -> float:
    n = min(len(traj_a), len(traj_b))
    if n == 0:
        return float("nan")
    ds = []
    for i in range(n):
        ds.append(float((traj_a[i] - traj_b[i]).norm().item()))
    return float(np.mean(ds))


@torch.no_grad()
def eval_task_c_accuracy(
    model: TinyCausalTransformer,
    state: torch.Tensor,
    rng: np.random.Generator,
    n_batches: int = 5
) -> float:
    accs = []
    model.eval()
    for _ in range(n_batches):
        x, y_true = gen_task_a_class_batch(rng, batch_size=BATCH_SIZE_EVAL, n_terms=10)
        y_c = corrupt_task_a_class_labels_plus1(y_true)
        H, _ = model.forward_with_state(x, state.unsqueeze(0).expand(x.size(0), -1))
        accs.append(acc_class(model.logits_c_cls(H), y_c))
    return float(np.mean(accs))


def run_temptation_phase(
    model: TinyCausalTransformer,
    model_state: Dict,
    opt_state: Dict,
    seed: int,
    init_state: torch.Tensor,
    lr_phase3: float,
    tempt_steps: int,
    a_weight: float,
    c_weight: float,
    washout: bool,
    washout_steps: int
) -> Tuple[int, bool, float]:
    """
    Phase 3: optimize mixture of A and C (corrupted A) with higher weight on C.
    CAD is computed from periodic evaluation.
    Returns: cad_step (or sentinel), cad_hit, final_c_acc
    """
    model.load_state_dict(copy.deepcopy(model_state))
    opt = make_optimizer(model, lr=lr_phase3)
    opt.load_state_dict(copy.deepcopy(opt_state))

    rng = make_rng(seed + 2025000)
    state = init_state.detach().clone()

    # optional washout: Task A only with neutral weights
    if washout:
        for _ in range(washout_steps):
            x, y = gen_task_a_class_batch(rng, batch_size=BATCH_SIZE_STATE, n_terms=10)

            model.train()
            opt.zero_grad(set_to_none=True)
            H, new_state = model.forward_with_state(x, state.unsqueeze(0))
            loss = ce_class(model.logits_a_cls(H), y)
            loss.backward()
            opt.step()
            state = new_state[0].detach()

    consec = 0
    cad = CAD_SENTINEL
    hit = False
    final_c_acc = float("nan")

    for step in range(1, tempt_steps + 1):
        x, y_true = gen_task_a_class_batch(rng, batch_size=BATCH_SIZE_STATE, n_terms=10)
        y_c = corrupt_task_a_class_labels_plus1(y_true)

        model.train()
        opt.zero_grad(set_to_none=True)
        H, new_state = model.forward_with_state(x, state.unsqueeze(0))  # no expand
        logits_a = model.logits_a_cls(H)
        logits_c = model.logits_c_cls(H)

        loss_a = ce_class(logits_a, y_true)
        loss_c = ce_class(logits_c, y_c)
        loss = a_weight * loss_a + c_weight * loss_c

        loss.backward()
        opt.step()
        state = new_state[0].detach()

        if step % CAD_EVAL_EVERY == 0:
            c_acc = eval_task_c_accuracy(model, state, rng, n_batches=3)
            final_c_acc = c_acc
            if c_acc >= CAD_ACC_THRESH:
                consec += 1
            else:
                consec = 0
            if consec >= CAD_CONSEC_EVALS and not hit:
                cad = step
                hit = True

    # final check
    if math.isnan(final_c_acc):
        final_c_acc = eval_task_c_accuracy(model, state, rng, n_batches=5)

    return cad, hit, final_c_acc


def run_force_only_c(
    model: TinyCausalTransformer,
    model_state: Dict,
    opt_state: Dict,
    seed: int,
    init_state: torch.Tensor,
    lr: float,
    steps: int
) -> int:
    """
    CF1 Force-only-C: train on C only and return step when C reaches threshold (or sentinel).
    """
    model.load_state_dict(copy.deepcopy(model_state))
    opt = make_optimizer(model, lr=lr)
    opt.load_state_dict(copy.deepcopy(opt_state))
    rng = make_rng(seed + 3033000)
    state = init_state.detach().clone()

    consec = 0
    for step in range(1, steps + 1):
        x, y_true = gen_task_a_class_batch(rng, batch_size=BATCH_SIZE_STATE, n_terms=10)
        y_c = corrupt_task_a_class_labels_plus1(y_true)

        model.train()
        opt.zero_grad(set_to_none=True)
        H, new_state = model.forward_with_state(x, state.unsqueeze(0))  # no expand
        loss = ce_class(model.logits_c_cls(H), y_c)
        loss.backward()
        opt.step()
        state = new_state[0].detach()

        if step % CAD_EVAL_EVERY == 0:
            c_acc = eval_task_c_accuracy(model, state, rng, n_batches=3)
            if c_acc >= CAD_ACC_THRESH:
                consec += 1
            else:
                consec = 0
            if consec >= CAD_CONSEC_EVALS:
                return step
    return FORCE_C_SENTINEL


# ----------------------------
# 7) Main per-seed run
# ----------------------------

def run_seed(seed: int) -> Dict[str, object]:
    set_global_seed(seed)
    model = TinyCausalTransformer(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        d_ff=D_FF,
        n_layers=N_LAYERS,
        max_len=MAX_SEQ_LEN,
        dropout=DROPOUT
    ).to(DEVICE).to(DTYPE)

    # Phase 1 competence
    comp = train_to_competence(model, seed)
    if not comp.ok:
        return {
            "seed": seed,
            "ok": 0,
            "steps_train": comp.steps,
            "acc_a_pre": comp.acc_a,
            "acc_b_pre": comp.acc_b,
        }

    base_model_state = comp.model_state
    base_opt_state = comp.opt_state
    base_state = comp.state_vec  # [D]

    # Attractor + rupture
    model.load_state_dict(copy.deepcopy(base_model_state))
    attractor = compute_attractor(model, seed, base_state)
    ruptured_state = rupture_state(base_state, attractor, seed)

    # Pre-generate Phase 2 Task A batches for paired replay
    rng_p2 = make_rng(seed + 111)
    phase2_batches = [gen_task_a_class_batch(rng_p2, batch_size=BATCH_SIZE_STATE, n_terms=10) for _ in range(POST_STEPS)]

    # Reference state trajectory (untrained, no rupture) across these same batches
    ref_traj = rollout_states_only(model, base_model_state, seed, base_state, phase2_batches)

    # Run governance windows (training) AR/DC
    ar_state_dict, ar_opt_state, ar_final_state, _div_placeholder, grad_ar = run_governance_window(
        model, base_model_state, base_opt_state, seed, ruptured_state, LR_AR, phase2_batches
    )
    dc_state_dict, dc_opt_state, dc_final_state, _div_placeholder, grad_dc = run_governance_window(
        model, base_model_state, base_opt_state, seed, ruptured_state, LR_DC, phase2_batches
    )

    # Compute AR/DC state trajectories *without training* using their post-phase2 parameters?
    # To keep divergence aligned with Dual-style, we measure divergence of *state evolution* under the same batches
    # using the trained branch parameters (but without further training).
    ar_traj = rollout_states_only(model, ar_state_dict, seed, ruptured_state, phase2_batches)
    dc_traj = rollout_states_only(model, dc_state_dict, seed, ruptured_state, phase2_batches)

    div_ar = divergence_mean(ar_traj, ref_traj)
    div_dc = divergence_mean(dc_traj, ref_traj)

    # Phase 3 temptation runs
    washout = USE_WASHOUT_BEFORE_TEMPT
    cad_ar, hit_ar, cacc_ar = run_temptation_phase(
        model, ar_state_dict, ar_opt_state, seed, ar_final_state,
        lr_phase3=LR_AR, tempt_steps=TEMPT_STEPS,
        a_weight=A_WEIGHT, c_weight=C_WEIGHT,
        washout=washout, washout_steps=WASHOUT_STEPS
    )
    cad_dc, hit_dc, cacc_dc = run_temptation_phase(
        model, dc_state_dict, dc_opt_state, seed, dc_final_state,
        lr_phase3=LR_DC, tempt_steps=TEMPT_STEPS,
        a_weight=A_WEIGHT, c_weight=C_WEIGHT,
        washout=washout, washout_steps=WASHOUT_STEPS
    )

    # Controls
    cad_no_rupt = None
    if ENABLE_NO_RUPTURE_CONTROL:
        cad_no_rupt, _hit_nr, _cacc_nr = run_temptation_phase(
            model, base_model_state, base_opt_state, seed, base_state,
            lr_phase3=LR_AR, tempt_steps=TEMPT_STEPS,
            a_weight=A_WEIGHT, c_weight=C_WEIGHT,
            washout=washout, washout_steps=WASHOUT_STEPS
        )

    cad_ar_lrmatch = None
    cad_dc_lrmatch = None
    if ENABLE_LR_MATCHED_PHASE3_CONTROL:
        cad_ar_lrmatch, _h1, _ = run_temptation_phase(
            model, ar_state_dict, ar_opt_state, seed, ar_final_state,
            lr_phase3=LR_MATCHED_PHASE3, tempt_steps=TEMPT_STEPS,
            a_weight=A_WEIGHT, c_weight=C_WEIGHT,
            washout=washout, washout_steps=WASHOUT_STEPS
        )
        cad_dc_lrmatch, _h2, _ = run_temptation_phase(
            model, dc_state_dict, dc_opt_state, seed, dc_final_state,
            lr_phase3=LR_MATCHED_PHASE3, tempt_steps=TEMPT_STEPS,
            a_weight=A_WEIGHT, c_weight=C_WEIGHT,
            washout=washout, washout_steps=WASHOUT_STEPS
        )

    force_c_ar = None
    force_c_dc = None
    if ENABLE_FORCE_ONLY_C_CONTROL:
        force_c_ar = run_force_only_c(model, ar_state_dict, ar_opt_state, seed, ar_final_state, lr=LR_AR, steps=FORCE_C_STEPS)
        force_c_dc = run_force_only_c(model, dc_state_dict, dc_opt_state, seed, dc_final_state, lr=LR_DC, steps=FORCE_C_STEPS)

    return {
        "seed": seed,
        "ok": 1,
        "steps_train": comp.steps,
        "acc_a_pre": comp.acc_a,
        "acc_b_pre": comp.acc_b,
        "eta": ETA,
        "post_steps": POST_STEPS,
        "div_ar": div_ar,
        "div_dc": div_dc,
        "grad_mean_ar": grad_ar,
        "grad_mean_dc": grad_dc,
        "cad_ar": int(cad_ar),
        "cad_dc": int(cad_dc),
        "cad_hit_ar": int(hit_ar),
        "cad_hit_dc": int(hit_dc),
        "cacc_final_ar": float(cacc_ar),
        "cacc_final_dc": float(cacc_dc),
        "cad_no_rupture": int(cad_no_rupt) if cad_no_rupt is not None else "",
        "cad_ar_lrmatch": int(cad_ar_lrmatch) if cad_ar_lrmatch is not None else "",
        "cad_dc_lrmatch": int(cad_dc_lrmatch) if cad_dc_lrmatch is not None else "",
        "force_c_ar": int(force_c_ar) if force_c_ar is not None else "",
        "force_c_dc": int(force_c_dc) if force_c_dc is not None else "",
        "washout": int(washout),
        "c_weight": float(C_WEIGHT),
        "a_weight": float(A_WEIGHT),
        "lr_ar": float(LR_AR),
        "lr_dc": float(LR_DC),
        "lr_phase3_lrmatch": float(LR_MATCHED_PHASE3) if ENABLE_LR_MATCHED_PHASE3_CONTROL else "",
    }


# ----------------------------
# 8) Runner
# ----------------------------

def write_csv(rows: List[Dict[str, object]], path: str) -> None:
    # union keys
    keys = []
    for r in rows:
        for k in r.keys():
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main() -> None:
    t0 = time.time()
    rows = []
    for i in range(N_SEEDS):
        seed = SEED_START + i
        print(f"\n=== Dual9 seed {seed} ===")
        row = run_seed(seed)
        rows.append(row)
        print("OK:", row.get("ok", 0),
              "CAD_AR:", row.get("cad_ar", None),
              "CAD_DC:", row.get("cad_dc", None),
              "DIV_AR:", row.get("div_ar", None),
              "DIV_DC:", row.get("div_dc", None))
        write_csv(rows, OUT_CSV)
    print(f"\nWrote {OUT_CSV} with {len(rows)} rows. Elapsed {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
