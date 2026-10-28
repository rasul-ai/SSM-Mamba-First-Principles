#!/usr/bin/env python3
"""
mamba_lm.py - train and run your own Mamba language model, end to end, in plain PyTorch.

The code follows the deck "State space models and the Mamba architecture":

    PART 1  data             text -> characters -> token ids
    PART 2  selective scan   delta, A, B, C, D and the recurrence h_t = A_bar h_{t-1} + B_bar x_t  (Figures 3-6)
    PART 3  Mamba block      expand -> conv -> SiLU -> selective SSM -> gate -> project down       (Figures 5, 7)
    PART 4  Mamba-2 block    one projection makes z, x, B, C, delta; chunked "duality" scan        (Figure 8)
    PART 5  language model   embedding -> N residual blocks -> RMSNorm -> next-token logits
    PART 6  training         AdamW, warmup + cosine schedule, validation, checkpoints
    PART 7  generation       reads one token at a time and carries only a fixed-size state

Usage (needs only PyTorch >= 2.0):
    python mamba_lm.py test                          # quick check that the maths is right (~10 s)
    python mamba_lm.py train                         # downloads Tiny Shakespeare, trains, saves mamba.pt
    python mamba_lm.py train --data corpus.txt       # any UTF-8 text; character-level, so Bangla works too
    python mamba_lm.py train --arch mamba2           # use the Mamba-2 block instead
    python mamba_lm.py generate --prompt "ROMEO:"    # sample from the trained model
    python mamba_lm.py train --help                  # model size and training options

Plain PyTorch keeps every step readable and runs on CPU, CUDA or Apple MPS. It stores the state at every
position during training, so memory grows with batch x length x width x state size. For models in the hundreds
of millions of parameters, the official fused kernels (pip install mamba-ssm) compute the same scan without that.
"""

from __future__ import annotations

import argparse
import contextlib
import math
import os
import sys
import time
import unicodedata
import urllib.request
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


# =====================================================================================================
# PART 1 - Data: text -> token ids
# =====================================================================================================

TINY_SHAKESPEARE = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


class CharTokenizer:
    """One token per Unicode character: no dependencies, and it works for any script, Bangla included."""

    def __init__(self, chars):
        self.chars = list(chars)
        self.index = {c: i for i, c in enumerate(self.chars)}

    @classmethod
    def from_text(cls, text):
        return cls(sorted(set(text)))

    @property
    def vocab_size(self):
        return len(self.chars)

    def encode(self, text):
        text = unicodedata.normalize("NFC", text)  # the same character always becomes the same code points
        missing = sorted(set(text).difference(self.index))
        if missing:
            print(f"[skipping characters the model never saw: {''.join(missing)!r}]", file=sys.stderr)
        return [self.index[c] for c in text if c in self.index]

    def decode(self, ids):
        return "".join(self.chars[i] for i in ids)


def load_text(path):
    if path is None:
        path = "tinyshakespeare.txt"
        if not os.path.exists(path):
            print(f"no --data given, downloading Tiny Shakespeare to {path}")
            urllib.request.urlretrieve(TINY_SHAKESPEARE, path)
    with open(path, encoding="utf-8") as f:
        return unicodedata.normalize("NFC", f.read())


def get_batch(data, batch_size, seq_len, device):
    """Random windows of seq_len tokens. The targets are the same windows shifted one token to the left."""
    starts = torch.randint(len(data) - seq_len - 1, (batch_size,))
    x = torch.stack([data[s : s + seq_len] for s in starts])
    y = torch.stack([data[s + 1 : s + seq_len + 1] for s in starts])
    return x.to(device), y.to(device)


# =====================================================================================================
# PART 2 - The selective scan
# =====================================================================================================


def _no_autocast(t):
    """The recurrence runs in float32 even under mixed precision, as the official kernels do."""
    return torch.autocast(t.device.type, enabled=False) if t.device.type == "cuda" else contextlib.nullcontext()


def sequential_scan(a, b):
    """h_t = a_t * h_{t-1} + b_t with h_{-1} = 0, one step at a time.

    This is the deck's slide "The discretized SSM is a recurrence", written literally.
    a, b: (batch, length, ...). Returns every h_t, same shape. Simple, but strictly sequential.
    """
    h, hs = torch.zeros_like(b[:, 0]), []
    for a_t, b_t in zip(a.unbind(1), b.unbind(1)):  # unbind, not a[:, t]: its backward is one stack, not L copies
        h = a_t * h + b_t
        hs.append(h)
    return torch.stack(hs, dim=1)


def parallel_scan(a, b):
    """The same h_t, computed in log2(length) levels instead of length steps (deck Figure 6).

    Two neighbouring steps fuse into one step, because each step is only multiply-and-add:
        h_{2i+1} = (a_{2i+1} a_{2i}) h_{2i-1} + (a_{2i+1} b_{2i} + b_{2i+1})
    Fuse every pair, solve that half-length problem the same way, then fill in the first state of each
    pair from the state just before it. Eight tokens need three levels; a thousand need ten.
    """
    L = b.size(1)
    if L == 1:
        return b
    if L % 2:  # odd length: append an identity step (a=1, b=0) so that every token has a partner
        a = torch.cat([a, torch.ones_like(a[:, :1])], dim=1)
        b = torch.cat([b, torch.zeros_like(b[:, :1])], dim=1)
    (a_even, a_odd), (b_even, b_odd) = a.unflatten(1, (-1, 2)).unbind(2), b.unflatten(1, (-1, 2)).unbind(2)
    h_odd = parallel_scan(a_odd * a_even, a_odd * b_even + b_odd)  # the state after each pair
    h_before = torch.cat([torch.zeros_like(h_odd[:, :1]), h_odd[:, :-1]], dim=1)
    h_even = a_even * h_before + b_even  # the state after the first token of each pair
    return torch.stack([h_even, h_odd], dim=2).flatten(1, 2)[:, :L]


def selective_scan(x, delta, A, B, C, D, mode="parallel"):
    """The selective SSM (S6) over a whole sequence.

    x      (batch, L, d)   input to the SSM, one value per channel   x_t
    delta  (batch, L, d)   step size, per token and per channel      "how much to update"
    A      (d, n)          negative, the same for every input        carries the past forward
    B, C   (batch, L, n)   chosen by each token                      how to write in, how to read out
    D      (d,)            skip connection from input straight to output
    Every channel has its own n-number state, so the state h_t is (batch, d, n).
    """
    with _no_autocast(x):
        x, delta, B, C = x.float(), delta.float(), B.float(), C.float()
        # Discretize (Figure 3). Zero-order hold gives A_bar = exp(delta * A). For B the official
        # implementation uses the simpler first-order form B_bar = delta * B, and so do we.
        A_bar = torch.exp(delta.unsqueeze(-1) * A)  # (batch, L, d, n)
        Bx_bar = (delta * x).unsqueeze(-1) * B.unsqueeze(2)  # (batch, L, d, n)
        h = (parallel_scan if mode == "parallel" else sequential_scan)(A_bar, Bx_bar)
        y = torch.einsum("bldn,bln->bld", h, C)  # y_t = C_t h_t
        return y + x * D  #     + D x_t


# =====================================================================================================
# PART 3 - The Mamba block (deck Figures 5 and 7)
# =====================================================================================================


def _init_dt_bias(bias, dt_min=1e-3, dt_max=0.1):
    """Start delta log-uniformly in [dt_min, dt_max], the official init. delta = softplus(bias), so invert softplus."""
    dt = torch.exp(torch.rand(bias.shape) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)).clamp(min=1e-4)
    with torch.no_grad():
        bias.copy_(dt + torch.log(-torch.expm1(-dt)))
    bias._no_reinit = True


class MambaBlock(nn.Module):
    """Steps 01-07 of the slide "Inside the block, step by step". Step 08, the residual add, is in ResidualBlock."""

    def __init__(self, cfg):
        super().__init__()
        d_inner = cfg.expand * cfg.d_model
        self.d_inner, self.d_state, self.d_conv, self.dt_rank = d_inner, cfg.d_state, cfg.d_conv, cfg.dt_rank
        self.scan = cfg.scan
        # 01-02 Expand and split: one projection makes the SSM path x and the gate path z, each expand x d_model wide
        self.in_proj = nn.Linear(cfg.d_model, 2 * d_inner, bias=False)
        # 03 Convolve: a short causal convolution over the last d_conv tokens, one filter per channel
        self.conv1d = nn.Conv1d(d_inner, d_inner, cfg.d_conv, groups=d_inner, padding=cfg.d_conv - 1)
        # Figure 5: every token produces its own delta, B and C. One linear layer split three ways; delta goes
        # through a small rank-dt_rank bottleneck and is widened back to one value per channel.
        self.x_proj = nn.Linear(d_inner, cfg.dt_rank + 2 * cfg.d_state, bias=False)
        self.dt_proj = nn.Linear(cfg.dt_rank, d_inner)
        nn.init.uniform_(self.dt_proj.weight, -(cfg.dt_rank**-0.5), cfg.dt_rank**-0.5)
        _init_dt_bias(self.dt_proj.bias)
        # A and D are learned but do not depend on the input. A is stored as log(-A) so it stays negative, a
        # memory that decays. A = -1, -2, ..., -N per channel is the simple diagonal ("S4D-real") init that
        # Mamba uses in place of the full HiPPO matrix.
        self.A_log = nn.Parameter(torch.log(torch.arange(1, cfg.d_state + 1, dtype=torch.float32)).repeat(d_inner, 1))
        self.D = nn.Parameter(torch.ones(d_inner))
        self.A_log._no_weight_decay = self.D._no_weight_decay = True
        # 07 Project down to the model width
        self.out_proj = nn.Linear(d_inner, cfg.d_model, bias=False)

    def forward(self, u):  # u: (batch, L, d_model)
        L = u.size(1)
        x, z = self.in_proj(u).chunk(2, dim=-1)  # 01-02
        x = self.conv1d(x.transpose(1, 2))[..., :L].transpose(1, 2)  # 03 (the first L outputs are the causal ones)
        x = F.silu(x)  # 04
        delta, B, C = self.x_proj(x).split([self.dt_rank, self.d_state, self.d_state], dim=-1)
        delta = F.softplus(self.dt_proj(delta))  # delta_t > 0
        A = -torch.exp(self.A_log.float())
        mode = self.scan if self.scan != "auto" else ("sequential" if u.device.type == "cpu" else "parallel")
        y = selective_scan(x, delta, A, B, C, self.D, mode)  # 05 Select and scan
        y = y * F.silu(z)  # 06 Gate: the z path's SiLU is the volume knob
        return self.out_proj(y)  # 07

    def init_cache(self, batch, device, dtype=torch.float32):
        conv_window = torch.zeros(batch, self.d_inner, self.d_conv, device=device, dtype=dtype)
        h = torch.zeros(batch, self.d_inner, self.d_state, device=device, dtype=dtype)
        return conv_window, h

    def step(self, u, cache):  # u: (batch, d_model), a single token
        conv_window, h = cache
        x, z = self.in_proj(u).chunk(2, dim=-1)
        conv_window = torch.cat([conv_window[..., 1:], x.unsqueeze(-1)], dim=-1)
        x = F.silu((conv_window * self.conv1d.weight.squeeze(1)).sum(-1) + self.conv1d.bias)
        delta, B, C = self.x_proj(x).split([self.dt_rank, self.d_state, self.d_state], dim=-1)
        delta = F.softplus(self.dt_proj(delta)).unsqueeze(-1)  # (batch, d, 1)
        A = -torch.exp(self.A_log.float())
        h = torch.exp(delta * A) * h + delta * x.unsqueeze(-1) * B.unsqueeze(1)  # h_t = A_bar h_{t-1} + B_bar x_t
        y = torch.einsum("bdn,bn->bd", h, C) + self.D * x  # y_t = C_t h_t + D x_t
        return self.out_proj(y * F.silu(z)), (conv_window, h)


# =====================================================================================================
# PART 4 - The Mamba-2 block (deck Figure 8)
# =====================================================================================================


def segsum(x):
    """out[..., i, j] = x[..., j+1] + ... + x[..., i] for j <= i, and -inf above the diagonal.

    With x the per-step log-decays, exp(segsum(x)) says how much of step j is still in the state at step i.
    """
    T = x.size(-1)
    x = x.unsqueeze(-1).expand(*x.shape, T)  # x[..., i, j] = x_i
    x = x.masked_fill(~torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device), -1), 0.0)
    out = torch.cumsum(x, dim=-2)
    return out.masked_fill(~torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device)), float("-inf"))


def ssd(x, log_decay, B, C, chunk):
    """Mamba-2's chunked scan, from "Transformers are SSMs" (state space duality).

    x          (batch, L, heads, p)   input already scaled by delta
    log_decay  (batch, L, heads)      delta * A, so exp() of it is A_bar; one scalar per head
    B, C       (batch, L, n)          shared by all heads
    Inside a chunk the SSM is computed exactly like causal attention: C plays the queries, B the keys, x the
    values, and the decay matrix replaces the softmax. Between chunks it is a recurrence on one state per chunk.
    """
    b, L, nh, p = x.shape
    pad = -L % chunk
    if pad:  # pad the end; later positions never affect earlier ones
        x, log_decay, B, C = (F.pad(t, (0, 0) * (t.dim() - 2) + (0, pad)) for t in (x, log_decay, B, C))
    c = x.size(1) // chunk
    x = x.reshape(b, c, chunk, nh, p)
    B, C = B.reshape(b, c, chunk, -1), C.reshape(b, c, chunk, -1)
    a = log_decay.reshape(b, c, chunk, nh).permute(0, 3, 1, 2)  # (batch, heads, chunks, chunk)
    a_cum = torch.cumsum(a, dim=-1)

    # 1. within each chunk: masked attention, scores = C . B, mask = decay matrix
    scores = torch.einsum("bcln,bcsn->bcls", C, B)
    y_diag = torch.einsum("bcls,bhcls,bcshp->bclhp", scores, torch.exp(segsum(a)), x)
    # 2. the state each chunk would leave behind if it had started from zero
    states = torch.einsum("bcln,bhcl,bclhp->bchpn", B, torch.exp(a_cum[..., -1:] - a_cum), x)
    # 3. pass states from chunk to chunk: the recurrence, one step per chunk
    states = torch.cat([torch.zeros_like(states[:, :1]), states], dim=1)
    chunk_decay = torch.exp(segsum(F.pad(a_cum[..., -1], (1, 0))))
    states = torch.einsum("bhzc,bchpn->bzhpn", chunk_decay, states)[:, :-1]  # the state entering each chunk
    # 4. what each position reads out of the state that entered its chunk
    y_off = torch.einsum("bcln,bchpn,bhcl->bclhp", C, states, torch.exp(a_cum))
    return (y_diag + y_off).reshape(b, c * chunk, nh, p)[:, :L]


class Mamba2Block(nn.Module):
    """The "parallel block": z, x, B, C and delta all come out of one projection at the entrance.

    Differences from Mamba-1: delta and A are one scalar per head instead of one value per channel, which is
    what lets the chunked algorithm use matrix multiplications; the state is larger; and the gated output is
    normalized before projecting down.
    """

    def __init__(self, cfg):
        super().__init__()
        d_inner = cfg.expand * cfg.d_model
        assert d_inner % cfg.headdim == 0, "expand * d_model must be divisible by headdim"
        self.d_inner, self.d_state, self.d_conv = d_inner, cfg.d_state, cfg.d_conv
        self.headdim, self.nheads, self.chunk = cfg.headdim, d_inner // cfg.headdim, cfg.chunk_size
        conv_dim = d_inner + 2 * cfg.d_state  # x, B and C all go through the convolution
        self.in_proj = nn.Linear(cfg.d_model, 2 * d_inner + 2 * cfg.d_state + self.nheads, bias=False)
        self.conv1d = nn.Conv1d(conv_dim, conv_dim, cfg.d_conv, groups=conv_dim, padding=cfg.d_conv - 1)
        self.dt_bias = nn.Parameter(torch.empty(self.nheads))
        _init_dt_bias(self.dt_bias)
        self.A_log = nn.Parameter(torch.log(torch.empty(self.nheads).uniform_(1, 16)))
        self.D = nn.Parameter(torch.ones(self.nheads))
        self.dt_bias._no_weight_decay = self.A_log._no_weight_decay = self.D._no_weight_decay = True
        self.norm = RMSNorm(d_inner)
        self.out_proj = nn.Linear(d_inner, cfg.d_model, bias=False)

    def _split_proj(self, u):
        return self.in_proj(u).split([self.d_inner, self.d_inner + 2 * self.d_state, self.nheads], dim=-1)

    def forward(self, u):  # u: (batch, L, d_model)
        b, L, _ = u.shape
        z, xBC, dt = self._split_proj(u)
        xBC = F.silu(self.conv1d(xBC.transpose(1, 2))[..., :L].transpose(1, 2))
        x, B, C = xBC.split([self.d_inner, self.d_state, self.d_state], dim=-1)
        with _no_autocast(u):
            x, B, C = x.float().reshape(b, L, self.nheads, self.headdim), B.float(), C.float()
            dt = F.softplus(dt.float() + self.dt_bias)  # (batch, L, heads)
            A = -torch.exp(self.A_log)
            y = ssd(x * dt.unsqueeze(-1), dt * A, B, C, self.chunk) + x * self.D.unsqueeze(-1)
        return self.out_proj(self.norm(y.reshape(b, L, self.d_inner) * F.silu(z)))

    def init_cache(self, batch, device, dtype=torch.float32):
        conv_window = torch.zeros(batch, self.d_inner + 2 * self.d_state, self.d_conv, device=device, dtype=dtype)
        h = torch.zeros(batch, self.nheads, self.headdim, self.d_state, device=device, dtype=dtype)
        return conv_window, h

    def step(self, u, cache):  # u: (batch, d_model)
        conv_window, h = cache
        z, xBC, dt = self._split_proj(u)
        conv_window = torch.cat([conv_window[..., 1:], xBC.unsqueeze(-1)], dim=-1)
        xBC = F.silu((conv_window * self.conv1d.weight.squeeze(1)).sum(-1) + self.conv1d.bias)
        x, B, C = xBC.split([self.d_inner, self.d_state, self.d_state], dim=-1)
        x = x.reshape(-1, self.nheads, self.headdim)
        dt = F.softplus(dt + self.dt_bias)  # (batch, heads)
        A = -torch.exp(self.A_log)
        h = torch.exp(dt * A)[..., None, None] * h + torch.einsum("bh,bn,bhp->bhpn", dt, B, x)
        y = torch.einsum("bhpn,bn->bhp", h, C) + x * self.D.unsqueeze(-1)
        return self.out_proj(self.norm(y.reshape(-1, self.d_inner) * F.silu(z))), (conv_window, h)


# =====================================================================================================
# PART 5 - The language model
# =====================================================================================================


@dataclass
class MambaConfig:
    vocab_size: int
    arch: str = "mamba1"  # "mamba1" (Figure 7) or "mamba2" (Figure 8)
    d_model: int = 128  # width of the residual stream
    n_layers: int = 4
    d_state: int = 0  # N, the state size per channel. 0 = 16 for mamba1, 64 for mamba2
    d_conv: int = 4  # width of the short causal convolution
    expand: int = 2  # the paths inside a block are expand x d_model wide
    dt_rank: int = 0  # mamba1: rank of the delta projection. 0 = ceil(d_model / 16)
    headdim: int = 32  # mamba2: channels per head
    chunk_size: int = 64  # mamba2: chunk length of the SSD algorithm
    scan: str = "auto"  # mamba1: "parallel" (Figure 6), "sequential" (plain recurrence), or "auto":
    #                     sequential on CPU (less total work), parallel on GPU (fewer, larger operations)

    def __post_init__(self):
        if self.d_state == 0:
            self.d_state = 16 if self.arch == "mamba1" else 64
        if self.dt_rank == 0:
            self.dt_rank = math.ceil(self.d_model / 16)


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        xf = x.float()
        return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight).type_as(x)


class ResidualBlock(nn.Module):
    """x + Block(RMSNorm(x)). Step 08 of the deck: the block's update is added to the untouched input."""

    def __init__(self, cfg):
        super().__init__()
        self.norm = RMSNorm(cfg.d_model)
        self.mixer = MambaBlock(cfg) if cfg.arch == "mamba1" else Mamba2Block(cfg)

    def forward(self, x):
        return x + self.mixer(self.norm(x))

    def step(self, x, cache):
        y, cache = self.mixer.step(self.norm(x), cache)
        return x + y, cache


class MambaLM(nn.Module):
    def __init__(self, cfg: MambaConfig):
        super().__init__()
        self.cfg = cfg
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.layers = nn.ModuleList(ResidualBlock(cfg) for _ in range(cfg.n_layers))
        self.norm_f = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight  # input and output embeddings are shared
        self.grad_checkpoint = False
        # Initialization as in the official code: zero biases (except delta's), small embeddings, and the
        # projections that write into the residual stream shrunk by sqrt(n_layers).
        for m in self.modules():
            if isinstance(m, nn.Linear) and m.bias is not None and not getattr(m.bias, "_no_reinit", False):
                nn.init.zeros_(m.bias)
        nn.init.normal_(self.embedding.weight, std=0.02)
        with torch.no_grad():
            for name, p in self.named_parameters():
                if name.endswith("out_proj.weight"):
                    p /= math.sqrt(cfg.n_layers)

    def forward(self, idx, targets=None):
        """Whole sequences at once (training). idx: (batch, L) token ids."""
        x = self.embedding(idx)
        for layer in self.layers:
            if self.grad_checkpoint and self.training:  # recompute in backward instead of storing
                x = checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x)
        logits = self.lm_head(self.norm_f(x))
        if targets is None:
            return logits
        return logits, F.cross_entropy(logits.float().flatten(0, 1), targets.flatten())

    def init_cache(self, batch=1):
        w = self.embedding.weight
        return [layer.mixer.init_cache(batch, w.device, w.dtype) for layer in self.layers]

    def step(self, token, cache):
        """One token at a time (generation). Updates each layer's fixed-size state, returns next-token logits."""
        x = self.embedding(token)
        new_cache = []
        for layer, layer_cache in zip(self.layers, cache):
            x, layer_cache = layer.step(x, layer_cache)
            new_cache.append(layer_cache)
        return self.lm_head(self.norm_f(x)), new_cache

    @torch.no_grad()
    def generate(self, prompt_ids, max_new_tokens, temperature=1.0, top_k=0):
        """Yields new token ids one by one. Only `cache` is carried forward, so memory never grows."""
        device = self.embedding.weight.device
        cache = self.init_cache(1)
        for t in prompt_ids:  # read the prompt like the deck's notebook: one token at a time
            logits, cache = self.step(torch.tensor([t], device=device), cache)
        for _ in range(max_new_tokens):
            if temperature <= 0:
                next_id = logits.argmax(-1)
            else:
                logits = logits / temperature
                if top_k:
                    kth = torch.topk(logits, min(top_k, logits.size(-1))).values[..., -1:]
                    logits = logits.masked_fill(logits < kth, float("-inf"))
                next_id = torch.multinomial(F.softmax(logits, dim=-1), 1)[:, 0]
            yield next_id.item()
            logits, cache = self.step(next_id, cache)


def state_bytes(model):
    """Size of everything generation carries from one token to the next."""
    return sum(t.numel() * t.element_size() for layer_cache in model.init_cache(1) for t in layer_cache)


# =====================================================================================================
# PART 6 - Training
# =====================================================================================================


def pick_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_checkpoint(path, model, tok, step, val_loss):
    torch.save(
        {"config": asdict(model.cfg), "model": model.state_dict(), "vocab": tok.chars, "step": step, "val_loss": val_loss},
        path,
    )


def load_checkpoint(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model = MambaLM(MambaConfig(**ckpt["config"])).to(device)
    model.load_state_dict(ckpt["model"])
    return model.eval(), CharTokenizer(ckpt["vocab"]), ckpt


def sample_text(model, tok, prompt, n, temperature=0.8, top_k=0):
    was_training = model.training
    model.eval()
    ids = tok.encode(prompt) or [0]
    text = tok.decode(list(model.generate(ids, n, temperature, top_k)))
    model.train(was_training)
    return text


def train(args):
    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    text = load_text(args.data)
    tok = CharTokenizer.from_text(text)
    data = torch.tensor(tok.encode(text), dtype=torch.long)
    n_val = max(len(data) // 10, args.seq_len + 2)
    train_data, val_data = data[:-n_val], data[-n_val:]
    print(f"data: {len(data):,} characters, vocabulary of {tok.vocab_size}, last {n_val:,} held out for validation")

    cfg = MambaConfig(
        vocab_size=tok.vocab_size, arch=args.arch, d_model=args.d_model, n_layers=args.n_layers, d_state=args.d_state,
        expand=args.expand, headdim=args.headdim, chunk_size=args.chunk_size, scan=args.scan,
    )
    model = MambaLM(cfg).to(device)
    model.grad_checkpoint = args.grad_checkpoint
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {cfg.arch}, {cfg.n_layers} layers, d_model {cfg.d_model}, state {cfg.d_state}, "
          f"{n_params:,} parameters, on {device}")

    # Weight decay on the weight matrices only; not on biases, norms, A, D or delta's bias.
    decay = lambda p: p.dim() >= 2 and not getattr(p, "_no_weight_decay", False)
    groups = [
        {"params": [p for p in model.parameters() if decay(p)], "weight_decay": args.weight_decay},
        {"params": [p for p in model.parameters() if not decay(p)], "weight_decay": 0.0},
    ]
    opt = torch.optim.AdamW(groups, lr=args.lr, betas=(0.9, 0.95))

    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    amp = lambda: torch.autocast("cuda", dtype=torch.bfloat16) if use_bf16 else contextlib.nullcontext()

    def lr_at(step):  # linear warmup, then cosine decay to 10% of the peak
        if step < args.warmup:
            return args.lr * (step + 1) / args.warmup
        progress = (step - args.warmup) / max(1, args.steps - args.warmup)
        return args.lr * (0.1 + 0.45 * (1 + math.cos(math.pi * progress)))

    @torch.no_grad()
    def val_loss():
        model.eval()
        losses = []
        for _ in range(args.eval_iters):
            x, y = get_batch(val_data, args.batch_size, args.seq_len, device)
            with amp():
                losses.append(model(x, y)[1].item())
        model.train()
        return sum(losses) / len(losses)

    best, saved = float("inf"), False
    sample_prompt = "\n" if "\n" in tok.index else tok.chars[0]
    model.train()
    t0, tokens = time.time(), 0
    try:
        for step in range(1, args.steps + 1):
            lr = lr_at(step - 1)
            for g in opt.param_groups:
                g["lr"] = lr
            x, y = get_batch(train_data, args.batch_size, args.seq_len, device)
            with amp():
                _, loss = model(x, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            tokens += x.numel()

            if step % args.log_every == 0 or step == 1:
                rate = tokens / (time.time() - t0)
                minutes_left = (args.steps - step) * x.numel() / rate / 60
                print(f"step {step:5d}/{args.steps} | loss {loss.item():.3f} | lr {lr:.1e} | "
                      f"{rate:,.0f} tokens/s | ~{minutes_left:.0f} min left")
                t0, tokens = time.time(), 0

            if step % args.eval_every == 0 or step == args.steps:
                v = val_loss()
                note = ""
                if v < best:
                    best, saved = v, True
                    save_checkpoint(args.out, model, tok, step, v)
                    note = f" | saved {args.out}"
                print(f"  validation loss {v:.3f} ({v / math.log(2):.2f} bits per character){note}")
                print("  sample: " + sample_text(model, tok, sample_prompt, 200).replace("\n", "\n          "))
                t0, tokens = time.time(), 0
    except KeyboardInterrupt:
        print("\nstopped early")
        if not saved:
            save_checkpoint(args.out, model, tok, step, float("nan"))
            print(f"saved the current weights to {args.out}")
    result = f" best validation loss {best:.3f}." if best < float("inf") else ""
    print(f"done.{result} try:  python {os.path.basename(sys.argv[0])} generate --ckpt {args.out}")


# =====================================================================================================
# PART 7 - Generation
# =====================================================================================================


def generate(args):
    device = pick_device(args.device)
    model, tok, ckpt = load_checkpoint(args.ckpt, device)
    if args.seed is not None:
        torch.manual_seed(args.seed)
    prompt = args.prompt or ("\n" if "\n" in tok.index else tok.chars[0])
    ids = tok.encode(prompt) or [0]
    val = f", validation loss {ckpt['val_loss']:.3f}" if not math.isnan(ckpt["val_loss"]) else ""
    print(f"[{ckpt['config']['arch']} model from step {ckpt['step']}{val}]", file=sys.stderr)
    print(tok.decode(ids), end="", flush=True)
    for i in model.generate(ids, args.max_new, args.temperature, args.top_k):
        print(tok.decode([i]), end="", flush=True)
    print()
    print(f"[state carried between tokens: {state_bytes(model) / 1024:.1f} KB, "
          f"the same after {len(ids) + args.max_new:,} tokens as after one]", file=sys.stderr)


# =====================================================================================================
# PART 8 - Self-test: the fast paths must agree with the plain recurrence
# =====================================================================================================


def self_test():
    torch.manual_seed(0)

    # 1. The parallel scan gives exactly the sequential recurrence, values and gradients.
    for L in (1, 2, 5, 8, 37, 128):
        a = torch.rand(2, L, 3, 4, dtype=torch.float64, requires_grad=True)
        b = torch.randn(2, L, 3, 4, dtype=torch.float64, requires_grad=True)
        h_par, h_seq = parallel_scan(a, b), sequential_scan(a, b)
        w = torch.randn_like(h_seq)
        grads = lambda h: [torch.zeros_like(t) if g is None else g  # with one token, h never touches a
                           for t, g in zip((a, b), torch.autograd.grad((h * w).sum(), (a, b), allow_unused=True))]
        assert torch.allclose(h_par, h_seq, atol=1e-10), f"parallel scan differs at length {L}"
        assert all(torch.allclose(p, s, atol=1e-10) for p, s in zip(grads(h_par), grads(h_seq))), f"gradients differ at length {L}"
    print("ok  parallel scan == sequential recurrence (values and gradients, lengths 1 to 128)")

    # 2. Reading a sequence all at once == reading it one token at a time with a fixed-size state.
    for arch in ("mamba1", "mamba2"):
        cfg = MambaConfig(vocab_size=13, arch=arch, d_model=32, n_layers=2, headdim=16, chunk_size=8, scan="parallel")
        model = MambaLM(cfg).eval()
        idx = torch.randint(0, 13, (2, 37))  # 37 is deliberately not a multiple of the chunk size
        with torch.no_grad():
            full = model(idx)
            cache, steps = model.init_cache(2), []
            for t in range(idx.size(1)):
                logits, cache = model.step(idx[:, t], cache)
                steps.append(logits)
            stepwise = torch.stack(steps, dim=1)
            if arch == "mamba1":
                for layer in model.layers:
                    layer.mixer.scan = "sequential"
                assert torch.allclose(model(idx), full, atol=1e-5), "mamba1: sequential and parallel scans differ"
        err = (full - stepwise).abs().max().item()
        assert err < 1e-4, f"{arch}: forward and step-by-step differ by {err:.2e}"
        print(f"ok  {arch}: whole-sequence forward == token-by-token recurrence (max difference {err:.1e})")

    # 3. Both architectures can learn: memorize a short repeating text.
    text = "He is Rasul, an AI engineer at Softworks. He works on building ASR model for Bangla language. "
    tok = CharTokenizer.from_text(text)
    data = torch.tensor(tok.encode(text * 50))
    for arch in ("mamba1", "mamba2"):
        torch.manual_seed(0)
        model = MambaLM(MambaConfig(vocab_size=tok.vocab_size, arch=arch, d_model=64, n_layers=2, chunk_size=32))
        opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
        for i in range(150):
            x, y = get_batch(data, 16, 64, "cpu")
            loss = model(x, y)[1]
            opt.zero_grad()
            loss.backward()
            opt.step()
            if i == 0:
                first = loss.item()
        assert loss.item() < 0.25 * first, f"{arch}: loss only went from {first:.2f} to {loss.item():.2f}"
        print(f"ok  {arch}: learns a repeating sentence (loss {first:.2f} -> {loss.item():.2f})")
    print("all tests passed")


# =====================================================================================================
# PART 9 - Command line
# =====================================================================================================


def main():
    if hasattr(sys.stdout, "reconfigure"):  # print Bangla on any console, and show progress even when piped to a file
        sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    p = argparse.ArgumentParser(description="Train and run a small Mamba language model.")
    sub = p.add_subparsers(dest="command", required=True)

    t = sub.add_parser("train", help="train on a text file", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    t.add_argument("--data", help="UTF-8 text file; if omitted, Tiny Shakespeare is downloaded")
    t.add_argument("--out", default="mamba.pt", help="where to save the checkpoint")
    t.add_argument("--arch", choices=["mamba1", "mamba2"], default="mamba1")
    t.add_argument("--d-model", type=int, default=128)
    t.add_argument("--n-layers", type=int, default=4)
    t.add_argument("--d-state", type=int, default=0, help="state size N; 0 = 16 for mamba1, 64 for mamba2")
    t.add_argument("--expand", type=int, default=2)
    t.add_argument("--headdim", type=int, default=32, help="mamba2 only")
    t.add_argument("--chunk-size", type=int, default=64, help="mamba2 only")
    t.add_argument("--scan", choices=["auto", "parallel", "sequential"], default="auto",
                   help="mamba1 only; auto = sequential on CPU, parallel on GPU")
    t.add_argument("--seq-len", type=int, default=128)
    t.add_argument("--batch-size", type=int, default=16)
    t.add_argument("--steps", type=int, default=2000)
    t.add_argument("--lr", type=float, default=3e-3)
    t.add_argument("--warmup", type=int, default=100)
    t.add_argument("--weight-decay", type=float, default=0.1)
    t.add_argument("--grad-clip", type=float, default=1.0)
    t.add_argument("--log-every", type=int, default=50)
    t.add_argument("--eval-every", type=int, default=250)
    t.add_argument("--eval-iters", type=int, default=20)
    t.add_argument("--grad-checkpoint", action="store_true", help="recompute activations in backward to save memory")
    t.add_argument("--device", default="auto", help="auto, cpu, cuda or mps")
    t.add_argument("--seed", type=int, default=1337)

    g = sub.add_parser("generate", help="sample text from a checkpoint", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    g.add_argument("--ckpt", default="mamba.pt")
    g.add_argument("--prompt", default="")
    g.add_argument("--max-new", type=int, default=500, help="number of characters to generate")
    g.add_argument("--temperature", type=float, default=0.8, help="0 = always pick the most likely character")
    g.add_argument("--top-k", type=int, default=0, help="sample only from the k most likely characters; 0 = all")
    g.add_argument("--device", default="auto")
    g.add_argument("--seed", type=int)

    sub.add_parser("test", help="check that the parallel and step-by-step paths agree")

    args = p.parse_args()
    {"train": train, "generate": generate, "test": lambda _: self_test()}[args.command](args)


if __name__ == "__main__":
    main()
