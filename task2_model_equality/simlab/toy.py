"""Toy language models that stand in for the LLM checkpoints of task 2 (numpy only, seeded).

A base model is a bigram softmax over V tokens with logits W[prev] + b (b = system-prompt bias). Endpoints add the
disguises and traps of the competition on top of it:
  disguises (same model): replaced system prompt, persona, temperature, top_k, keyed watermark sampler
  traps (other models)  : fp8-like rounding of W, one low-rank adapter W + dose * A B^T served at 4 doses
A "reset" request neutralises the system-prompt disguises, as the chat-template reset did on the real endpoints.
"""
import hashlib
from dataclasses import dataclass, field

import numpy as np

V = 200          # vocabulary
T0 = 0.7         # default sampling temperature of every family


def softmax(z):
    z = z - z.max(-1, keepdims=True); e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def q8(W, scale=8.0):
    """fp8-like weight rounding: 3 mantissa bits after scaling (a crude stand-in for e4m3)."""
    x = W * scale; e = np.floor(np.log2(np.maximum(np.abs(x), 1e-9))); step = 2.0 ** (e - 3)
    return np.round(x / step) * step / scale


@dataclass
class Base:
    W: np.ndarray
    b_default: np.ndarray

    @classmethod
    def random(cls, seed, scale=2.2):
        r = np.random.default_rng(seed)
        return cls(r.normal(0, scale, (V, V)), r.normal(0, 0.3, V))


@dataclass
class Endpoint:
    name: str
    family: str
    role: str                       # plain, system, persona, hot, topk, watermark, fp8, lora@<dose>
    W: np.ndarray
    b: np.ndarray                   # bias of its system context (plain answers)
    b_reset: np.ndarray             # bias after a reset (default system prompt)
    T: float = T0
    top_k: int = 0
    wm_key: int = 0
    model: str = "base"             # ground-truth model id inside the family
    rng: np.random.Generator = field(default=None, repr=False)

    def probs(self, prev, reset):
        z = (self.W[prev] + (self.b_reset if reset else self.b)) / self.T
        if self.top_k:
            kth = np.partition(z, -self.top_k)[-self.top_k]; z = np.where(z >= kth, z, -np.inf)
        return softmax(z)

    def generate(self, prompt, n_tok, reset=True):
        prev = int(hashlib.md5(prompt.encode()).hexdigest(), 16) % V; out = []
        for t in range(n_tok):
            p = self.probs(prev, reset)
            if self.wm_key:              # keyed sampler: Gumbel-max with noise fixed by (key, prompt, position)
                h = int(hashlib.md5(f"{self.wm_key}|{prompt}|{t}".encode()).hexdigest()[:8], 16)
                g = np.random.default_rng(h).gumbel(size=V)
                x = int(np.argmax(np.log(np.maximum(p, 1e-300)) + g))
            else:
                x = int(self.rng.choice(V, p=p))
            out.append(x); prev = x
        return out


def family_endpoints(fam, base, seed, doses=(1.0, 0.6, 0.35, 0.2), rank=4, lora_scale=0.6):
    """11 endpoints per family, as on the real leaderboard: 6 disguises of the base, one fp8 copy, one adapter at 4 doses."""
    r = np.random.default_rng(seed)
    A = r.normal(0, 1, (V, rank)); B = r.normal(0, 1, (V, rank)); dW = lora_scale * A @ B.T / np.sqrt(rank)
    alt = lambda s: base.b_default + r.normal(0, s, V)
    W, b0 = base.W, base.b_default
    spec = [("plain", W, b0, T0, 0, 0, "base"), ("system", W, alt(1.2), T0, 0, 0, "base"),
            ("persona", W, alt(2.5), T0, 0, 0, "base"), ("hot", W, b0, 1.0, 0, 0, "base"),
            ("topk", W, b0, 0.9, 20, 0, "base"), ("watermark", W, b0, T0, 0, int(r.integers(1, 2 ** 30)), "base"),
            ("fp8", q8(W), b0, T0, 0, 0, "fp8")]
    spec += [(f"lora@{d}", W + d * dW, b0, T0, 0, 0, f"lora{d}") for d in doses]
    eps = []
    for role, Wm, b, T, k, key, model in spec:
        eps.append(Endpoint("", fam, role, Wm, b, b0 if role not in ("system", "persona") else b0, T, k, key, model,
                            np.random.default_rng(r.integers(2 ** 31))))
    return eps


def build_bench(seed, n_fam=3):
    """33 endpoints (3 families x 11) in a seeded random order, named ep-01 .. ep-33; returns endpoints and the bases."""
    r = np.random.default_rng(seed)
    fams = [f"F{i}" for i in range(n_fam)]
    bases = {f: Base.random(int(r.integers(2 ** 31))) for f in fams}
    eps = [e for f in fams for e in family_endpoints(f, bases[f], int(r.integers(2 ** 31)))]
    order = r.permutation(len(eps))
    eps = [eps[i] for i in order]
    for i, e in enumerate(eps):
        e.name = f"ep-{i + 1:02d}"
    return eps, bases


def collect(ep, n_plain=16, n_rep=4, rep_k=4, n_long=60, short=24, long=64):
    """at most 100 answers per endpoint, as on the real endpoints: plain prompts, repeated identical requests, long reset answers."""
    rows = []
    for i in range(n_plain):
        rows.append(dict(kind="plain", prompt=f"plain-{i}", toks=ep.generate(f"plain-{i}", short, reset=False)))
    for i in range(n_rep):
        for _ in range(rep_k):
            rows.append(dict(kind="repeat", prompt=f"rep-{i}", toks=ep.generate(f"rep-{i}", short, reset=True)))
    for i in range(n_long):
        rows.append(dict(kind="long", prompt=f"long-{i}", toks=ep.generate(f"long-{i}", long, reset=True)))
    return rows
