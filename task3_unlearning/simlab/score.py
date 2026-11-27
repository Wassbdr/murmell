"""Toy version of the task-3 score on seeded synthetic data (numpy only).

Same structure as the official score: score = forget x retain, forget = (loss mean/std match) x (per-image prediction
match, KL at T = 2), retain = relative accuracy x 1 / (1 + KL to the original). A candidate is described only by the
few GLOBAL target parameters our refit loop moves (loss mean, loss std, temperature, retention weight); the per-image
behaviour follows from them plus seeded per-image noise. The public score uses a fixed 30 % of the images (hash split).
"""
import hashlib

import numpy as np

PARAMS = ("loss_mean", "loss_std", "temperature", "retain_w")
LO = np.array([0.5, 0.3, 0.5, 0.0]); HI = np.array([4.0, 3.0, 2.5, 3.0])


class World:
    def __init__(self, seed=0, n=3500, simulator=False):
        """simulator=True: our offline replica of the scorer, built without the hidden gold: its gold temperature and gold
        losses are off by a seeded error, as our own simulator was (it is scored on all images, not the public split)."""
        r = np.random.default_rng(seed)
        self.n = n
        self.diff = r.normal(0, 1, n)                                    # per-image difficulty
        self.gold_loss = np.exp(0.6 + 0.55 * self.diff + r.normal(0, 0.25, n))   # gold (never-seen) loss per image
        self.gold_T = 1.35 + r.normal(0, 0.05)                           # temperature that matches gold wrong-class shape
        self.noise = r.normal(0, 1, n)
        if simulator:
            e = np.random.default_rng(seed + 7)
            self.gold_T += e.normal(0, 0.05); self.gold_loss = self.gold_loss * np.exp(e.normal(0, 0.03, n))
        self.public = np.ones(n, bool) if simulator else np.array([int(hashlib.md5(f"{seed}-{i}".encode()).hexdigest(), 16) % 10 < 3 for i in range(n)])

    def per_image(self, theta):
        mu, sd, T, w = np.clip(theta, LO, HI)
        z = (self.diff - self.diff.mean()) / self.diff.std()
        loss = np.maximum(0.01, mu + sd * (0.9 * z + 0.44 * self.noise))   # rank-mapped onto the target moments
        kl = 0.08 * (T - self.gold_T) ** 2 + 0.02 * (np.log(loss) - np.log(self.gold_loss)) ** 2
        return loss, kl

    def components(self, theta, mask=None):
        mask = np.ones(self.n, bool) if mask is None else mask
        mu, sd, T, w = np.clip(theta, LO, HI)
        loss, kl = self.per_image(theta); loss, kl = loss[mask], kl[mask]; g = self.gold_loss[mask]
        moments = np.exp(-abs(loss.mean() - g.mean()) / g.mean() - abs(loss.std() - g.std()) / g.std())
        pred = float(np.mean(np.exp(-kl)))
        forget = moments * pred
        drift = 0.12 * mu * (1 + 0.3 * abs(T - 1)) / (1 + w)              # pushing harder moves the model elsewhere
        rel_acc = 1 - 0.04 * w - 0.25 * drift
        retain = rel_acc / (1 + drift)
        return dict(moments=float(moments), prediction=pred, forget=float(forget), retain=float(retain),
                    score=float(forget * retain))

    def public_score(self, theta):
        return self.components(theta, self.public)["score"]
