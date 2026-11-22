"""Fit the full-frame embedding geometry (s, L, T) of an image with a KNOWN message.
canvas_s = reflect-pad of y to S = round(512 s) (centred); d_s = Emb(canvas_s, m) - mean_r Emb(canvas_s, r_r)
(message-specific part); score(s, offset) = normalised xcorr(hp(d_s) cropped, hp(y)); best over a coarse-to-fine s grid
and all integer offsets. Returns s, (L, T) = frame position inside the canvas, z vs random messages.
    python -m src.geo_fit --validate tm --n 12 --device cuda:1      (check on synthetic pool items with known geometry)
"""
import argparse
import sys

import numpy as np
import torch

from src.data import ROOT
from src.geo_check2 import hp, score_map

sys.path.insert(0, str(ROOT / "kit"))


class Fitter:
    def __init__(self, wm, device, nrand=4, seed=0):
        self.wm, self.dev, self.nrand = wm, device, nrand
        self.rng = np.random.RandomState(seed)

    def _maps(self, y, m, s, yhp, R):
        S = int(round(512 * s)); P = S - 512; c0 = P // 2
        can = np.pad(y, ((c0, P - c0), (c0, P - c0), (0, 0)), mode="reflect") if P else y
        W = torch.stack([torch.from_numpy(self.wm.embed(can, mm).astype(np.float32) - can.astype(np.float32))
                         .permute(2, 0, 1).to(self.dev) for mm in [m] + list(R)])
        tot = W.sum(0); out = []
        for k in range(len(W)):
            d = W[k] - (tot - W[k]) / (len(W) - 1)
            out.append(score_map(hp(d[None]), yhp))
        return out                                      # list of (P+1)x(P+1) maps; [0] = true message

    def fit(self, y, m, scales=None, refine=True):
        yhp = hp(torch.from_numpy(y).permute(2, 0, 1)[None].float().to(self.dev))
        R = [self.rng.randint(0, 2, len(m)) for _ in range(self.nrand)]
        scales = scales if scales is not None else np.round(np.arange(1.0, 1.255, 0.025), 3)
        res = []
        for s in scales:
            mp = self._maps(y, m, s, yhp, R)
            true = mp[0]; j = int(true.argmax()); T, L = divmod(j, true.shape[1])
            null = torch.stack([x.max() for x in mp[1:]])
            res.append((float(true.max()), float(s), L, T, float((true.max() - null.mean()) / (null.std() + 1e-9))))
        best = max(res, key=lambda r: r[0])
        if refine:
            fine = [round(best[1] + d, 3) for d in (-0.015, -0.01, -0.005, 0.005, 0.01, 0.015) if 1.0 <= best[1] + d <= 1.3]
            if fine:
                best = max([best] + [r for r in self.fit(y, m, fine, refine=False)[1]], key=lambda r: r[0])
        return best, res


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--validate", default="tm"); ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--device", default="cuda:1")
    a = ap.parse_args()
    from wmloc.watermarks import load_watermark
    from src.synth_full import ROOT as R0
    meth = {"ps": "pixelseal", "tm": "trustmark", "ss": "stegastamp"}[a.validate]
    wm = load_watermark(meth, models_dir=ROOT / "models", device=a.device); F = Fitter(wm, a.device)
    Y = np.load(R0 / "work/synth_full" / f"{meth}_s0.npy", mmap_mode="r"); z = np.load(R0 / "work/synth_full" / f"{meth}_s0.npz")
    for k in range(a.n):
        (sc, s, L, T, zz), _ = F.fit(np.asarray(Y[k]), z["msg"][k])
        print(f"{meth} item {k}: true s={z['s'][k]:.3f} L={z['L'][k]} T={z['T'][k]} | fit s={s:.3f} L={L} T={T} "
              f"(offset rel. centred: true {z['L'][k] - (int(round(512 * z['s'][k])) - 512) // 2},{z['T'][k] - (int(round(512 * z['s'][k])) - 512) // 2})"
              f" corr {sc:.4f} z {zz:.1f}", flush=True)


if __name__ == "__main__":
    main()
