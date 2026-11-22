"""Independent check of the teammate claim "frames are crop+resize of larger watermarked images (scale 1.03-1.15,
shift <= 20 px)". For full-regime images: reflect-pad the 512 frame to a canvas covering a hypothetical embedding
rectangle of size S = 512*s at offset (L, T) <= 0, decode the canvas with the official decoder, and measure the
agreement with the KNOWN message, agree = mean (2m-1) tanh(logit). Null: same search with random messages.
    python -m src.geo_check --n 6 --device cuda:0
"""
import argparse
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from src.data import ROOT, image, message, methods, regime

sys.path.insert(0, str(ROOT / "kit"))
from wmloc.watermarks import load_watermark  # noqa: E402

SCALES = (1.0, 1.03, 1.06, 1.1, 1.15, 1.2)


def canvases(x, step=8):
    """x: 1x3x512x512 -> (params list, B x 3 x S x S tensors grouped by S)."""
    out = []
    for s in SCALES:
        S = int(round(512 * s)); P = S - 512
        offs = sorted(set(list(range(0, P + 1, step)) + [P // 2, P]))
        batch, prm = [], []
        for L in offs:
            for T in offs:
                c = F.pad(x, (L, P - L, T, P - T), mode="reflect") if P else x
                batch.append(c); prm.append((s, -L, -T))
        out.append((prm, torch.cat(batch)))
    return out


def agree(lg, m):
    return (np.tanh(lg) * (2 * m[None].astype(np.float32) - 1)).mean(1)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=6); ap.add_argument("--nsmall", type=int, default=3)
    ap.add_argument("--device", default="cuda:0"); ap.add_argument("--nrand", type=int, default=20)
    a = ap.parse_args()
    me, rg = methods(), regime(); rng = np.random.RandomState(0)
    for meth in ("trustmark", "stegastamp", "pixelseal"):
        wm = load_watermark(meth, models_dir=ROOT / "models", device=a.device); t0 = time.time()
        idx = list(np.where((me == meth) & rg)[0][:a.n]) + list(np.where((me == meth) & ~rg)[0][:a.nsmall])
        for i in idx:
            x = torch.from_numpy(image("train", int(i))).permute(2, 0, 1)[None].float().div(255).to(a.device)
            m = message("train", int(i)); R = rng.randint(0, 2, (a.nrand, len(m)))
            prm, lg = [], []
            for p, c in canvases(x):
                prm += p
                lg.append(np.concatenate([wm.logits(c[j:j + 64]).cpu().numpy() for j in range(0, len(c), 64)]))
            lg = np.concatenate(lg)
            ag = agree(lg, m); agr = np.stack([agree(lg, r) for r in R])          # nrand x ncand
            b = int(np.argmax(ag)); null_gain = (agr.max(1) - agr[:, 0])
            print(f"{meth:10s} {'FULL ' if rg[i] else 'small'} {i:3d} agree@id {ag[0]:+.3f} best {ag[b]:+.3f} at s={prm[b][0]} "
                  f"L={prm[b][1]} T={prm[b][2]} | gain {ag[b] - ag[0]:+.3f} vs null gain {null_gain.mean():+.3f}±{null_gain.std():.3f}"
                  f" | best per s: " + " ".join(f"{s}:{max(ag[k] for k, q in enumerate(prm) if q[0] == s):+.2f}" for s in SCALES), flush=True)
        print(f"{meth} {time.time() - t0:.0f}s", flush=True)
        del wm; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
