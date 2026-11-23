"""Independent check (matched filter) of "frames are crop+resize of larger watermarked images".
For each candidate canvas size S = 512*s: canvas = centred reflect-pad of the frame y, w_S = Emb(canvas, msg) - canvas
(official kit embedder). Score(s, offset) = normalised correlation between high-passed w_S cropped at offset and
the high-passed frame y. All offsets at once via conv2d. Null: same search with random messages.
    python -m experiments.geo_check2 --n 5 --nsmall 2 --device cuda:0
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


def hp(x, sigma=3.0):
    k = int(3 * sigma) * 2 + 1; t = torch.arange(k, device=x.device) - k // 2
    g = torch.exp(-t.float() ** 2 / (2 * sigma ** 2)); g = g / g.sum()
    c = x.shape[1]
    b = F.conv2d(F.pad(x, (k // 2,) * 4, mode="reflect"), g.view(1, 1, 1, k).repeat(c, 1, 1, 1), groups=c)
    b = F.conv2d(b, g.view(1, 1, k, 1).repeat(c, 1, 1, 1), groups=c)
    return x - b


def score_map(wS, yhp):
    """wS: 1x3xSxS high-passed watermark; yhp: 1x3x512x512. -> (P+1)x(P+1) normalised correlation."""
    num = F.conv2d(wS, yhp)[0, 0]
    en = F.conv2d((wS ** 2).sum(1, keepdim=True), torch.ones(1, 1, 512, 512, device=wS.device))[0, 0]
    return num / (en.sqrt() * yhp.norm() + 1e-9)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=5); ap.add_argument("--nsmall", type=int, default=2)
    ap.add_argument("--device", default="cuda:0"); ap.add_argument("--nrand", type=int, default=10)
    ap.add_argument("--methods", default="stegastamp,trustmark,pixelseal")
    a = ap.parse_args()
    me, rg = methods(), regime(); rng = np.random.RandomState(0); dev = a.device
    for meth in a.methods.split(","):
        wm = load_watermark(meth, models_dir=ROOT / "models", device=dev); t0 = time.time()
        idx = list(np.where((me == meth) & rg)[0][:a.n]) + list(np.where((me == meth) & ~rg)[0][:a.nsmall])
        for i in idx:
            y = image("train", int(i)); yt = torch.from_numpy(y).permute(2, 0, 1)[None].float().to(dev)
            yhp = hp(yt); m = message("train", int(i))
            msgs = [m] + [rng.randint(0, 2, len(m)) for _ in range(a.nrand)]
            res = np.zeros((len(msgs), len(SCALES))); arg = {}
            for si, s in enumerate(SCALES):
                S = int(round(512 * s)); P = S - 512; L = P // 2
                canvas = np.pad(y, ((L, P - L), (L, P - L), (0, 0)), mode="reflect") if P else y
                Ws = torch.stack([torch.from_numpy(wm.embed(canvas, mm).astype(np.float32) - canvas.astype(np.float32))
                                  .permute(2, 0, 1).to(dev) for mm in msgs])
                tot = Ws.sum(0)
                for k in range(len(msgs)):
                    d = Ws[k] - (tot - Ws[k]) / (len(msgs) - 1)          # message-specific part (leave-one-out mean)
                    wS = hp(d[None])
                    sm = score_map(wS, yhp)
                    res[k, si] = sm.max().item()
                    if k == 0:
                        j = int(sm.argmax()); arg[s] = (j // sm.shape[1] - L, j % sm.shape[1] - L)   # (dy, dx) vs centred
            true, null = res[0], res[1:]
            z = (true - null.mean(0)) / (null.std(0) + 1e-9)
            zbest_null = (null.max(1) - null.mean()) / (null.std() + 1e-9)
            b = int(np.argmax(true))
            print(f"{meth:10s} {'FULL ' if rg[i] else 'small'} {i:3d} corr@s=1 {true[0]:.4f} (z {z[0]:+.1f}) | best s={SCALES[b]} "
                  f"corr {true[b]:.4f} z {z[b]:+.1f} shift(dy,dx)={arg[SCALES[b]]} | z per s: "
                  + " ".join(f"{s}:{v:+.1f}" for s, v in zip(SCALES, z)), flush=True)
        print(f"{meth} {time.time() - t0:.0f}s", flush=True)
        del wm; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
