"""PixelSeal INFORMED map (human list 20:25): re-embed the KNOWN message at a fitted geometry, keep only its
message-specific part, correlate it locally with the image.
  geometry (as in synth_full): frame = crop at (T, L) of an S x S canvas (S = 512 s, s in [1, 1.2]) embedded by PixelSeal
  (which works on the canvas resized to 256).  We rebuild a canvas = reflect-pad of the frame with the frame at (T', L'),
  T', L' in {0, 1} (sub-grid phase), S on a grid; pattern p = embed(canvas, m) - mean_r embed(canvas, r_j) cropped to the
  frame (K random messages r_j cancel the content-dependent, message-independent part).
  Pick (S, T', L') maximising the global normalised correlation <hp(img), hp(p)>; map = local normalised correlation in
  W x W windows (box filter), pooled to 128 x 128.  GT-free (method + message only).
    python -m src.ps_informed --split train --device cuda:0 [--limit 6 --diag 1] [--shard 0 --nshards 4]
Output work/psinf/<split>_<shard>of<n>.npz : names, maps [n,128,128] float16, S, T, L, global corr.
"""
import argparse
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from src.data import ROOT, image, masks128, message, methods, names, regime

sys.path.insert(0, str(ROOT / "kit"))


def hp(x):
    k = torch.ones(x.shape[1], 1, 7, 7, device=x.device) / 49
    return x - F.conv2d(F.pad(x, (3, 3, 3, 3), mode="reflect"), k, groups=x.shape[1])


def box(x, w):
    return F.avg_pool2d(F.pad(x, (w // 2, w - 1 - w // 2, w // 2, w - 1 - w // 2), mode="reflect"), w, 1)


@torch.no_grad()
def patterns(model, x, msg, S, T, L, K, gen):
    """x: 1x3x512x512 in [0,1] -> message-specific pattern 1x3x512x512 for canvas size S, frame offset (T, L)."""
    c = F.pad(x, (L, S - 512 - L, T, S - 512 - T), mode="reflect")
    m = torch.tensor(msg, dtype=torch.float32, device=x.device)[None]
    r = torch.randint(0, 2, (K, m.shape[1]), generator=gen).float().to(x.device)
    out = model.embed(c.repeat(K + 1, 1, 1, 1), msgs=torch.cat([m, r]), is_video=False)["imgs_w"]
    d = out[:1] - out[1:].mean(0, keepdim=True)          # clamp-free difference of watermarked canvases
    return d[:, :, T:T + 512, L:L + 512]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train"); ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0); ap.add_argument("--diag", type=int, default=0)
    ap.add_argument("--K", type=int, default=6); ap.add_argument("--W", type=int, default=32)
    ap.add_argument("--Sstep", type=int, default=6)
    a = ap.parse_args(); dev = a.device
    from wmloc.watermarks import load_watermark
    wm = load_watermark("pixelseal", models_dir=ROOT / "models", device=dev); model = wm.model
    me = methods(a.split); idx = np.where(me == "pixelseal")[0]
    if a.diag:     # diagnostic: a few full + small train images
        rg = regime(); idx = np.r_[idx[rg[idx]][:a.limit], idx[~rg[idx]][:max(2, a.limit // 2)]]
    else:
        idx = idx[a.shard::a.nshards]
        if a.limit:
            idx = idx[:a.limit]
    Ss = list(range(512, 616, a.Sstep)); gen = torch.Generator().manual_seed(0)
    maps, info, t0 = [], [], time.time()
    G = masks128() if a.split == "train" else None
    for n_, i in enumerate(idx):
        x = torch.from_numpy(image(a.split, int(i))).permute(2, 0, 1)[None].float().div(255).to(dev)
        hx = hp(x); msg = message(a.split, int(i))
        best = (-9, None, None)
        for S in Ss:
            for T in ((0, 1) if S > 512 else (0,)):
                for L in ((0, 1) if S > 512 else (0,)):
                    p = hp(patterns(model, x, msg, S, T, L, a.K, gen))
                    g = float((hx * p).sum() / (hx.norm() * p.norm() + 1e-12))
                    if g > best[0]:
                        best = (g, (S, T, L), p)
        g, (S, T, L), p = best
        num = box((hx * p).sum(1, keepdim=True), a.W)
        den = torch.sqrt(box((hx ** 2).sum(1, keepdim=True), a.W) * box((p ** 2).sum(1, keepdim=True), a.W)) + 1e-12
        M = F.avg_pool2d(num / den, 4)[0, 0].cpu().numpy()
        maps.append(M.astype(np.float16)); info.append((S, T, L, g))
        if a.diag and G is not None:
            from scipy.stats import rankdata
            y = G[i].ravel(); r = rankdata(M.ravel()); npos = y.sum(); nneg = len(y) - npos
            auc = (r[y > 0].sum() - npos * (npos + 1) / 2) / (npos * nneg) if 0 < npos < len(y) else float("nan")
            print(f"{names(a.split)[i]} full={regime()[i]} wm={y.mean():.2f} S={S} T={T} L={L} g={g:.4f} AUC={auc:.3f} "
                  f"map mean in/out {M.ravel()[y > 0].mean() if y.any() else 0:.4f}/{M.ravel()[y == 0].mean() if (y == 0).any() else 0:.4f} "
                  f"{time.time() - t0:.0f}s", flush=True)
        elif n_ % 10 == 0:
            print(f"{n_}/{len(idx)} S={S} g={g:.4f} {time.time() - t0:.0f}s", flush=True)
    if not a.diag:
        d = ROOT / "work/psinf"; d.mkdir(parents=True, exist_ok=True)
        np.savez(d / f"{a.split}_{a.shard}of{a.nshards}.npz", names=np.array([names(a.split)[i] for i in idx]),
                 maps=np.stack(maps), info=np.array(info))
    print("done", len(idx), f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
