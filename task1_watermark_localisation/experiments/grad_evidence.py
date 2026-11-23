"""G1: decoder BIT-GRADIENT evidence map (PixelSeal, TrustMark) — informed, GT-free, no training, no geometry fit.
For every message bit b: g_b = d logit_b / d image (official decoder, whole frame), r = high-pass residual of the image,
c_b(cell) = window sum of <g_b, r> (4x4-px cells, box window of W cells).  With the KNOWN message signs s_b = +-1:
    z(cell) = sum_b s_b c_b / sqrt(sum_b c_b^2)
Where there is no watermark, c_b is driven by content and is independent of s_b  =>  z ~ N(0, 1) (calibrated);
where the watermark for this message is present, the residual contains it and z > 0.
    python -m experiments.grad_evidence --method pixelseal --diag 12 --device cuda:0        # AUC vs GT on a few train images
    python -m experiments.grad_evidence --method pixelseal --exp G1_ps --device cuda:0       # maps for all images -> work/runs/G1_ps
Run output (as a "run", so nested selection can use it like any source): train.npy / test.npy = sigmoid(z / 2) float16
[n,128,128] for window --win cells; args.json with fold_excluded=true (no label used => usable in every cell).
"""
import argparse
import json
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from src.data import ROOT, image, masks128, message, methods, regime

sys.path.insert(0, str(ROOT / "kit"))
RS = dict(mode="bilinear", align_corners=False, antialias=True)


def forward_fn(wm, method):
    if method == "pixelseal":
        return lambda x: wm.det(F.interpolate(x, size=(256, 256), **RS))[:, 1:].float()
    if method == "trustmark":
        return lambda x: wm.net(F.interpolate(x, size=(wm.resolution,) * 2, **RS) * 2 - 1).float()
    raise ValueError(method)


def hp(x, k=7):
    w = torch.ones(3, 1, k, k, device=x.device) / (k * k)
    return x - F.conv2d(F.pad(x, (k // 2,) * 4, mode="reflect"), w, groups=3)


def box(c, w):
    """box SUM over w x w cells (same size, reflect pad); c: B x 1 x 128 x 128"""
    if w <= 1:
        return c
    return F.avg_pool2d(F.pad(c, (w // 2, w - 1 - w // 2, w // 2, w - 1 - w // 2), mode="reflect"), w, 1) * (w * w)


def evidence(fwd, x, msg, wins=(4, 8, 16), chunk=64, hpk=7):
    """x: 1x3x512x512 in [0,1] on device; msg: 0/1 array. Returns dict win -> z map (128x128, numpy) and bit accuracy."""
    B = len(msg); sgn = torch.from_numpy(2.0 * msg.astype(np.float32) - 1).to(x.device)
    r = hp(x, hpk)
    C = []; lg0 = None
    for s in range(0, B, chunk):
        idx = torch.arange(s, min(B, s + chunk), device=x.device); n = len(idx)
        xb = x.repeat(n, 1, 1, 1).requires_grad_(True)
        with torch.enable_grad():
            lg = fwd(xb)
            g, = torch.autograd.grad(lg[torch.arange(n, device=x.device), idx].sum(), xb)
        if lg0 is None:
            lg0 = lg[0].detach()
        C.append(F.avg_pool2d((g * r).sum(1, keepdim=True), 4) * 16)          # n x 1 x 128 x 128 (cell sums)
    C = torch.cat(C)                                                          # B x 1 x 128 x 128
    out = {}
    for w in wins:
        cw = box(C, w)
        out[w] = ((cw * sgn[:, None, None, None]).sum(0) / torch.sqrt((cw ** 2).sum(0) + 1e-20))[0].cpu().numpy()
    acc = float(((lg0 > 0).float() == (sgn > 0).float()).float().mean())
    return out, acc


def auc(v, y):
    from scipy.stats import rankdata
    y = y.ravel().astype(bool); n1 = y.sum(); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(v.ravel())
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method"); ap.add_argument("--exp", default=""); ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--diag", type=int, default=0); ap.add_argument("--win", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=64); ap.add_argument("--hpk", type=int, default=7)
    a = ap.parse_args(); dev = a.device
    from wmloc.watermarks import load_watermark
    wm = load_watermark(a.method, models_dir=ROOT / "models", device=dev); fwd = forward_fn(wm, a.method)
    for p in (wm.det if a.method == "pixelseal" else wm.net).parameters():
        p.requires_grad_(False)
    me, rg, G = methods(), regime(), masks128()
    mi = np.where(me == a.method)[0]; t0 = time.time()

    def run(split, i, wins):
        x = torch.from_numpy(image(split, int(i))).permute(2, 0, 1)[None].float().div(255).to(dev)
        return evidence(fwd, x, message(split, int(i)), wins=wins, chunk=a.chunk, hpk=a.hpk)

    if a.diag:
        from src.combine2 import src as srcmap
        base = {"pixelseal": ["F2_ps_amp4", "S1_ps_amp4", "snapDF"], "trustmark": ["F1_tm_long", "S1_tm", "snapDF"]}[a.method]
        bm = {b: srcmap(b, a.method)[0] for b in base}
        wins = (2, 4, 8, 16, 32)
        for lab, sel in (("FULL", mi[rg[mi]][:a.diag]), ("SMALL", mi[~rg[mi]][:a.diag])):
            A = {w: [] for w in wins}; Bm = {b: [] for b in base}; accs = []; zin, zout = [], []
            for i in sel:
                z, acc = run("train", i, wins); accs.append(acc); k = int(np.where(mi == i)[0][0])
                for w in wins:
                    A[w].append(auc(z[w], G[i]))
                for b in base:
                    Bm[b].append(auc(bm[b][k], G[i]))
                zin.append(z[8][G[i] > 0].mean() if (G[i] > 0).any() else np.nan); zout.append(z[8][G[i] == 0].mean() if (G[i] == 0).any() else np.nan)
            print(f"{a.method} {lab} n={len(sel)} decode acc {np.mean(accs):.3f} | AUC grad-z " +
                  " ".join(f"w{w}:{np.nanmean(A[w]):.3f}" for w in wins) + " | baselines " +
                  " ".join(f"{b}:{np.nanmean(Bm[b]):.3f}" for b in base) +
                  f" | mean z(w8) inside wm {np.nanmean(zin):.2f} / outside {np.nanmean(zout):.2f} | {time.time() - t0:.0f}s", flush=True)
        return
    out = ROOT / "work/runs" / a.exp; out.mkdir(parents=True, exist_ok=True)
    te = np.where(methods("test") == a.method)[0]
    for split, idx, fn in (("train", mi, "train.npy"), ("test", te, "test.npy")):
        Z = np.zeros((len(idx), 128, 128), np.float32)
        for k, i in enumerate(idx):
            Z[k] = run(split, i, (a.win,))[0][a.win]
            if k % 50 == 0:
                print(f"{split} {k}/{len(idx)} {time.time() - t0:.0f}s", flush=True)
        np.save(out / fn, (1 / (1 + np.exp(-Z / 2))).astype(np.float16))
    (out / "args.json").write_text(json.dumps(dict(vars(a), regime="full", fold_excluded=True, kind="grad_evidence")))
    print("DONE", a.exp, f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
