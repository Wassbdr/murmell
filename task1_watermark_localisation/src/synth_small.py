"""Synthetic SMALL-regime pool for one method: crop-embed-paste of one rectangle (official kit embedder).
Item k: base b (real image of ANOTHER method), rect R = (x0, y0, w, h) with area fraction log-uniform in
[0.002, 0.45] and aspect in [1/3, 3] (TrustMark: [1/2, 2], as the kit says the dataset only uses aspect <= 2 for TM),
Yk = b with b[R] replaced by Emb(b[R], random msg). The patch SHAPE inside R (full rect or an object shape) is drawn
on the fly by the trainer: y = where(shape, Yk, b). No real label is used here.
    python -m src.synth_small --method trustmark --n 6000 --seed 0 --device cuda:0
Writes work/synth_small/<method>_s<seed>.npy (N x 512 x 512 x 3 uint8) and .npz (split, name, rect, msg).
"""
import argparse
import sys
import time

import numpy as np

from src.data import ROOT, image
from src.synth_full import bases_for

sys.path.insert(0, str(ROOT / "kit"))


def sample_rect(rng, method):
    amax = 2.0 if method == "trustmark" else 3.0
    for _ in range(100):
        a = np.exp(rng.uniform(np.log(0.002), np.log(0.45))) * 512 * 512
        ar = np.exp(rng.uniform(-np.log(amax), np.log(amax)))
        w, h = int(round(np.sqrt(a * ar))), int(round(np.sqrt(a / ar)))
        if 12 <= w <= 512 and 12 <= h <= 512:
            return rng.randint(0, 512 - w + 1), rng.randint(0, 512 - h + 1), w, h
    return 200, 200, 64, 64


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method"); ap.add_argument("--n", type=int, default=6000); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--bases", default="all", choices=["all", "train"]); ap.add_argument("--outdir", default="synth_small")
    a = ap.parse_args()
    from wmloc.watermarks import load_watermark
    wm = load_watermark(a.method, models_dir=ROOT / "models", device=a.device)
    rng = np.random.RandomState(5000 + 1000 * a.seed + {"pixelseal": 0, "trustmark": 1, "stegastamp": 2}[a.method])
    bases = bases_for(a.method, ("train", "test") if a.bases == "all" else ("train",))
    out = ROOT / "work" / a.outdir; out.mkdir(parents=True, exist_ok=True)
    stem = out / f"{a.method}_s{a.seed}"
    Y = np.lib.format.open_memmap(str(stem) + ".npy", mode="w+", dtype=np.uint8, shape=(a.n, 512, 512, 3))
    split, name = [], []; rect = np.zeros((a.n, 4), np.int32); msg = np.zeros((a.n, wm.nbits), np.uint8)
    cache = {}; t0 = time.time(); fails = 0
    for k in range(a.n):
        sp, nm = bases[rng.randint(len(bases))]
        if nm not in cache:
            cache[nm] = image(sp, nm)
        b = cache[nm].copy()
        bits = rng.randint(0, 2, wm.nbits).astype(np.uint8)
        x0, y0, w, h = sample_rect(rng, a.method)
        try:
            b[y0:y0 + h, x0:x0 + w] = wm.embed(np.ascontiguousarray(b[y0:y0 + h, x0:x0 + w]), bits)
        except Exception as e:                       # tiny crops some embedders refuse: fall back to a 64 px square
            fails += 1; x0, y0, w, h = rng.randint(0, 448), rng.randint(0, 448), 64, 64
            b[y0:y0 + h, x0:x0 + w] = wm.embed(np.ascontiguousarray(b[y0:y0 + h, x0:x0 + w]), bits)
            if fails < 5:
                print("embed failure", repr(e)[:200], flush=True)
        Y[k] = b; split.append(sp); name.append(nm); rect[k] = (x0, y0, w, h); msg[k] = bits
        if k % 500 == 0:
            print(f"{a.method} {k}/{a.n} {time.time() - t0:.0f}s fails {fails}", flush=True)
    Y.flush()
    np.savez(str(stem) + ".npz", split=np.array(split), name=np.array(name), rect=rect, msg=msg)
    print("done", stem, f"{time.time() - t0:.0f}s fails {fails}", flush=True)


if __name__ == "__main__":
    main()
