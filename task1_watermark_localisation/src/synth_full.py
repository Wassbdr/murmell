"""Synthetic FULL-regime pool for one method (official kit embedder, faithful to the measured geometry).
Item k: base frame b (a real train/test image of ANOTHER method: carries no watermark of this method),
canvas = reflect-pad of b to S = round(512*s), s ~ U[1.0, 1.2], frame offset (L, T) ~ U in the canvas,
Yk = crop(Emb(canvas, random msg), frame) as uint8. Holes are composited later, on the fly, by the trainer:
y = where(hole, base, Yk). No real label is used here.
    python -m src.synth_full --method trustmark --n 6000 --seed 0 --device cuda:0
Writes work/synth_full/<method>_s<seed>.u8 (memmap N x 512 x 512 x 3) and <method>_s<seed>.npz (base names, s, L, T, msgs).
"""
import argparse
import sys
import time

import numpy as np

from src.data import ROOT, image, meta

sys.path.insert(0, str(ROOT / "kit"))


def bases_for(method, splits=("train", "test")):
    out = []
    for split in splits:
        out += [(split, m["image_name"]) for m in meta(split) if m["method"] != method]
    # HARD ASSERTION: the base pool of method M excludes every image (train AND test) of method M
    own = {m["image_name"] for sp in ("train", "test") for m in meta(sp) if m["method"] == method}
    assert own and not (own & {n for _, n in out}), f"base pool of {method} contains {method} images"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method"); ap.add_argument("--n", type=int, default=6000); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--bases", default="all", choices=["all", "train"]); ap.add_argument("--outdir", default="synth_full"); ap.add_argument("--smin", type=float, default=1.0); ap.add_argument("--smax", type=float, default=1.2)
    a = ap.parse_args()
    from wmloc.watermarks import load_watermark
    wm = load_watermark(a.method, models_dir=ROOT / "models", device=a.device)
    rng = np.random.RandomState(1000 * a.seed + {"pixelseal": 0, "trustmark": 1, "stegastamp": 2}[a.method])
    bases = bases_for(a.method, ("train", "test") if a.bases == "all" else ("train",))
    out = ROOT / "work" / a.outdir; out.mkdir(parents=True, exist_ok=True)
    stem = out / f"{a.method}_s{a.seed}"
    Y = np.lib.format.open_memmap(str(stem) + ".npy", mode="w+", dtype=np.uint8, shape=(a.n, 512, 512, 3))
    rec = dict(split=[], name=[], s=np.zeros(a.n, np.float32), L=np.zeros(a.n, np.int32), T=np.zeros(a.n, np.int32),
               msg=np.zeros((a.n, wm.nbits), np.uint8))
    cache = {}; t0 = time.time()
    for k in range(a.n):
        sp, nm = bases[rng.randint(len(bases))]
        if nm not in cache:
            cache[nm] = image(sp, nm)
        b = cache[nm]
        s = rng.uniform(a.smin, a.smax); S = int(round(512 * s)); P = S - 512
        L, T = rng.randint(P + 1), rng.randint(P + 1)
        can = np.pad(b, ((T, P - T), (L, P - L), (0, 0)), mode="reflect") if P else b
        bits = rng.randint(0, 2, wm.nbits).astype(np.uint8)
        Y[k] = wm.embed(can, bits)[T:T + 512, L:L + 512]
        rec["split"].append(sp); rec["name"].append(nm); rec["s"][k] = s; rec["L"][k] = L; rec["T"][k] = T; rec["msg"][k] = bits
        if k % 500 == 0:
            print(f"{a.method} {k}/{a.n} {time.time() - t0:.0f}s", flush=True)
    Y.flush()
    np.savez(str(stem) + ".npz", split=np.array(rec["split"]), name=np.array(rec["name"]), s=rec["s"], L=rec["L"],
             T=rec["T"], msg=rec["msg"])
    print("done", stem, f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
