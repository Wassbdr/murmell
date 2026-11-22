"""Synthetic SMALL-regime pool v2, following the OFFICIAL participants document (21 Nov ~20:50):
"each image was watermarked on a sub-rectangle of the original image, as a whole image at its own size (aspect <= 2:1
TrustMark, <= 16:1 PixelSeal/StegaStamp); then one edit: outpainting (mask = the rectangle) | watermarked object
transplanted into a clean image (mask = the object) | object removed/modified/added (holes, = full regime)".
Differences vs src/synth_small.py (kept unchanged):
  * the watermarked content is embedded at its OWN resolution f x (frame size), f log-uniform in [FMIN, FMAX], then
    resized to frame size with a random kernel (real SS decode-acc tail: real .887 / <0.9 in 32 % vs synth f=1 .985 / 0 %,
    f=0.5 .928 / 25 %, f=0.7 .957 / 14 %);
  * rectangle aspect log-uniform up to 16:1 for PS/SS (train: 19/41 PS and 18/43 SS interior rects have aspect > 3), 2:1 TM;
  * 'transplant' items: a whole clean base image (other method) is watermarked at its own size (512 f), a region is cut
    with an object shape (full-regime hole components, as hole_library: no small-regime GT shape used) and pasted at a
    random scale into a different clean base.  Mask stored per item (128x128).
    python -m src.synth_small2 --method stegastamp --n 6000 --seed 0 --device cuda:0
Writes work/synth_small2/<method>_s<seed>.npy (N,512,512,3 uint8), _m.npy (N,128,128 uint8), .npz (split, name, kind, f, msg).
"""
import argparse
import sys
import time

import numpy as np
from PIL import Image

from src.data import ROOT, image
from src.synth_full import bases_for

sys.path.insert(0, str(ROOT / "kit"))
FMIN, FMAX = 0.35, 2.0
KER = [Image.Resampling.BILINEAR, Image.Resampling.BICUBIC, Image.Resampling.LANCZOS]


def to128(M):
    return (M.reshape(128, 4, 128, 4).mean((1, 3)) >= 0.5).astype(np.uint8)


def rs(a, w, h, rng):
    return np.asarray(Image.fromarray(np.ascontiguousarray(a)).resize((int(w), int(h)), KER[rng.randint(3)]))


def sample_wh(rng, method, amin=0.002, amax=0.45):
    asp = 2.0 if method == "trustmark" else 16.0
    for _ in range(200):
        a = np.exp(rng.uniform(np.log(amin), np.log(amax))) * 512 * 512
        ar = np.exp(rng.uniform(0, np.log(asp))) ** (1 if rng.rand() < 0.5 else -1)
        w, h = int(round(np.sqrt(a * ar))), int(round(np.sqrt(a / ar)))
        if 8 <= w <= 512 and 8 <= h <= 512:
            return w, h
    return 64, 64


def embed_at(wm, x, bits, f, rng):
    """embed x (h,w,3) as a whole image at its own resolution f*(h,w) (min side 16), resize back."""
    h, w = x.shape[:2]; H, W = max(16, int(round(h * f))), max(16, int(round(w * f)))
    e = wm.embed(rs(x, W, H, rng) if (H, W) != (h, w) else np.ascontiguousarray(x), bits)
    return rs(e, w, h, rng) if (H, W) != (h, w) else e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method"); ap.add_argument("--n", type=int, default=6000); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0"); ap.add_argument("--p_obj", type=float, default=0.5)
    ap.add_argument("--outdir", default="synth_small2")
    ap.add_argument("--p_f1", type=float, default=0.0)   # v3: probability of f = 1 exactly (real decode acc median .975)
    a = ap.parse_args()
    from wmloc.watermarks import load_watermark
    from src.train_full import hole_library
    wm = load_watermark(a.method, models_dir=ROOT / "models", device=a.device)
    rng = np.random.RandomState(9000 + 1000 * a.seed + {"pixelseal": 0, "trustmark": 1, "stegastamp": 2}[a.method])
    bases = bases_for(a.method, ("train", "test")); lib = hole_library(); print("shapes", len(lib), flush=True)
    out = ROOT / "work" / a.outdir; out.mkdir(parents=True, exist_ok=True); stem = out / f"{a.method}_s{a.seed}"
    Y = np.lib.format.open_memmap(str(stem) + ".npy", mode="w+", dtype=np.uint8, shape=(a.n, 512, 512, 3))
    Mk = np.lib.format.open_memmap(str(stem) + "_m.npy", mode="w+", dtype=np.uint8, shape=(a.n, 128, 128))
    split, name, kind, fs = [], [], [], []; msg = np.zeros((a.n, wm.nbits), np.uint8)
    cache = {}; t0 = time.time(); fails = 0

    def get(k):
        sp, nm = bases[k]
        if nm not in cache:
            if len(cache) > 600:
                cache.clear()
            cache[nm] = image(sp, nm)
        return cache[nm]

    for k in range(a.n):
        kb = rng.randint(len(bases)); b = get(kb).copy(); bits = rng.randint(0, 2, wm.nbits).astype(np.uint8)
        f = 1.0 if rng.rand() < a.p_f1 else float(np.exp(rng.uniform(np.log(FMIN), np.log(FMAX)))); M = np.zeros((512, 512), bool)
        obj = rng.rand() < a.p_obj and len(lib) > 0
        try:
            if not obj:                                   # outpainting: watermarked rectangle, rest = (other) content
                w, h = sample_wh(rng, a.method); x0, y0 = rng.randint(0, 513 - w), rng.randint(0, 513 - h)
                b[y0:y0 + h, x0:x0 + w] = embed_at(wm, b[y0:y0 + h, x0:x0 + w], bits, f, rng); M[y0:y0 + h, x0:x0 + w] = True
            else:                                         # transplant: object cut from a whole watermarked image
                src = get(rng.randint(len(bases))); S = int(np.clip(round(512 * f), 128, 1024))
                e = rs(wm.embed(rs(src, S, S, rng), bits), 512, 512, rng) if S != 512 else wm.embed(src, bits)
                o = lib[rng.randint(len(lib))]
                if rng.rand() < 0.5:
                    o = o[:, ::-1]
                w, h = sample_wh(rng, "trustmark", 0.005, 0.45)        # object bbox in the target (shape decides aspect)
                g = float(np.exp(rng.uniform(np.log(0.5), np.log(2.0))))   # paste scale
                sw, sh = int(np.clip(w / g, 8, 512)), int(np.clip(h / g, 8, 512))
                sx, sy = rng.randint(0, 513 - sw), rng.randint(0, 513 - sh)
                patch = rs(e[sy:sy + sh, sx:sx + sw], w, h, rng)
                om = np.asarray(Image.fromarray((np.ascontiguousarray(o) * 255).astype(np.uint8)).resize((w, h), Image.Resampling.BILINEAR)) > 127
                if om.sum() < 16:
                    om[:] = True
                x0, y0 = rng.randint(0, 513 - w), rng.randint(0, 513 - h)
                reg = b[y0:y0 + h, x0:x0 + w]; reg[om] = patch[om]; M[y0:y0 + h, x0:x0 + w] = om
        except Exception as ex:
            fails += 1
            if fails < 5:
                print("fail", repr(ex)[:200], flush=True)
            w = h = 64; x0, y0 = rng.randint(0, 448), rng.randint(0, 448); M[:] = False
            b[y0:y0 + h, x0:x0 + w] = wm.embed(np.ascontiguousarray(b[y0:y0 + h, x0:x0 + w]), bits); M[y0:y0 + h, x0:x0 + w] = True
            obj = False
        Y[k] = b; Mk[k] = to128(M); split.append(bases[kb][0]); name.append(bases[kb][1]); kind.append(int(obj)); fs.append(f); msg[k] = bits
        if k % 500 == 0:
            print(f"{a.method} {k}/{a.n} {time.time() - t0:.0f}s fails {fails}", flush=True)
    Y.flush(); Mk.flush()
    np.savez(str(stem) + ".npz", split=np.array(split), name=np.array(name), kind=np.array(kind), f=np.array(fs), msg=msg)
    print("done", stem, f"{time.time() - t0:.0f}s fails {fails}", flush=True)


if __name__ == "__main__":
    main()
