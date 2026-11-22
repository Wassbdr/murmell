"""<run>_sam*: SAM-snapped versions of a run's probability maps.  Label-free post-processing (segments of src/sam_segments.py,
computed from the image alone), so the new run keeps the args / regime / fold-cleanliness of its base and is judged like any
other candidate by src.full_nested (choices in inner folds).
  sam  : p'(c) = mean of p over the SMALLEST SAM segment containing cell c, among segments of AMIN cells .. AMAX of the frame;
         cells in no such segment keep p (large background regions, where a watermarked rectangle may end anywhere, are untouched)
  samb : 0.5 * p + 0.5 * p'                                   (soft version)
  samc : like sam, but a segment is snapped only when it is DECISIVE: >= 80 % or <= 20 % of its cells have p >= its base
         median-free cut 0.5 (a segment straddling the border of a rectangle is left alone)
    python -m src.sam_refine --exp D5_ps_noamp_tta --variants sam samb samc
    python -m src.sam_refine --diag 1 --exp D5_ps_noamp_tta FTxt_R_DT2_ps_amp4       # train-only diagnostic table, writes nothing
"""
import argparse
import json

import numpy as np

from src.data import METHODS, ROOT, masks128, methods

AMIN, AMAX = 3, 0.25


def load_segs(split, nshards=4, partial=False):
    """image index -> bool [m, 128, 128] cell masks sorted by decreasing area (AMIN cells .. 60 % of the frame)"""
    idx, bits = [], []
    for k in range(nshards):
        f = ROOT / "work/sam" / f"{split}_{k}of{nshards}.npz"
        if partial and f.stat().st_size == 0:
            continue
        d = np.load(f); idx.append(d["idx"]); bits.append(d["bits"])
    idx = np.concatenate(idx); bits = np.concatenate(bits); out = {}
    order = np.argsort(idx, kind="stable"); idx, bits = idx[order], bits[order]
    cut = np.flatnonzero(np.diff(idx)) + 1
    for i, b in zip(idx[np.r_[0, cut]], np.split(bits, cut)):
        m = np.unpackbits(b, axis=1).reshape(-1, 128, 128).astype(bool); n = m.reshape(len(m), -1).sum(1)
        keep = (n >= AMIN) & (n <= 0.6 * 16384); m, n = m[keep], n[keep]
        out[int(i)] = m[np.argsort(-n, kind="stable")]
    return out


def refine(p, segs, variant):
    """variants (all label-free; AMAX = 25 % of the frame unless the name ends with 50):
    sam  smallest containing segment's mean | samb 0.5 p + 0.5 sam | samc only decisive segments (>= 80 % or <= 20 % of cells >= .5)
    samL largest containing segment's mean  | samd mean of the containing segment with the largest |mean - 0.5|
    samo sam + cells in no segment pushed half-way to the image's majority side (halo of a blurred contour)"""
    p = p.astype(np.float32)
    if segs is None or len(segs) == 0:
        return p
    amax = 0.5 if variant.endswith("50") else AMAX; v_ = variant[:-2] if variant.endswith("50") else variant
    segs = [s for s in segs if s.sum() <= amax * 16384]
    out = p.copy()
    if v_ == "samS":                                    # the ONE segment with the best soft IoU against p (transplanted object = one segment)
        tot = float(p.sum()); best, bs = 0.0, None
        for s in segs:
            i_ = float(p[s].sum()); j_ = i_ / (float(s.sum()) + tot - i_ + 1e-6)
            if j_ > best:
                best, bs = j_, s
        if bs is None:
            return p
        out = 0.25 * p; out[bs] = 0.5 + 0.5 * float(p[bs].mean())
        return out
    if v_ == "samd":
        best = np.zeros(p.shape, np.float32)
        for s in segs:
            mu = float(p[s].mean()); dcs = abs(mu - 0.5); u = s & (dcs > best); out[u] = mu; best[u] = dcs
        return out
    cov = np.zeros(p.shape, bool)
    for s in (segs[::-1] if v_ == "samL" else segs):    # default large -> small: the smallest segment containing a cell writes last
        v = p[s]
        if v_ == "samc":
            f = (v >= 0.5).mean()
            if 0.2 < f < 0.8:
                continue
        out[s] = v.mean(); cov |= s
    if v_ == "samb":
        return 0.5 * (p + out)
    if v_ == "samo":
        r = ~cov; out[r] = (1 - 0.5 * (1 - p[r])) if p.mean() > 0.5 else 0.5 * p[r]
    return out


def refine_set(P, ids, segs, variant):
    return np.stack([refine(P[j], segs.get(int(i)), variant) for j, i in enumerate(ids)])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--exp", nargs="+"); ap.add_argument("--variants", nargs="+", default=["sam", "samb", "samc"])
    ap.add_argument("--diag", type=int, default=0)
    a = ap.parse_args()
    me, mt = methods(), methods("test"); St = load_segs("train", partial=bool(a.diag)); Se = None if a.diag else load_segs("test")
    print("segments per train image: median", int(np.median([len(St.get(i, [])) for i in range(510)])), "| images without segment", sum(i not in St for i in range(510)), flush=True)
    if a.diag:
        from src.combine2 import RT
        from src.eval_cv import THRS
        from src.metric import miou_batch
        G = masks128().astype(bool)
        rect = np.zeros(510, bool)                  # GT type: one filled rectangle (small regime) / rectangular hole (full regime)
        for i in range(510):
            g = G[i] if not RT[i] else ~G[i]
            if g.any():
                ys, xs = np.where(g); rect[i] = g.sum() >= 0.97 * (np.ptp(ys) + 1) * (np.ptp(xs) + 1)
    for e in a.exp:
        d = ROOT / "work/runs" / e; A = json.loads((d / "args.json").read_text()); m = A["method"]
        mi = np.where(me == m)[0]; te = np.where(mt == m)[0]; P = np.load(d / "train.npy").astype(np.float32)
        if a.diag:
            row = {}; has = np.array([i in St for i in range(510)])
            for v in ["raw"] + a.variants:
                Q = P if v == "raw" else refine_set(P, mi, St, v)
                for r, nm in ((True, "F"), (False, "S")):
                    for ty, tn in ((None, "all"), (True, "rect"), (False, "obj")):
                        sel = ((RT[mi] == r) if ty is None else ((RT[mi] == r) & (rect[mi] == ty))) & has[mi]
                        if sel.sum() < 3:
                            continue
                        sc = [miou_batch(Q[sel] >= t, G[mi[sel]]).mean() for t in THRS]
                        row.setdefault(f"{nm}:{tn}(n={int(sel.sum())})", []).append(f"{v} {max(sc):.4f}@{THRS[int(np.argmax(sc))]:.2f}")
            print(f"DIAG {e}"); [print("   ", k, " | ".join(x)) for k, x in row.items()]
            continue
        T = np.load(d / "test.npy").astype(np.float32)
        for v in a.variants:
            o = ROOT / "work/runs" / f"{e}_{v}"; o.mkdir(parents=True, exist_ok=True)
            with open(o / "train.npy", "wb") as f:
                np.save(f, refine_set(P, mi, St, v).astype(np.float16))
            with open(o / "test.npy", "wb") as f:
                np.save(f, refine_set(T, te, Se, v).astype(np.float16))
            (o / "args.json").write_text(json.dumps(dict(A, sam_refine=v, sam_base=e, sam_amin=AMIN, sam_amax=AMAX)))
            print(f"WROTE {e}_{v}", flush=True)


if __name__ == "__main__":
    main()
