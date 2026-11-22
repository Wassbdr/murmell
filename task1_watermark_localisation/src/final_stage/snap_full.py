"""snap_full (Sun 10:55): per-COMPONENT shape regularisation of the holes in full-regime cells.  Seen on the train masks:
holes are (a) thresholded smooth noise, (b) ellipses (262 of 832 components have a bbox fill of pi/4), (c) real objects of
the image (vase, cones, suitcase).  For each predicted hole component c (area between AMIN cells and AMAX of the frame):
  SAM  : the SAM segment q with the best IoU with c replaces c if IoU(q, c) >= GATE
  ELL  : else, if the bbox fill of c is within TOL of pi/4, c is replaced by the ellipse inscribed in its bbox
(GATE, AMAX, ellipse on/off, TOL) chosen per method in inner folds; label-free at test time.
    python src/final_stage/snap_full.py --dir DIR     -> DIR/FsnapX_train_scores.npy, DIR/FsnapX_test_bin.npy (chain for chains2.py)
"""
import argparse, glob, itertools, os, sys
import numpy as np
from scipy import ndimage as ndi
sys.path.insert(0, os.getcwd())
from src.combine2 import RE, RT
from src.data import METHODS, ROOT, folds, masks128, methods
from src.eval_cv import THRS
from src.metric import miou_batch, miou_image
from src.postproc_cv import OPTS, pp

ap = argparse.ArgumentParser(); ap.add_argument("--dir", required=True); a = ap.parse_args(); O = a.dir
G = masks128().astype(bool); Fo = folds(); me = methods(); met = methods("test")
CAND = [(o, float(t)) for o in OPTS for t in THRS]
CFG = [(g, amax, ell) for g in (0.3, 0.4, 0.5, 0.6, 0.7, 2.0) for amax in (0.1, 0.3) for ell in (0, 0.05, 0.09)]; CFG = [c for c in CFG if not (c[0] == 2.0 and c[2] == 0)] + [(2.0, 0.1, 0)]


def load_seg(split):
    seg = {}
    for f in sorted(glob.glob(str(ROOT / f"work/sam/{split}_*of4.npz"))):
        d = np.load(f); bits = np.unpackbits(d["bits"], axis=1).reshape(-1, 128, 128).astype(bool)
        for i in np.unique(d["idx"]):
            seg[int(i)] = bits[d["idx"] == i]
    return seg


def regul(mask, Q, cfg):
    gate, amax, ell = cfg; lab, n = ndi.label(~mask); out = mask.copy()
    if n == 0 or n > 12:
        return out
    Qf = Q.reshape(len(Q), -1) if len(Q) else None; ar = Qf.sum(1) if len(Q) else None
    for j in range(1, n + 1):
        c = lab == j; ac = int(c.sum())
        if ac < 12 or ac > amax * 16384:
            continue
        done = False
        if gate <= 1 and Qf is not None:
            inter = (Qf & c.ravel()).sum(1); iou = inter / np.maximum(1, ar + ac - inter); k = int(np.argmax(iou))
            if iou[k] >= gate:
                out[c] = True; out[Q[k]] = False; done = True
        if not done and ell > 0:
            ys, xs = np.where(c); y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1; h, w = y1 - y0, x1 - x0
            if h >= 4 and w >= 4 and abs(ac / (h * w) - np.pi / 4) <= ell:
                yy, xx = np.mgrid[y0:y1, x0:x1]; e = ((yy + 0.5 - (y0 + y1) / 2) / (h / 2)) ** 2 + ((xx + 0.5 - (x0 + x1) / 2) / (w / 2)) ** 2 <= 1.0
                out[c] = True; out[y0:y1, x0:x1][e] = False
    return out


def std_masks(m, idx, tidx, sel, tsel):
    """standard full-cell chain of B21: PixelSeal = D7_ps_Fsub with nested (pp, thr); TrustMark = T02; StegaStamp = T03"""
    if m == "pixelseal":
        X = np.load(ROOT / "work/runs/D7_ps_Fsub/train.npy").astype(np.float32)[sel]; Xt = np.load(ROOT / "work/runs/D7_ps_Fsub/test.npy").astype(np.float32)[tsel]
        S = np.zeros((len(idx), len(CAND))); M = np.zeros((len(CAND), len(idx), 128, 128), bool)
        for c, (o, t) in enumerate(CAND):
            M[c] = np.stack([pp(x >= t, o) for x in X]); S[:, c] = miou_batch(M[c], G[idx])
        Pm = np.zeros((len(idx), 128, 128), bool)
        for k in range(5):
            Pm[Fo[idx] == k] = M[int(np.argmax(S[Fo[idx] != k].mean(0)))][Fo[idx] == k]
        o, t = CAND[int(np.argmax(S.mean(0)))]
        return Pm, np.stack([pp(x >= t, o) for x in Xt])
    n = "T02" if m == "trustmark" else "T03"
    P = (np.load("work/hunt/T02_oof.npz") if n == "T02" else np.load(f"{O}/T03_oof.npz"))["P"].astype(bool)
    return P[idx], np.load(f"work/test/{n}_bin.npy").astype(bool)[tidx]


seg = load_seg("train"); segt = load_seg("test")
s_out = np.load("work/nested/B21.npy").copy(); T_out = np.load("work/test/B21_bin.npy").astype(bool).copy()
for m in METHODS:
    mi = np.where(me == m)[0]; ti = np.where(met == m)[0]; sel = RT[mi]; idx = mi[sel]; tsel = RE[ti]; tidx = ti[tsel]
    Pm, Tm = std_masks(m, idx, tidx, sel, tsel); s0 = miou_batch(Pm, G[idx])
    S = np.stack([np.array([miou_image(regul(Pm[j], seg.get(int(i), np.zeros((0, 128, 128), bool)), cfg), G[i]) for j, i in enumerate(idx)]) for cfg in CFG], 1)
    out = np.zeros(len(idx)); ch = []
    for k in range(5):
        j = int(np.argmax(S[Fo[idx] != k].mean(0))); out[Fo[idx] == k] = S[Fo[idx] == k, j]; ch.append(CFG[j])
    d = np.array([out[Fo[idx] == k].mean() - s0[Fo[idx] == k].mean() for k in range(5)]); jall = int(np.argmax(S.mean(0)))
    only_sam = max((S[:, j].mean(), CFG[j]) for j in range(len(CFG)) if CFG[j][2] == 0); only_ell = max((S[:, j].mean(), CFG[j]) for j in range(len(CFG)) if CFG[j][0] == 2.0)
    print(f"{m}|F n={len(idx)}: standard {s0.mean():.4f} -> regularised (nested) {out.mean():.4f} ({out.mean() - s0.mean():+.4f}; folds {np.round(d, 4).tolist()}; {int((d > 0).sum())}/5) | per fold {ch} | test {CFG[jall]} | in-sample: SAM only {only_sam[0]:.4f} {only_sam[1]}, ellipse only {only_ell[0]:.4f} {only_ell[1]} | images changed {int((S[:, jall] != s0).sum())}")
    s_out[idx] = out; T_out[tidx] = np.stack([regul(Tm[j], segt.get(int(i), np.zeros((0, 128, 128), bool)), CFG[jall]) for j, i in enumerate(tidx)])
np.save(f"{O}/FsnapX_train_scores.npy", s_out); np.save(f"{O}/FsnapX_test_bin.npy", T_out); print("written FsnapX chain")
