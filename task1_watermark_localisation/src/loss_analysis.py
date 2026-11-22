"""Where does a full_nested candidate lose its points?  (human mission 23:18)
Rebuilds the OUT-OF-FOLD predicted masks of candidate <name> from work/nested/<name>.json (per cell x fold choice of
combo / post-proc / threshold + decoder-ranked box override), checks they reproduce work/nested/<name>.npy, then breaks
the loss (1 - mIoU, share of the total) down by method, true regime, inferred edit type, patch size, failure class and
regime-routing errors.  GT is used for the analysis only.
    python -m src.loss_analysis --name N23 [--save 1]
--save writes work/nested/<name>_oofpred.npy (bool [510,128,128]) into an EXISTING file only (inode quota).
"""
import argparse
import json

import numpy as np
from scipy import ndimage as ndi

from src.apply_boxes import cells, load_boxes
from src.combine2 import RT, src
from src.data import METHODS, ROOT, folds, masks128, methods, names, regime
from src.metric import miou_image
from src.postproc_cv import pp


def rebuild(name):
    rep = json.loads((ROOT / "work/nested" / f"{name}.json").read_text())
    G, Fo, me = masks128(), folds(), methods(); ix = {n: i for i, n in enumerate(names())}
    ov = {}
    for spec in rep["boxes"]:
        m, mp = spec.split(":")
        for n, r in load_boxes(m, mp, "train").items():
            y0, y1, x0, x1 = r["box"]; h, w = y1 - y0, x1 - x0
            if r["acc"] < rep["thr_box"] or (m == "trustmark" and max(h, w) > 2.0 * min(h, w)):
                continue
            if ix[n] not in ov or r["acc"] >= ov[ix[n]][1]:
                ov[ix[n]] = (cells(r["box"]), r["acc"])
    P = np.zeros((510, 128, 128), bool); boxed = np.zeros(510, bool)
    for m in METHODS:
        mi = np.where(me == m)[0]
        for reg in ("F", "S"):
            for k in range(5):
                cb, o, t = rep["choices"][f"{m}|{reg}|{k}"]
                X = sum(src(c, m)[0] for c in cb) / len(cb)
                for j, i in enumerate(mi):
                    if RT[i] == (reg == "F") and Fo[i] == k:
                        P[i] = ov[i][0].astype(bool) if i in ov else pp(X[j] >= t, o); boxed[i] = i in ov
    s = np.array([miou_image(P[i], G[i]) for i in range(510)])
    ref = np.load(ROOT / "work/nested" / f"{name}.npy")
    print(f"rebuilt OOF masks: mean {s.mean():.6f} | stored nested per-image mean {ref.mean():.6f} | max abs diff {np.abs(s - ref).max():.2e}")
    return P, s, boxed, rep


def iou(a, b):
    u = (a | b).sum()
    return 1.0 if u == 0 else (a & b).sum() / u


def tab(title, keys, s, min_n=1):
    tot = (1 - s).sum()
    print(f"\n{title}\n{'group':34s} {'n':>4s} {'mIoU':>6s} {'loss share':>10s} {'pts lost':>8s}")
    for k in sorted(set(keys)):
        m = keys == k
        if m.sum() >= min_n:
            print(f"{k:34s} {m.sum():4d} {s[m].mean():6.3f} {100 * (1 - s[m]).sum() / tot:9.1f}% {(1 - s[m]).sum() / len(s):8.4f}")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--name", default="N23"); ap.add_argument("--save", type=int, default=0)
    a = ap.parse_args()
    P, s, boxed, rep = rebuild(a.name)
    G, me, rg = masks128().astype(bool), methods(), regime()
    if a.save:
        f = ROOT / "work/nested" / f"{a.name}_oofpred.npy"
        if f.exists():
            np.save(f, P)
    short = {"pixelseal": "PS", "trustmark": "TM", "stegastamp": "SS"}
    area = G.mean((1, 2)); iw = np.array([iou(P[i], G[i]) for i in range(510)]); ic = np.array([iou(~P[i], ~G[i]) for i in range(510)])
    # inferred edit type
    typ = np.empty(510, object); size = np.empty(510, object); asp = np.ones(510)
    for i in range(510):
        g = G[i] if not rg[i] else ~G[i]                  # the "object": patch (small regime) or hole (full regime)
        if not g.any():
            typ[i] = "full: no hole" if rg[i] else "small: empty"; size[i] = "-"; continue
        ys, xs = np.where(g); h, w = np.ptp(ys) + 1, np.ptp(xs) + 1; fill = g.sum() / (h * w); asp[i] = max(h, w) / min(h, w)
        n = ndi.label(g)[1]
        if not rg[i]:
            typ[i] = "small: rectangle (outpainting)" if fill >= 0.95 else "small: object (transplant)"
        else:
            typ[i] = "full: 1 hole" if n == 1 else "full: 2+ holes"
        fr = g.mean(); size[i] = "<2%" if fr < 0.02 else "2-5%" if fr < 0.05 else "5-10%" if fr < 0.10 else "10-20%" if fr < 0.2 else ">20%"
    M = np.array([short[m] for m in me]); R = np.where(rg, "full", "small")
    print(f"\ncandidate {a.name}: nested CV {rep['cv_foldmean']:.6f}; mean loss {1 - s.mean():.4f}")
    tab("A. method x TRUE regime", np.array([f"{m} {r}" for m, r in zip(M, R)]), s)
    tab("B. inferred edit type (all methods)", typ, s)
    tab("C. method x edit type", np.array([f"{m} {t}" for m, t in zip(M, typ)]), s)
    tab("D. method x regime x size of patch/hole (fraction of frame)", np.array([f"{m} {r} {z}" for m, r, z in zip(M, R, size)]), s)
    # failure classes
    fc = np.empty(510, object)
    for i in range(510):
        if rg[i]:      # the hard class is the clean hole
            fc[i] = "full: hole missed (IoU_clean<.1)" if ic[i] < 0.1 else "full: hole partial (.1-.6)" if ic[i] < 0.6 else "full: hole found (>.6)"
            if not (~G[i]).any():
                fc[i] = "full: no hole"
        else:
            if not P[i].any():
                fc[i] = "small: nothing predicted"
            elif iw[i] < 0.1:
                fc[i] = "small: wrong place (IoU_wm<.1)"
            elif iw[i] < 0.6:
                fc[i] = "small: partial (.1-.6)"
            else:
                fc[i] = "small: found (>.6)"
    tab("E. failure class x method", np.array([f"{m} {c}" for m, c in zip(M, fc)]), s)
    # regime routing
    wrong = RT != rg
    tab("F. regime routing (B0 classifier) x method x true regime",
        np.array([f"{m} {r} {'MISROUTED' if w else 'ok'}" for m, r, w in zip(M, R, wrong)]), s)
    cost = 0.0
    for m in set(M):
        for r in ("full", "small"):
            c = (M == m) & (R == r)
            if (c & wrong).any():
                cost += ((s[c & ~wrong].mean() - s[c & wrong]).sum()) / 510
    print(f"regime errors: {wrong.sum()} images; cost if they scored like well-routed images of the same cell: {cost:.4f}")
    # boxes
    sm = ~rg
    tab("G. small regime: decoder-ranked box accepted (acc>=.8) or not, x method x type",
        np.array([f"{m} {t[7:16]} {'BOX' if b else 'map'}" for m, t, b in zip(M[sm], typ[sm], boxed[sm])]), s[sm])
    # false-positive vs miss in small regime (areas)
    for m in ("PS", "TM", "SS"):
        c = sm & (M == m)
        print(f"{m} small: GT area {100 * area[c].mean():.1f}% | predicted area {100 * P[c].mean():.1f}% | IoU_wm {iw[c].mean():.3f} IoU_clean {ic[c].mean():.3f}")
        c = rg & (M == m)
        print(f"{m} full : GT hole {100 * (1 - area[c]).mean():.1f}% | predicted hole {100 * (1 - P[c].mean()):.1f}% | IoU_wm {iw[c].mean():.3f} IoU_clean {ic[c].mean():.3f}")
    np.save(ROOT / "work/nested" / "FN7.npy", np.stack([s, iw, ic]))      # scratch (existing file): per-image score, IoU_wm, IoU_clean


if __name__ == "__main__":
    main()
