"""LEARNED choice of the SAM segment (Sun 07:15).  src.sam_ens takes the automatic segment with the best soft IoU against
the ensemble map E and a fixed threshold on that soft IoU to decide object / rectangle.  In-sample, the union of automatic and
prompted masks (src.sam_prompt) contains a better mask (oracle on PixelSeal objects .974 vs .955) but the soft IoU picks it
less often.  Here a LightGBM regressor predicts the mIoU of each candidate mask from label-free features:
  candidates per image = 6 best automatic segments by soft IoU + the prompted masks + "blob" (largest component of
                         E >= 0.5 max E = no snapping)
  features = soft IoU, rank, gap to the best, mean E inside / in a 3-cell ring / contrast, share of the mass of E, area,
             bbox fill, aspect, IoU with the blob, SAM's own score, source id, image-level stats of E, method
OUT-OF-FOLD like src.finetune_real: the map of a train image of fold k comes from a regressor fitted on the small-regime
images of the 4 other folds (true regime label = training target side only); test = regressor fitted on all 5 folds.
Fixed hyper-parameters (those of src/selector.py), nothing tuned.  Output runs <base>_samL (regime "small"):
snap(base, chosen segment), or the base map unchanged when "blob" is chosen.
    python -m src.sam_rank            # prints out-of-fold scores of the chosen masks, writes the runs
"""
import json

import numpy as np
from scipy import ndimage as ndi

from src.data import METHODS, ROOT, folds, masks128, methods, regime
from src.metric import miou_image
from src.sam_ens import BASES, ENS, save, snap
from src.sam_refine import load_segs

import os
MODE = os.environ.get("SAMRANK_MODE", "L")      # "L" (07:12): target of the no-snap candidate = mIoU of the blob of E.
# "M" (07:36): PixelSeal rectangles were snapped half of the time (T01: rectangles .662 vs .709 for S11, objects .893 vs .777).
#   The no-snap alternative is not the blob but the whole pipeline, so its training target becomes the out-of-fold
#   per-image score of S11 (pipeline without SAM, work/nested/S11.npy), and the ranker gets three more label-free
#   features: a decoder-verified box exists (the pipeline will replace the mask by it), its bit accuracy, the shape of the blob.
BOXES = ["pixelseal:S1_ps_amp4_grid", "pixelseal:S1_ps_amp4_grid_g2", "stegastamp:S1_ss", "stegastamp:S1_ss_grid_g2", "trustmark:S1_tm_grid"]


def box_info(split):
    """image index -> best bit accuracy of a decoder box that the pipeline would accept (same rule as src.full_nested)"""
    from src.apply_boxes import load_boxes
    from src.data import names
    ix = {n: i for i, n in enumerate(names(split))}; out = {}
    for spec in BOXES:
        m, mp = spec.split(":")
        for n, r in load_boxes(m, mp, split).items():
            y0, y1, x0, x1 = r["box"]; h, w = y1 - y0, x1 - x0
            if m == "trustmark" and max(h, w) > 2.0 * min(h, w):
                continue
            out[ix[n]] = max(out.get(ix[n], 0.0), float(r["acc"]))
    return out


PARAMS = dict(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=40, subsample=0.7, subsample_freq=1,
              colsample_bytree=0.7, reg_lambda=5.0, verbose=-1, n_jobs=16)
TOPA = 6


def load_prompt(split, nshards=2):
    out = {}
    for k in range(nshards):
        d = np.load(ROOT / "work/sam" / f"prompt_{split}_{k}of{nshards}.npz")
        for i, b, q, pid in zip(d["idx"], d["bits"], d["piou"], d["stab"]):
            out.setdefault(int(i), []).append((np.unpackbits(b).reshape(128, 128).astype(bool), float(q), int(pid)))
    return out


def blob_of(p):
    b = p >= 0.5 * float(p.max()); lab, n = ndi.label(b)
    if n == 0:
        return b, 0
    sz = ndi.sum(b, lab, range(1, n + 1))
    return lab == 1 + int(np.argmax(sz)), n


def cands(p, auto, prompt):
    """list of (mask, sam_score, source id) : 0 = blob / no snapping, 1 = automatic, 2-4 = prompted"""
    tot = float(p.sum()); blob, ncomp = blob_of(p); out = [(blob, 0.0, 0)]
    sj = []
    for s in auto:
        n = float(s.sum())
        if n > 0.5 * 16384:
            continue
        i_ = float(p[s].sum()); sj.append((i_ / (n + tot - i_ + 1e-6), s))
    sj.sort(key=lambda t: -t[0])
    out += [(s, 0.0, 1) for _, s in sj[:TOPA]]
    out += [(s, q, 1 + pid) for s, q, pid in prompt if 2 <= s.sum() <= 0.5 * 16384]
    return out, blob, ncomp


def feats(p, C, blob, ncomp, mcode, bacc=0.0):
    tot = float(p.sum()) + 1e-6; rows = []; nb = float(blob.sum()) + 1e-6
    ys, xs = np.where(blob) if blob.any() else (np.array([0]), np.array([0]))
    img = [float(p.max()), float(p.mean()), nb / 16384, nb / ((np.ptp(ys) + 1) * (np.ptp(xs) + 1)), float(ncomp)]
    if MODE == "M":
        bh, bw = float(np.ptp(ys) + 1), float(np.ptp(xs) + 1)
        img += [float(bacc >= 0.8), float(bacc), min(bh, bw) / max(bh, bw), min(bh, bw) / 128]
    J = []
    for s, q, src in C:
        n = float(s.sum()) + 1e-6; i_ = float(p[s].sum()); J.append(i_ / (n + tot - i_))
    order = np.argsort(-np.array(J[1:])) if len(J) > 1 else np.array([], int); rank = np.zeros(len(C)); rank[1 + order] = np.arange(len(order))
    jb = max(J[1:]) if len(J) > 1 else 0.0
    for c, (s, q, src) in enumerate(C):
        n = float(s.sum()) + 1e-6; ring = ndi.binary_dilation(s, iterations=3) & ~s
        mi_ = float(p[s].mean()) if s.any() else 0.0; mr = float(p[ring].mean()) if ring.any() else 0.0
        y2, x2 = np.where(s) if s.any() else (np.array([0]), np.array([0])); h, w = np.ptp(y2) + 1, np.ptp(x2) + 1
        ib = float((s & blob).sum())
        rows.append([J[c], rank[c], jb - J[c], mi_, mr, mi_ - mr, float(p[s].sum()) / tot, n / 16384, n / (h * w), min(h, w) / max(h, w),
                     ib / (n + nb - ib), ib / nb, ib / n, q, float(src == 0), float(src == 1), float(src >= 2), float(src), jb, J[0]]
                    + img + [float(mcode == k) for k in range(3)])
    return np.array(rows, np.float32)


def main():
    import lightgbm as lgb
    me = {"train": methods(), "test": methods("test")}; Fo = folds(); G = masks128().astype(bool); rg = regime()
    A = {sp: load_segs(sp) for sp in ("train", "test")}; Pp = {sp: load_prompt(sp) for sp in ("train", "test")}
    E = {sp: np.zeros((len(me[sp]), 128, 128), np.float32) for sp in me}
    for m in METHODS:
        for sp in me:
            E[sp][me[sp] == m] = np.load(ROOT / "work/runs" / ENS[m] / f"{sp}.npy").astype(np.float32)
    D = {}; BX = {sp: (box_info(sp) if MODE == "M" else {}) for sp in ("train", "test")}; S11 = np.load(ROOT / "work/nested/S11.npy")
    for sp in ("train", "test"):
        for i in range(len(me[sp])):
            C, blob, ncomp = cands(E[sp][i], A[sp].get(i, []), Pp[sp].get(i, []))
            X = feats(E[sp][i], C, blob, ncomp, METHODS.index(me[sp][i]), BX[sp].get(i, 0.0))
            y = np.array([miou_image(s, G[i]) for s, _, _ in C], np.float32) if sp == "train" else None
            if sp == "train" and MODE == "M":
                y[0] = S11[i]                       # no snapping = what the pipeline without SAM scores on this image (out of fold)
            D[(sp, i)] = (C, X, y)
    small = np.where(~rg)[0]
    def fit(tr_idx):
        X = np.concatenate([D[("train", i)][1] for i in tr_idx]); y = np.concatenate([D[("train", i)][2] for i in tr_idx])
        return [lgb.LGBMRegressor(**PARAMS, random_state=s).fit(X, y) for s in (0, 1, 2)]
    def choose(models, key):
        C, X, _ = D[key]; pr = np.mean([mdl.predict(X) for mdl in models], 0)
        return int(np.argmax(pr))
    pick = {}
    for k in range(5):
        models = fit([i for i in small if Fo[i] != k])
        for i in np.where(Fo == k)[0]:
            pick[("train", int(i))] = choose(models, ("train", int(i)))
    models = fit(list(small))
    for i in range(len(me["test"])):
        pick[("test", i)] = choose(models, ("test", i))
    # out-of-fold report on the small-regime train images (binary chosen masks, before any pipeline choice)
    for m in METHODS:
        idx = [i for i in small if me["train"][i] == m]
        ch = np.array([D[("train", i)][2][pick[("train", i)]] for i in idx]); orc = np.array([D[("train", i)][2].max() for i in idx])
        j1 = np.array([D[("train", i)][2][1] if len(D[("train", i)][2]) > 1 else D[("train", i)][2][0] for i in idx])
        snapped = np.mean([pick[("train", i)] != 0 for i in idx]); fd = [np.mean([c - S11[i] for c, i in zip(ch, idx) if Fo[i] == k]) for k in range(5)]
        print(f"SAMRANK {m}|small n={len(idx)}: chosen mask OOF {ch.mean():.4f} | S11 {S11[idx].mean():.4f} | best-soft-IoU segment always {j1.mean():.4f} | oracle {orc.mean():.4f} | snapped {snapped:.2f} | per-fold diff vs S11 {np.round(fd, 3)}", flush=True)
    print("test: share of images snapped per method", {m: round(float(np.mean([pick[('test', i)] != 0 for i in np.where(me['test'] == m)[0]])), 3) for m in METHODS}, flush=True)
    for m in METHODS:
        idx = {sp: np.where(me[sp] == m)[0] for sp in me}
        args = dict(method=m, regime="small", sam_rank=True, sam_ens=ENS[m], params=PARAMS, note="OOF LightGBM choice of the SAM segment (src/sam_rank.py)")
        for b in BASES[m] + [ENS[m]]:
            out = {}
            for sp in ("train", "test"):
                P = np.load(ROOT / "work/runs" / b / f"{sp}.npy").astype(np.float32); Q = P.copy()
                for kk, i in enumerate(idx[sp]):
                    c = pick[(sp, int(i))]
                    if c != 0:
                        Q[kk] = snap(P[kk], D[(sp, int(i))][0][c][0])
                out[sp] = Q
            save(f"{b}_sam{MODE}", out["train"], out["test"], dict(args, sam_base=b, sam_rank_mode=MODE))


if __name__ == "__main__":
    main()
