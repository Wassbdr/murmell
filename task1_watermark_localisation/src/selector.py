"""SEL: learned PER-IMAGE mask selector on top of a full_nested candidate.
For each image, the base candidate (work/nested/<base>.json) uses ONE (post-proc, threshold) per cell.  The oracle that
picks the best (post-proc, threshold) PER IMAGE is +0.038 above it.  SEL learns that choice from GT-free features:
  candidates per image = 6 post-procs x (21 absolute thresholds + 7 relative thresholds a*max(map))  [168]
  one LightGBM regressor (all 6 cells together, cell one-hot) predicts the mIoU of (image, candidate) from the map's
  statistics (area, components, p inside/outside, plug-in expected mIoU, bbox fill, edge contrast, quantiles ...)
  the image switches from the cell's base candidate to the predicted-best one only if predicted gain > margin.
Nesting: for outer fold k the combo and base candidate are those the base chose on folds != k, the regressor is fitted
on folds != k, and the margin is chosen by an inner CV over folds != k.  => train masks are out-of-fold.
Test: combos/base candidates of the base's "final" choice, regressor fitted on all 510, margin = median of the 5 folds.

Output = ordinary runs (one per method), binary masks stored as float16 maps (any threshold gives the same mask):
  <out_root>/SEL_<base>_{ps,tm,ss}/{train.npy [170,128,128], test.npy [340,128,128], args.json (fold_excluded=true)}
    python -m src.selector --base N02                      (copied into src/: writes work/runs/SEL_N02_*)
"""
import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

from src.apply_boxes import cells, load_boxes
from src.combine2 import RE, RT, src
from src.data import METHODS, ROOT, folds, masks128, methods, names
from src.eval_cv import THRS
from src.metric import miou_image
from src.postproc_cv import OPTS, pp


RELS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
CANDS = [(o, "abs", round(float(t), 3)) for o in OPTS for t in THRS] + [(o, "rel", a) for o in OPTS for a in RELS]
MARGINS = [0.0, 0.01, 0.02, 0.03, 0.04, 0.06]
SEEDS = [0, 1, 2]
PARAMS = dict(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=40, subsample=0.7, subsample_freq=1,
              colsample_bytree=0.7, reg_lambda=5.0, verbose=-1, n_jobs=24)      # fixed a priori, never tuned
KEYS = [f"{m}|{r}" for m in METHODS for r in ("F", "S")]
SHORT = {"pixelseal": "ps", "trustmark": "tm", "stegastamp": "ss"}


def cand_id(o, t):
    return CANDS.index((o, "abs", round(float(t), 3)))


def cand_mask(x, ci):
    o, mode, v = CANDS[ci]
    return pp(x >= (v if mode == "abs" else v * float(x.max())), o)


def feats(x, mask, t, o):
    a = mask.mean(); n = 0; bf = 0.0; fl = 0.0
    if mask.any():
        lab, n = ndi.label(mask); sz = ndi.sum(mask, lab, range(1, n + 1)); c = lab == 1 + int(np.argmax(sz))
        ys, xs = np.where(c); bf = c.sum() / ((np.ptp(ys) + 1) * (np.ptp(xs) + 1)); fl = sz.max() / mask.sum()
    p_in = x[mask].mean() if mask.any() else 0.0; p_out = x[~mask].mean() if (~mask).any() else 1.0
    q = x.ravel(); m = mask.ravel().astype(np.float32); N = q.size
    I1 = (m * q).sum(); s1 = m.sum(); pos = q.sum(); U1 = s1 + pos - I1; I0 = (N - s1) - (pos - I1); U0 = N - I1
    e = 0.5 * ((I1 / U1 if U1 > 1e-6 else 1.0) + (I0 / U0 if U0 > 1e-6 else 1.0))       # plug-in expected mIoU
    ec = 0.0                                                                           # contrast across the mask border
    if mask.any() and (~mask).any():
        inner = mask & ~ndi.binary_erosion(mask, iterations=2); outer = ndi.binary_dilation(mask, iterations=2) & ~mask
        if inner.any() and outer.any():
            ec = x[inner].mean() - x[outer].mean()
    return [t, 0.0, a, n, p_in, p_out, e, bf, fl, ec] + [float(o == oo) for oo in OPTS]


def one(args):
    """x: 128x128 map, g: GT mask or None -> (scores [C] or nan, candidate features [C, d], image features)."""
    x, g = args
    x = x.astype(np.float32); mx = float(x.max())
    S = np.full(len(CANDS), np.nan, np.float32); Fe = np.zeros((len(CANDS), 16), np.float32); cache = {}
    for ci, (o, mode, v) in enumerate(CANDS):
        t = v if mode == "abs" else v * mx
        key = (o, round(t, 5))
        if key not in cache:
            mk = pp(x >= t, o); cache[key] = (np.nan if g is None else miou_image(mk, g), feats(x, mk, t, o))
        S[ci] = cache[key][0]; f = list(cache[key][1]); f[1] = float(mode == "rel"); Fe[ci] = f
    n = ndi.label(x >= 0.5)[1]
    imf = [mx, x.mean(), np.quantile(x, .5), np.quantile(x, .9), np.quantile(x, .99), (x >= .3).mean(), (x >= .5).mean(),
           (x >= .7).mean(), n, x.std()]
    return S, Fe, np.array(imf, np.float32)


def rows(tab, key, idx):
    n, C = len(idx), len(CANDS)
    Fe = tab["F"][idx]; I = tab["I"][idx]; Im = np.repeat(I[:, None], C, 1)
    ko = np.zeros((n, C, len(KEYS)), np.float32); ko[:, :, KEYS.index(key)] = 1
    rel = np.stack([Fe[:, :, 2] / (I[:, None, 6] + 1e-4), Fe[:, :, 0] / (I[:, None, 0] + 1e-4)], 2)
    return np.concatenate([Fe, Im, ko, rel], 2).reshape(n * C, -1)


def pick(p, bi, margin):
    b = p.argmax(1)
    return np.where(p.max(1) - p[:, bi] > margin, b, bi)


def main():
    import lightgbm as lgb
    ap = argparse.ArgumentParser(); ap.add_argument("--base", default="N02"); ap.add_argument("--name", default="")
    ap.add_argument("--out_root", default=str(ROOT / "work/runs")); ap.add_argument("--procs", type=int, default=32)
    ap.add_argument("--cand", default="")          # task1: also write an autosubmit-ready candidate under this (pre-created) name
    ap.add_argument("--gate", type=float, default=0.003)   # ... only if nested gain >= gate AND >= 4/5 outer folds positive
    ap.add_argument("--write_runs", type=int, default=0)   # per-method SEL_* runs (not a pool source: see the warning)
    a = ap.parse_args(); name = a.name or f"SEL_{a.base}"
    rep = json.loads((ROOT / "work/nested" / f"{a.base}.json").read_text())
    G, Fo, me, mt = masks128(), folds(), methods(), methods("test"); ix = {n: i for i, n in enumerate(names())}
    ixt = {n: i for i, n in enumerate(names("test"))}; ov, ovt = {}, {}      # decoder-ranked box overrides, exactly as full_nested
    for spec in rep["boxes"]:
        m, mp = spec.split(":")
        for split, o_, idx_ in (("train", ov, ix), ("test", ovt, ixt)):
            for n, r in load_boxes(m, mp, split).items():
                y0, y1, x0, x1 = r["box"]; h, w = y1 - y0, x1 - x0
                if r["acc"] < rep["thr_box"] or (m == "trustmark" and max(h, w) > 2.0 * min(h, w)):
                    continue
                if idx_[n] not in o_ or r["acc"] >= o_[idx_[n]][1]:
                    o_[idx_[n]] = (cells(r["box"]), r["acc"])
    t0 = time.time(); C = {}
    with Pool(a.procs) as pool:
        for key in KEYS:
            m, reg = key.split("|"); rf = reg == "F"
            sel = np.where((me == m) & (RT == rf))[0]; loc = RT[me == m] == rf
            selt = np.where((mt == m) & (RE == rf))[0]; loct = RE[mt == m] == rf
            fin = tuple(rep["final"][key][0])
            cell = {"sel": sel, "selt": selt, "fold": Fo[sel], "boxed": np.array([i in ov for i in sel]), "tab": {}, "X": {}}
            for cb in sorted({tuple(rep["choices"][f"{key}|{k}"][0]) for k in range(5)} | {fin}):
                X = (sum(src(c, m)[0] for c in cb) / len(cb))[loc]
                r = pool.map(one, [(X[j], G[i]) for j, i in enumerate(sel)], chunksize=2)
                cell["tab"][cb] = {"S": np.stack([v[0] for v in r]), "F": np.stack([v[1] for v in r]), "I": np.stack([v[2] for v in r])}
                cell["X"][cb] = X
            Xt = (sum(src(c, m)[1] for c in fin) / len(fin))[loct]
            r = pool.map(one, [(Xt[j], None) for j in range(len(selt))], chunksize=2)
            cell["test"] = {"F": np.stack([v[1] for v in r]), "I": np.stack([v[2] for v in r])}; cell["Xt"] = Xt
            C[key] = cell
            print(f"features {key} train {len(sel)} test {len(selt)} {time.time() - t0:.0f}s", flush=True)

    def cb_of(key, k):
        return tuple(rep["choices"][f"{key}|{k}"][0]) if k != "all" else tuple(rep["final"][key][0])

    def base_ci(key, k):
        _, o, t = rep["choices"][f"{key}|{k}"] if k != "all" else rep["final"][key]
        return cand_id(o, t)

    def fit(train_folds, k):
        Xs, ys = [], []
        for key in KEYS:
            cell = C[key]; idx = np.where(np.isin(cell["fold"], train_folds) & ~cell["boxed"])[0]
            Xs.append(rows(cell["tab"][cb_of(key, k)], key, idx)); ys.append(cell["tab"][cb_of(key, k)]["S"][idx].reshape(-1))
        Xs, ys = np.concatenate(Xs), np.concatenate(ys)
        return [lgb.LGBMRegressor(**PARAMS, random_state=sd).fit(Xs, ys) for sd in SEEDS]

    def predict(models, tab, key, idx):
        return np.mean([mdl.predict(rows(tab, key, idx)) for mdl in models], 0).reshape(len(idx), len(CANDS))

    P = {m: np.zeros(((me == m).sum(), 128, 128), np.float16) for m in METHODS}
    T = {m: np.zeros(((mt == m).sum(), 128, 128), np.float16) for m in METHODS}
    pos = {m: {i: j for j, i in enumerate(np.where(me == m)[0])} for m in METHODS}
    post = {m: {i: j for j, i in enumerate(np.where(mt == m)[0])} for m in METHODS}
    s_sel = np.full(510, np.nan); s_base = np.full(510, np.nan); margins = []; nsw = 0
    for k in range(5):
        inner = [j for j in range(5) if j != k]; acc = np.zeros(len(MARGINS))
        for j in inner:                                                   # inner CV for the margin (folds != k only)
            ms = fit([f for f in inner if f != j], k)
            for key in KEYS:
                cell = C[key]; idx = np.where((cell["fold"] == j) & ~cell["boxed"])[0]
                if len(idx) == 0:
                    continue
                tab = cell["tab"][cb_of(key, k)]; p = predict(ms, tab, key, idx); bi = base_ci(key, k); S = tab["S"][idx]
                for mi, mg in enumerate(MARGINS):
                    acc[mi] += (S[np.arange(len(idx)), pick(p, bi, mg)] - S[:, bi]).sum()
        mg = MARGINS[int(np.argmax(acc))]; margins.append(mg)
        ms = fit(inner, k)
        for key in KEYS:
            m = key.split("|")[0]; cell = C[key]; idx = np.where(cell["fold"] == k)[0]; cb = cb_of(key, k)
            tab = cell["tab"][cb]; p = predict(ms, tab, key, idx); bi = base_ci(key, k); ch = pick(p, bi, mg); nsw += int((ch != bi).sum())
            for n_, j in enumerate(idx):
                i = cell["sel"][j]; P[m][pos[m][i]] = cand_mask(cell["X"][cb][j], ch[n_])
                s_sel[i] = miou_image(ov[i][0], G[i]) if i in ov else tab["S"][j, ch[n_]]
                s_base[i] = miou_image(ov[i][0], G[i]) if i in ov else tab["S"][j, bi]
        print(f"outer fold {k}: margin {mg} (inner gains {np.round(acc, 2).tolist()}) {time.time() - t0:.0f}s", flush=True)
    fs = np.array([s_sel[Fo == k].mean() for k in range(5)]); fb = np.array([s_base[Fo == k].mean() for k in range(5)]); d = fs - fb
    print(f"{name}: nested CV of the base {fb.mean():.6f} -> with the selector {fs.mean():.6f} (gain {d.mean():+.6f}; fold diffs "
          + " ".join(f"{x:+.4f}" for x in d) + f"; {int((d > 0).sum())}/5 folds; {nsw} images switched)", flush=True)
    cellrep = {}
    for key in KEYS:
        sel = C[key]["sel"]; cellrep[key] = [float(s_base[sel].mean()), float(s_sel[sel].mean())]
        print(f"  {key:14s} n={len(sel):3d} base {s_base[sel].mean():.4f} selector {s_sel[sel].mean():.4f} ({s_sel[sel].mean() - s_base[sel].mean():+.4f})")
    # test: fitted on all train with the base's final choices
    mg = float(np.median(margins)); ms = fit([0, 1, 2, 3, 4], "all"); nt = 0
    for key in KEYS:
        m = key.split("|")[0]; cell = C[key]; idx = np.arange(len(cell["selt"]))
        p = predict(ms, cell["test"], key, idx); bi = base_ci(key, "all"); ch = pick(p, bi, mg); nt += int((ch != bi).sum())
        for j in idx:
            T[m][post[m][cell["selt"][j]]] = cand_mask(cell["Xt"][j], ch[j])
    print(f"test: margin {mg}, {nt}/1020 images switched from the base candidate", flush=True)
    # final candidate = selector masks in every cell + box overrides (same rule as the base) -> ready for src.submit.write
    from src.nested_cv import summarize
    B = np.zeros((1020, 128, 128), bool)
    for m in METHODS:
        for i, j in post[m].items():
            B[i] = ovt[i][0].astype(bool) if i in ovt else T[m][j] >= 0.5
    r2 = summarize(s_sel); r2.update(clean=True, kind="full_nested+selector", base=a.base, base_cv=float(fb.mean()), gain=float(d.mean()),
                                     fold_diffs=d.tolist(), margins=margins, margin_test=mg, boxes=rep["boxes"], thr_box=rep["thr_box"],
                                     test_switched=nt, test_boxed=len(ovt), test_wm_frac=float(B.mean()))
    gate = bool(d.mean() >= a.gate and (d > 0).sum() >= 4)
    print(f"SEL gate: base {fb.mean():.6f} -> selector {fs.mean():.6f} (gain {d.mean():+.6f}, {int((d > 0).sum())}/5 folds) => "
          + ("KEEP" if gate else "drop (gate not met)"), flush=True)
    if a.cand and gate:
        import hashlib, os
        from src.submit import write
        r2.update(variants_tried=20, base_command=json.loads((ROOT / "work/archive" / a.base / "spec.json").read_text()).get("command"))
        (ROOT / "work/nested" / f"{a.cand}.json").write_text(json.dumps(r2, indent=1)); np.save(ROOT / "work/nested" / f"{a.cand}.npy", s_sel)
        np.save(ROOT / "work/test" / f"{a.cand}_bin.npy", B)
        dd = ROOT / "work/archive" / a.cand; os.makedirs(dd, exist_ok=True); pre = dd / f"{a.cand}.pt"
        outp = write(B, a.cand, f"{r2['cv_foldmean']:.4f}", out=pre if pre.exists() else None)
        (dd / "spec.json").write_text(json.dumps({"name": a.cand, "kind": "full_nested+selector", "base": a.base,
            "command": f"python -m src.selector --base {a.base} --cand {a.cand}   # after rebuilding the base with its own command",
            "base_command": r2["base_command"], "margins": margins, "margin_test": mg, "params": PARAMS, "seeds": SEEDS,
            "submission_file": str(outp.relative_to(ROOT)), "submission_md5": hashlib.md5(open(outp, "rb").read()).hexdigest()}, indent=1))
        print(f"SEL candidate {a.cand}: nested {r2['cv_foldmean']:.6f} archived {outp}", flush=True)
    if a.cand and not gate and (ROOT / "work/nested" / f"{a.cand}.json").exists():     # never leave a stale sendable candidate
        r2.update(clean=False, clean_note="selector gate not met on this base: not a candidate")
        (ROOT / "work/nested" / f"{a.cand}.json").write_text(json.dumps(r2, indent=1))
    for m in METHODS if a.write_runs else []:
        out = Path(a.out_root) / f"{name}_{SHORT[m]}"; out.mkdir(parents=True, exist_ok=True)
        np.save(out / "train.npy", P[m]); np.save(out / "test.npy", T[m])
        (out / "args.json").write_text(json.dumps({
            "method": m, "exp": out.name, "kind": "selector", "regime": "any", "fold_excluded": True, "base": a.base,
            "binary": True, "margins": margins, "margin_test": mg, "seeds": SEEDS, "params": PARAMS,
            "nested_base": float(fb.mean()), "nested_selector": float(fs.mean()), "fold_diffs": d.tolist(), "cells": cellrep,
            "note": "0/1 masks stored as maps; images routed by the B0 regime vector (RT/RE); boxed images keep the base mask "
                    "(full_nested overrides them anyway). Use as a SINGLE source (pairs with probability maps are meaningless)."}))
        print("written", out, flush=True)


if __name__ == "__main__":
    main()
