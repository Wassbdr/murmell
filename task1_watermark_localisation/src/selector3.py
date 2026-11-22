"""SEL3: MULTI-SOURCE per-image mask selector on top of a full_nested candidate.
v1 (src/selector.py) chooses (post-proc, threshold) per image on the ONE combo the base picked for the cell.
SEL3 chooses among masks of 5 sources per image:
  M = the base's pair mean, A / B = each member alone,
  E1 / E2 = two extra single runs of the method, picked on folds != k among the runs allowed in the cell (run_regime rule):
            small cells: best RECTANGLE specialist and best OBJECT specialist (type from the GT of folds != k only),
            full cells : the two best other singles.
  candidates per source = 6 post-procs x (21 abs + 7 relative thresholds) + max-sum rectangle of (p - t) for 5 t  => 865 / image
One LightGBM regressor (6 cells together) predicts the mIoU of (image, candidate) from GT-free statistics, including the
mean of EVERY source map inside / outside the candidate mask (consensus) and the source id.  The image leaves the base
candidate only if the predicted gain exceeds a margin chosen by inner CV.
Nesting: outer fold k -> combo, base candidate, extra sources, regressor and margin all come from folds != k.
Test: the base's final choice, extras picked on all 510, regressor on all 510, margin = median of the 5 folds.
    python -m src.selector3 --base M03                       -> work/runs/SEL3_M03_{ps,tm,ss} + SEL3_M03_final
Prints: oracle bounds, nested CV vs the base, and (with --ref) vs a reference per-image score file (e.g. base + v1 selector).
"""
import argparse
import json
import os
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
from src.metric import miou_batch, miou_image
from src.nested_cv import run_regime, summarize
from src.postproc_cv import OPTS, pp


RELS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
RECT = [0.2, 0.3, 0.4, 0.5, 0.6]
SRCS = os.environ.get("SEL_SOURCES", "MABEF")           # M pair mean, A / B members, E / F extra specialists
POPTS = list(OPTS) + ["rect"]
C1 = [(o, "abs", round(float(t), 3)) for o in OPTS for t in THRS] + [(o, "rel", a) for o in OPTS for a in RELS] + [("rect", "abs", t) for t in RECT]
CANDS = [(s_,) + c for s_ in SRCS for c in C1]
NF = 10 + 10 + len(POPTS) + 5
MARGINS = [0.0, 0.01, 0.02, 0.03, 0.04, 0.06, 0.08]
SEEDS = [0, 1, 2]
PARAMS = dict(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=40, subsample=0.7, subsample_freq=1,
              colsample_bytree=0.7, reg_lambda=5.0, verbose=-1, n_jobs=24)      # same fixed values as v1
KEYS = [f"{m}|{r}" for m in METHODS for r in ("F", "S")]
SHORT = {"pixelseal": "ps", "trustmark": "tm", "stegastamp": "ss"}
LIGHT = [(o, round(float(t), 3)) for o in OPTS for t in THRS]


def cand_id(o, t):
    return CANDS.index(("M", o, "abs", round(float(t), 3)))


def rectfit(x, t):
    w = x.astype(np.float64) - t; n = w.shape[0]
    Cs = np.concatenate([np.zeros((1, w.shape[1])), np.cumsum(w, 0)]); r0, r1 = np.triu_indices(n + 1, 1)
    Pm = np.concatenate([np.zeros((len(r0), 1)), np.cumsum(Cs[r1] - Cs[r0], 1)], 1)
    rm = np.minimum.accumulate(Pm, 1); d = Pm[:, 1:] - rm[:, :-1]
    k = int(np.argmax(d.max(1))); j = int(np.argmax(d[k])); i = int(np.argmin(Pm[k, :j + 1]))
    m = np.zeros(x.shape, bool)
    if d[k, j] > 0:
        m[r0[k]:r1[k], i:j + 1] = True
    return m


def stack5(xa, xb, xe, xf):
    """the 5 source maps M, A, B, E, F as float32 [5,128,128]"""
    xa, xb, xe, xf = (v.astype(np.float32) for v in (xa, xb, xe, xf))
    return np.stack([(xa + xb) / 2, xa, xb, xe, xf])


def make_mask(x, o, mode, v):
    if o == "rect":
        return rectfit(x, v)
    return pp(x >= (v if mode == "abs" else v * float(x.max())), o)


def cand_mask(X5, ci):
    s_, o, mode, v = CANDS[ci]
    return make_mask(X5["MABEF".index(s_)], o, mode, v)


def feats(x, mask, t, o, X5f, si):
    a = mask.mean(); n = 0; bf = 0.0; fl = 0.0; mf = mask.ravel()
    if mask.any():
        lab, n = ndi.label(mask); sz = ndi.sum(mask, lab, range(1, n + 1)); c = lab == 1 + int(np.argmax(sz))
        ys, xs = np.where(c); bf = c.sum() / ((np.ptp(ys) + 1) * (np.ptp(xs) + 1)); fl = sz.max() / mask.sum()
    q = x.ravel(); m = mf.astype(np.float32); N = q.size
    I1 = (m * q).sum(); s1 = m.sum(); pos = q.sum(); U1 = s1 + pos - I1; I0 = (N - s1) - (pos - I1); U0 = N - I1
    e = 0.5 * ((I1 / U1 if U1 > 1e-6 else 1.0) + (I0 / U0 if U0 > 1e-6 else 1.0))
    ec = 0.0
    if mask.any() and (~mask).any():
        inner = mask & ~ndi.binary_erosion(mask, iterations=2); outer = ndi.binary_dilation(mask, iterations=2) & ~mask
        if inner.any() and outer.any():
            ec = x[inner].mean() - x[outer].mean()
    pin = X5f[:, mf].mean(1) if mf.any() else np.zeros(5); pout = X5f[:, ~mf].mean(1) if (~mf).any() else np.ones(5)
    return ([t, 0.0, a, n, pin[si], pout[si], e, bf, fl, ec] + pin.tolist() + pout.tolist() + [float(o == oo) for oo in POPTS]
            + [float(si == j) for j in range(5)])


def one(args):
    X5, g = args
    X5 = X5.astype(np.float32); X5f = X5.reshape(5, -1)
    S = np.full(len(CANDS), np.nan, np.float32); Fe = np.zeros((len(CANDS), NF), np.float32); cache = {}; dup = {}
    for si in range(1, 5):                      # identical source maps (single-model base, missing extra) are computed once
        for sj in range(si):
            if si not in dup and np.array_equal(X5[si], X5[sj]):
                dup[si] = sj
    for ci, (s_, o, mode, v) in enumerate(CANDS):
        si = "MABEF".index(s_); x = X5[si]; t = v if mode == "abs" else v * float(x.max())
        key = (dup.get(si, si), o, round(t, 5))
        if key not in cache:
            mk = make_mask(x, o, mode, v); cache[key] = (np.nan if g is None else miou_image(mk, g), feats(x, mk, t, o, X5f, si))
        S[ci] = cache[key][0]; f = list(cache[key][1]); f[1] = float(mode == "rel"); f[-5:] = [float(si == j) for j in range(5)]; Fe[ci] = f
    x = X5[0]; H = X5 >= 0.5; u = lambda a_, b_: ((a_ & b_).sum() / (a_ | b_).sum()) if (a_ | b_).any() else 1.0
    imf = ([float(x.max()), x.mean(), np.quantile(x, .5), np.quantile(x, .9), np.quantile(x, .99), (x >= .3).mean(), (x >= .5).mean(),
            (x >= .7).mean(), ndi.label(x >= 0.5)[1], x.std()] + X5f.max(1).tolist() + H.reshape(5, -1).mean(1).tolist()
           + [u(H[0], H[j]) for j in range(1, 5)] + [float(np.abs(X5[0] - X5[j]).mean()) for j in range(1, 5)] + [u(H[3], H[4])])
    return S, Fe, np.array(imf, np.float32)


def light(args):
    """one single run on the cell's images: [n, 126] mIoU for OPTS x THRS (used to pick the extra sources)."""
    X, Gs = args
    return np.stack([miou_batch(np.stack([pp(x >= t, o) for x in X]), Gs) for o, t in LIGHT], 1)


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
    ap = argparse.ArgumentParser(); ap.add_argument("--base", default="M03"); ap.add_argument("--name", default="")
    ap.add_argument("--out_root", default=str(ROOT / "work/runs")); ap.add_argument("--procs", type=int, default=48)
    ap.add_argument("--ref", default="")       # per-image nested scores to compare with (e.g. base + v1 selector)
    ap.add_argument("--extras", type=int, default=1)
    ap.add_argument("--cand", default="")          # task1: write an autosubmit-ready candidate under this (pre-created) name
    ap.add_argument("--gate", type=float, default=0.003)   # ... only if nested gain over the base >= gate AND >= 4/5 folds
    ap.add_argument("--write_runs", type=int, default=0)   # per-method SEL3_* runs + _final dir (diagnostics; costs inodes)    # 0: E = F = M (no extra sources; = v2)
    a = ap.parse_args(); name = a.name or f"SEL3_{a.base}"
    rep = json.loads((ROOT / "work/nested" / f"{a.base}.json").read_text())
    G, Fo, me, mt = masks128(), folds(), methods(), methods("test"); ix = {n: i for i, n in enumerate(names())}
    ixt = {n: i for i, n in enumerate(names("test"))}; ov, ovt = {}, {}
    for spec in rep["boxes"]:
        m, mp = spec.split(":")
        for split, o_, idx_ in (("train", ov, ix), ("test", ovt, ixt)):
            for n, r in load_boxes(m, mp, split).items():
                y0, y1, x0, x1 = r["box"]; h, w = y1 - y0, x1 - x0
                if r["acc"] < rep["thr_box"] or (m == "trustmark" and max(h, w) > 2.0 * min(h, w)):
                    continue
                if idx_[n] not in o_ or r["acc"] >= o_[idx_[n]][1]:
                    o_[idx_[n]] = (cells(r["box"]), r["acc"])
    FK = list(range(5)) + ["all"]

    def pair_of(key, k):
        cb = tuple(rep["choices"][f"{key}|{k}"][0]) if k != "all" else tuple(rep["final"][key][0])
        return (cb[0], cb[-1])

    def base_ci(key, k):
        _, o, t = rep["choices"][f"{key}|{k}"] if k != "all" else rep["final"][key]
        return cand_id(o, t)

    t0 = time.time(); C = {}
    with Pool(a.procs) as pool:
        for key in KEYS:
            m, reg = key.split("|"); rf = reg == "F"
            sel = np.where((me == m) & (RT == rf))[0]; loc = RT[me == m] == rf
            selt = np.where((mt == m) & (RE == rf))[0]; loct = RE[mt == m] == rf
            cell = {"sel": sel, "selt": selt, "fold": Fo[sel], "boxed": np.array([i in ov for i in sel]), "tab": {}, "X": {}, "srcs": {}}
            # ---- extra sources, picked on folds != k among the runs allowed in this cell
            ok = [c for c in rep["cands"] if c != "snapDF" and (ROOT / "work/runs" / c / "args.json").exists()
                  and json.loads((ROOT / "work/runs" / c / "args.json").read_text()).get("method") == m]
            ok = [c for c in ok if run_regime(c) in ("any", {"F": "full", "S": "small"}[reg]) or run_regime(c).endswith("_proc")]
            ok = [c for c in ok if src(c, m)[1] is not None and not np.isnan(src(c, m)[0]).any()]
            if a.extras:
                L = dict(zip(ok, pool.map(light, [(src(c, m)[0][loc], G[sel]) for c in ok])))
            Gc = G[sel].astype(bool); isrect = np.zeros(len(sel), bool); isobj = np.zeros(len(sel), bool)
            for j in range(len(sel)):
                if Gc[j].any():
                    ys, xs = np.where(Gc[j]); r_ = Gc[j].sum() / ((np.ptp(ys) + 1) * (np.ptp(xs) + 1)) >= 0.95; isrect[j] = r_; isobj[j] = not r_
            for k in FK:
                pa, pb = pair_of(key, k)
                if not a.extras:
                    cell["srcs"][k] = (pa, pb, pa, pb); continue
                inn = (np.ones(len(sel), bool) if k == "all" else cell["fold"] != k) & ~cell["boxed"]
                rest = [c for c in ok if c not in (pa, pb)]
                if reg == "S":
                    e = max(rest, key=lambda c: L[c][inn & isrect].mean(0).max()); f = max([c for c in rest if c != e], key=lambda c: L[c][inn & isobj].mean(0).max())
                else:
                    rk = sorted(rest, key=lambda c: -L[c][inn].mean(0).max()); e, f = rk[0], rk[1]
                cell["srcs"][k] = (pa, pb, e, f)
            for sq in sorted(set(cell["srcs"].values())):
                X = stack5T = np.stack([stack5(*(src(c, m)[0][loc][j] for c in sq)) for j in range(len(sel))])
                r = pool.map(one, [(X[j], G[i]) for j, i in enumerate(sel)], chunksize=1)
                cell["tab"][sq] = {"S": np.stack([v[0] for v in r]), "F": np.stack([v[1] for v in r]), "I": np.stack([v[2] for v in r])}
                cell["X"][sq] = X.astype(np.float16)
            sq = cell["srcs"]["all"]; Xt = np.stack([stack5(*(src(c, m)[1][loct][j] for c in sq)) for j in range(len(selt))])
            r = pool.map(one, [(Xt[j], None) for j in range(len(selt))], chunksize=1)
            cell["test"] = {"F": np.stack([v[1] for v in r]), "I": np.stack([v[2] for v in r])}; cell["Xt"] = Xt.astype(np.float16)
            C[key] = cell
            print(f"features {key} train {len(sel)} test {len(selt)} | sources per fold: " + "; ".join(f"{k}: E={cell['srcs'][k][2]} F={cell['srcs'][k][3]}" for k in FK)
                  + f" | {time.time() - t0:.0f}s", flush=True)

    def tab_of(key, k):
        return C[key]["tab"][C[key]["srcs"][k]]

    def fit(train_folds, k):
        Xs, ys = [], []
        for key in KEYS:
            cell = C[key]; idx = np.where(np.isin(cell["fold"], train_folds) & ~cell["boxed"])[0]
            Xs.append(rows(tab_of(key, k), key, idx)); ys.append(tab_of(key, k)["S"][idx].reshape(-1))
        Xs, ys = np.concatenate(Xs), np.concatenate(ys)
        return [lgb.LGBMRegressor(**PARAMS, random_state=sd).fit(Xs, ys) for sd in SEEDS]

    def predict(models, tab, key, idx):
        return np.mean([mdl.predict(rows(tab, key, idx)) for mdl in models], 0).reshape(len(idx), len(CANDS))

    P = {m: np.zeros(((me == m).sum(), 128, 128), np.float16) for m in METHODS}
    T = {m: np.zeros(((mt == m).sum(), 128, 128), np.float16) for m in METHODS}
    pos = {m: {i: j for j, i in enumerate(np.where(me == m)[0])} for m in METHODS}
    post = {m: {i: j for j, i in enumerate(np.where(mt == m)[0])} for m in METHODS}
    s_sel = np.full(510, np.nan); s_base = np.full(510, np.nan); s_or = np.full(510, np.nan); s_or1 = np.full(510, np.nan); margins = []; nsw = 0
    isM = np.array([c[0] == "M" and c[1] != "rect" for c in CANDS]); srcuse = {s_: 0 for s_ in "MABEF"}
    for k in range(5):
        inner = [j for j in range(5) if j != k]; acc = np.zeros(len(MARGINS))
        for j in inner:
            ms = fit([f for f in inner if f != j], k)
            for key in KEYS:
                cell = C[key]; idx = np.where((cell["fold"] == j) & ~cell["boxed"])[0]
                if len(idx) == 0:
                    continue
                tab = tab_of(key, k); p = predict(ms, tab, key, idx); bi = base_ci(key, k); S = tab["S"][idx]
                for mi_, mg in enumerate(MARGINS):
                    acc[mi_] += (S[np.arange(len(idx)), pick(p, bi, mg)] - S[:, bi]).sum()
        mg = MARGINS[int(np.argmax(acc))]; margins.append(mg)
        ms = fit(inner, k)
        for key in KEYS:
            m = key.split("|")[0]; cell = C[key]; idx = np.where(cell["fold"] == k)[0]; sq = cell["srcs"][k]
            tab = cell["tab"][sq]; p = predict(ms, tab, key, idx); bi = base_ci(key, k); ch = pick(p, bi, mg); nsw += int((ch != bi).sum())
            for n_, j in enumerate(idx):
                i = cell["sel"][j]; P[m][pos[m][i]] = cand_mask(cell["X"][sq][j].astype(np.float32), ch[n_])
                bx = miou_image(ov[i][0], G[i]) if i in ov else None
                s_sel[i] = bx if i in ov else tab["S"][j, ch[n_]]; s_base[i] = bx if i in ov else tab["S"][j, bi]
                s_or[i] = bx if i in ov else tab["S"][j].max(); s_or1[i] = bx if i in ov else tab["S"][j][isM].max()
                if i not in ov and ch[n_] != bi:
                    srcuse[CANDS[ch[n_]][0]] += 1
        print(f"outer fold {k}: margin {mg} (inner gains {np.round(acc, 2).tolist()}) {time.time() - t0:.0f}s", flush=True)
    fm = lambda s: np.array([s[Fo == k].mean() for k in range(5)])
    fs, fb = fm(s_sel), fm(s_base); d = fs - fb
    print(f"ORACLE per image: base {fb.mean():.6f} | best (post-proc, thr) on the base pair {fm(s_or1).mean():.6f} (+{fm(s_or1).mean() - fb.mean():.4f}) | "
          f"best of the 5 sources {fm(s_or).mean():.6f} (+{fm(s_or).mean() - fb.mean():.4f})")
    print(f"{name}: nested CV of the base {fb.mean():.6f} -> with SEL3 {fs.mean():.6f} (gain {d.mean():+.6f}; fold diffs "
          + " ".join(f"{x:+.4f}" for x in d) + f"; {int((d > 0).sum())}/5 folds; {nsw} images switched; source of the switch {srcuse})", flush=True)
    cellrep = {}
    for key in KEYS:
        sel = C[key]["sel"]; cellrep[key] = [float(s_base[sel].mean()), float(s_sel[sel].mean()), float(s_or[sel].mean())]
        print(f"  {key:14s} n={len(sel):3d} base {s_base[sel].mean():.4f} SEL3 {s_sel[sel].mean():.4f} ({s_sel[sel].mean() - s_base[sel].mean():+.4f}) oracle {s_or[sel].mean():.4f}")
    if a.ref:
        sr = np.load(a.ref); fr = fm(sr); d2 = fs - fr
        print(f"vs reference {a.ref}: {fr.mean():.6f} -> {fs.mean():.6f} (gain {d2.mean():+.6f}; fold diffs " + " ".join(f"{x:+.4f}" for x in d2)
              + f"; {int((d2 > 0).sum())}/5 folds) | cells " + " ".join(f"{key[:2]}{key[-1]} {s_sel[C[key]['sel']].mean() - sr[C[key]['sel']].mean():+.4f}" for key in KEYS), flush=True)
    mg = float(np.median(margins)); ms = fit([0, 1, 2, 3, 4], "all"); nt = 0
    for key in KEYS:
        m = key.split("|")[0]; cell = C[key]; idx = np.arange(len(cell["selt"]))
        p = predict(ms, cell["test"], key, idx); bi = base_ci(key, "all"); ch = pick(p, bi, mg); nt += int((ch != bi).sum())
        for j in idx:
            T[m][post[m][cell["selt"][j]]] = cand_mask(cell["Xt"][j].astype(np.float32), ch[j])
    print(f"test: margin {mg}, {nt}/1020 images switched from the base candidate", flush=True)
    B = np.zeros((1020, 128, 128), bool)
    for m in METHODS:
        for i, j in post[m].items():
            B[i] = ovt[i][0].astype(bool) if i in ovt else T[m][j] >= 0.5
    r2 = summarize(s_sel); r2.update(clean=True, kind="full_nested+selector3", base=a.base, base_cv=float(fb.mean()), gain=float(d.mean()),
                                     fold_diffs=d.tolist(), margins=margins, margin_test=mg, boxes=rep["boxes"], thr_box=rep["thr_box"],
                                     sources={key: {str(k): list(C[key]["srcs"][k]) for k in FK} for key in KEYS},
                                     test_switched=nt, test_boxed=len(ovt), test_wm_frac=float(B.mean()))
    gate = bool(d.mean() >= a.gate and (d > 0).sum() >= 4)
    print(f"SEL gate: base {fb.mean():.6f} -> selector {fs.mean():.6f} (gain {d.mean():+.6f}, {int((d > 0).sum())}/5 folds) => "
          + ("KEEP" if gate else "drop (gate not met)") + " [selector3]", flush=True)
    if a.cand and gate:
        import hashlib
        from src.submit import write
        r2.update(variants_tried="selector v1: 20 configs ; selector3: 1", base_command=json.loads((ROOT / "work/archive" / a.base / "spec.json").read_text()).get("command"))
        (ROOT / "work/nested" / f"{a.cand}.json").write_text(json.dumps(r2, indent=1)); np.save(ROOT / "work/nested" / f"{a.cand}.npy", s_sel)
        np.save(ROOT / "work/test" / f"{a.cand}_bin.npy", B)
        adir = ROOT / "work/archive" / a.cand; os.makedirs(adir, exist_ok=True); pre = adir / f"{a.cand}.pt"
        outp = write(B, a.cand, f"{r2['cv_foldmean']:.4f}", out=pre if pre.exists() else None)
        (adir / "spec.json").write_text(json.dumps({"name": a.cand, "kind": "full_nested+selector3", "base": a.base,
            "command": f"python -m src.selector3 --base {a.base} --cand {a.cand}   # after rebuilding the base with its own command",
            "base_command": r2["base_command"], "margins": margins, "margin_test": mg, "params": PARAMS, "seeds": SEEDS,
            "submission_file": str(outp.relative_to(ROOT)), "submission_md5": hashlib.md5(open(outp, "rb").read()).hexdigest()}, indent=1))
        print(f"SEL candidate {a.cand}: nested {r2['cv_foldmean']:.6f} archived {outp}", flush=True)
    if a.cand and not gate and (ROOT / "work/nested" / f"{a.cand}.json").exists():
        r2.update(clean=False, clean_note="selector3 gate not met on this base: not a candidate")
        (ROOT / "work/nested" / f"{a.cand}.json").write_text(json.dumps(r2, indent=1))
    for m in METHODS if a.write_runs else []:
        out = Path(a.out_root) / f"{name}_{SHORT[m]}"; out.mkdir(parents=True, exist_ok=True)
        np.save(out / "train.npy", P[m]); np.save(out / "test.npy", T[m])
        (out / "args.json").write_text(json.dumps({
            "method": m, "exp": out.name, "kind": "selector3", "regime": "any", "fold_excluded": True, "base": a.base, "binary": True,
            "margins": margins, "margin_test": mg, "seeds": SEEDS, "params": PARAMS, "nested_base": float(fb.mean()),
            "nested_selector": float(fs.mean()), "fold_diffs": d.tolist(), "cells": cellrep,
            "note": "0/1 masks stored as maps; final pipeline stage, not a pool source (see SEL note)."}))
        print("written", out, flush=True)


if __name__ == "__main__":
    main()
