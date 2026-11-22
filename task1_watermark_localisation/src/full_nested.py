"""ONE fully nested selection per (method x PREDICTED regime) cell of: source combination (singles + equal-weight pairs
of candidate runs, restricted like C4r unless the run is fold-excluded / procedural / snapDF) x structural post-proc
(postproc_cv.OPTS) x threshold, chosen on the 4 inner folds, scored on the outer fold; decoder-ranked box override
(GT-free rule acc >= thr_box) applied after post-processing, as at test time.  Final choice = best on all 510 => test.
    python -m src.full_nested --name C7 --cands F2_ps_amp4 S1_ps_amp4 ... snapDF --boxes trustmark:S1_tm_grid ... --write 1
Writes work/nested/<name>.{json,npy}, work/test/<name>_bin.npy, (--write) submission + work/archive/<name>/spec.json.
"""
import argparse
import hashlib
import itertools
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

from src.apply_boxes import cells, load_boxes
from src.combine2 import RE, RT, src
from src.data import METHODS, ROOT, folds, masks128, methods, names
from src.eval_cv import THRS
from src.metric import miou_batch, miou_image
from src.nested_cv import run_regime, summarize
from src.postproc_cv import OPTS, pp

_W = {}


def _score_combo(args):
    cb, = args
    X, G, ov = _W["X"][cb], _W["G"], _W["ov"]
    out = {}
    for o in OPTS:
        for t in THRS:
            pred = np.stack([pp(X[j] >= t, o) for j in range(len(X))])
            s = miou_batch(pred, G)
            for j, v in ov.items():
                s[j] = v
            out[(o, t)] = s
    return cb, out


# ---------------------------------------------------------------- per-image mask decoding (human 23:45, item 3)
CANDS = [(o, t) for o in OPTS for t in THRS]


def _logit(X):
    X = np.clip(X.astype(np.float64), 1e-4, 1 - 1e-4); return np.log(X / (1 - X))


def _platt(x, y, iters=25):
    """q = sigmoid(a x + b), 2 parameters fitted by Newton on the cells of the SELECTION images only."""
    a, b = 1.0, 0.0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(a * x + b, -30, 30))); w = p * (1 - p) + 1e-9; g = p - y
        ga, gb = (g * x).mean(), g.mean(); haa, hab, hbb = (w * x * x).mean(), (w * x).mean(), w.mean()
        det = haa * hbb - hab * hab + 1e-12
        a, b = a - (hbb * ga - hab * gb) / det, b - (haa * gb - hab * ga) / det
    return float(a), float(b)


def _exp_pick(M, q):
    """M [C, N] 0/1 candidate masks, q [N] calibrated probabilities -> candidate with the best plug-in expected mIoU."""
    N = M.shape[1]; I1 = M @ q; s1 = M.sum(1); pos = q.sum()
    U1 = s1 + pos - I1; iou1 = np.where(U1 > 1e-6, I1 / np.maximum(U1, 1e-6), 1.0)
    I0 = (N - s1) - (pos - I1); U0 = N - I1; iou0 = np.where(U0 > 1e-6, I0 / np.maximum(U0, 1e-6), 1.0)
    return int(np.argmax(0.5 * (iou1 + iou0)))


def _cand_masks(x):
    return np.stack([pp(x >= t, o).ravel() for o, t in CANDS]).astype(np.float32)


def _exp_combo(args):
    """per selection set (outer fold k -> its 4 inner folds / all): Platt calibration fitted on that set, then EACH image
    gets the candidate mask (threshold x structure) with the highest expected mIoU. Returns realised per-image scores."""
    cb, = args
    X, G, masks, tab, ov = _W["X"][cb], _W["G"], _W["masks"], _W["tabs"][cb], _W["ov"]
    n = len(X); L = _logit(X).reshape(n, -1); Y = (G.reshape(n, -1) > 0).astype(np.float64)
    notov = np.array([j not in ov for j in range(n)]); S = np.stack([tab[c] for c in CANDS], 1)
    cal = {}
    for key, msk in masks.items():
        fit = msk & notov
        cal[key] = _platt(L[fit][:, ::3].ravel(), Y[fit][:, ::3].ravel()) if fit.sum() >= 5 else (1.0, 0.0)
    out = {key: np.zeros(n) for key in masks}
    for j in range(n):
        M = _cand_masks(X[j])
        for key, (a_, b_) in cal.items():
            out[key][j] = S[j, _exp_pick(M, 1 / (1 + np.exp(-np.clip(a_ * L[j] + b_, -30, 30))))]
    return cb, out, cal


def _exp_test(args):
    x, a_, b_ = args
    M = _cand_masks(x); q = 1 / (1 + np.exp(-np.clip(a_ * _logit(x).ravel() + b_, -30, 30)))
    return M[_exp_pick(M, q)].reshape(128, 128) > 0.5


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--name"); ap.add_argument("--cands", nargs="+")
    ap.add_argument("--boxes", nargs="*", default=[]); ap.add_argument("--thr_box", type=float, default=0.8)
    ap.add_argument("--tm_maxasp", type=float, default=2.0)     # official doc: TrustMark rectangles <= 2:1
    ap.add_argument("--restrict", type=int, default=1); ap.add_argument("--pairs", type=int, default=1)
    ap.add_argument("--write", type=int, default=0); ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--topk", type=int, default=6)       # pairs only among the top-k singles (0 = all pairs)
    ap.add_argument("--means", default="")               # e.g. "3,4,5": also try the mean of the top-n singles
    ap.add_argument("--stack", type=int, default=0)      # 1: add the nested 128x128 stacker (src.stack2) as a candidate
    ap.add_argument("--stack_gate", type=float, default=0.003)   # keep it only if gain >= gate and >= 4/5 outer folds
    ap.add_argument("--topm", type=int, default=0)       # M > 0: ADV variant = equal-weight ensemble of the top-M combos (ranked on
    ap.add_argument("--expdec", type=int, default=0)     # N > 0: per-image expected-mIoU decoding for the top-N combos per cell
    a = ap.parse_args()
    MEANS = [int(v) for v in a.means.split(",") if v]
    G, Fo, me, mt = masks128(), folds(), methods(), methods("test")
    ix = {n: i for i, n in enumerate(names())}; ixt = {n: i for i, n in enumerate(names("test"))}
    ovr, ovt = {}, {}
    def ok_box(m, r):
        y0, y1, x0, x1 = r["box"]; h, w = y1 - y0, x1 - x0
        return r["acc"] >= a.thr_box and not (m == "trustmark" and a.tm_maxasp and max(h, w) > a.tm_maxasp * min(h, w))
    for spec in a.boxes:     # later specs override earlier ones for the same image; best acc wins between specs
        m, mp = spec.split(":")
        for split, ov, idx in (("train", ovr, ix), ("test", ovt, ixt)):
            for n, r in load_boxes(m, mp, split).items():
                if ok_box(m, r) and (idx[n] not in ov or r["acc"] >= ov[idx[n]][1]):
                    ov[idx[n]] = (cells(r["box"]), r["acc"])
    ovr = {i: v[0] for i, v in ovr.items()}; ovt = {i: v[0] for i, v in ovt.items()}
    s = np.full(510, np.nan); choice, final = {}, {}; B = np.zeros((1020, 128, 128), bool)
    STK = None
    if a.stack:
        from src.stack2 import stack_maps
        STK = stack_maps(ovr, ovt)
    s2 = np.full(510, np.nan); choice2, final2 = {}, {}; B2 = np.zeros((1020, 128, 128), bool)   # variant WITH the stacker
    for m in METHODS:
        ok = [c for c in a.cands if c == "snapDF" or json.loads((ROOT / "work/runs" / c / "args.json").read_text())["method"] == m]
        maps = {c: src(c, m) for c in ok}
        for reg in ("F", "S"):
            rf = reg == "F"
            okr = [c for c in ok if not a.restrict or run_regime(c) in ("any", {"F": "full", "S": "small"}[reg]) or run_regime(c).endswith("_proc")]
            sel = np.where((me == m) & (RT == rf))[0]; loc = RT[me == m] == rf
            selt = np.where((mt == m) & (RE == rf))[0]; loct = RE[mt == m] == rf
            _W["G"] = G[sel]; _W["ov"] = {j: miou_image(ovr[i], G[i]) for j, i in enumerate(sel) if i in ovr}
            Fc = Fo[sel]; masks = {k: Fc != k for k in range(5)}; masks["all"] = np.ones(len(sel), bool)

            def tables(cbs):
                _W["X"] = {cb: sum(maps[c][0] for c in cb)[loc] / len(cb) for cb in cbs}
                with Pool(a.procs) as pool:
                    return dict(pool.map(_score_combo, [(cb,) for cb in cbs]))
            tabs = tables([(c,) for c in okr])
            # pairs only among the top-k singles, ranked INSIDE each selection set (inner folds / all) => stays nested
            allowed = {}
            for key, msk in masks.items():
                rank = sorted(okr, key=lambda c: -max(v[msk].mean() for v in tabs[(c,)].values()))
                top = rank[:a.topk] if a.topk else rank
                allowed[key] = {(c,) for c in okr} | (set(itertools.combinations(sorted(top, key=okr.index), 2)) if a.pairs else set())
                for n in MEANS:      # equal-weight mean of the top-n singles (ranked inside the selection set => nested)
                    if len(rank) >= n:
                        allowed[key].add(tuple(sorted(rank[:n], key=okr.index)))
            pairs = sorted(set().union(*allowed.values()) - set(tabs))
            if pairs:
                tabs.update(tables(pairs))
            flat = {(cb, o, t): v for cb, d in tabs.items() for (o, t), v in d.items()}
            combos = list(tabs)
            stk = {}
            if STK is not None:      # fold-dependent source: maps of key k are the only ones used when k is the outer fold
                _W["X"] = {("STACK", key): STK["train"][key][sel] for key in list(range(5)) + ["all"]}
                with Pool(a.procs) as pool:
                    stk = dict(pool.map(_score_combo, [(cb,) for cb in _W["X"]]))
            tmc, tmt = {}, {}        # TOPM: per selection set, the multiset of members of its top-M combos, and its score tables
            if a.topm:
                for key, msk in masks.items():
                    rk = sorted(allowed[key], key=lambda cb: -max(v[msk].mean() for v in tabs[cb].values()))[:a.topm]
                    tmc[key] = tuple(sorted(c for cb in rk for c in cb * (2 // len(cb))))     # a single counts twice, a pair once each
                need = sorted(set(tmc.values()) - set(tabs))
                if need:
                    tabs.update(tables(need))
                tmt = {key: tabs[tmc[key]] for key in masks}
            exp, expcal, topc = {}, {}, {}
            if a.expdec:             # per-image decoding for the top combos of each selection set (calibration inside the set)
                topc = {key: sorted(allowed[key], key=lambda cb: -max(v[masks[key]].mean() for v in tabs[cb].values()))[:a.expdec]
                        for key in masks}
                need = sorted(set().union(*[set(v) for v in topc.values()]))
                _W["X"] = {cb: sum(maps[c][0] for c in cb)[loc] / len(cb) for cb in need}; _W["masks"] = masks; _W["tabs"] = tabs
                with Pool(a.procs) as pool:
                    for cb, o_, c_ in pool.map(_exp_combo, [(cb,) for cb in need]):
                        exp[cb] = o_; expcal[cb] = c_
            for k in range(5):
                inn, out = masks[k], Fc == k
                best = max((c for c in flat if c[0] in allowed[k]), key=lambda c: flat[c][inn].mean())
                s[sel[out]] = flat[best][out]
                choice[f"{m}|{reg}|{k}"] = [list(best[0]), best[1], round(float(best[2]), 3)]
                v2, sc2, ch2 = flat[best][inn].mean(), flat[best], choice[f"{m}|{reg}|{k}"]     # advanced variant: + STACK + EXP
                if stk:
                    tk = stk[("STACK", k)]; bo = max(tk, key=lambda c: tk[c][inn].mean())
                    if tk[bo][inn].mean() > v2:
                        v2, sc2, ch2 = tk[bo][inn].mean(), tk[bo], [["STACK"], bo[0], round(float(bo[1]), 3)]
                for cb in topc.get(k, []):
                    e = exp[cb][k]
                    if e[inn].mean() > v2:
                        v2, sc2, ch2 = e[inn].mean(), e, [list(cb), "EXP", 0.0]
                if tmt:              # TOPM replaces the argmax in the ADV variant
                    bo = max(tmt[k], key=lambda c: tmt[k][c][inn].mean())
                    sc2, ch2 = tmt[k][bo], [list(tmc[k]), bo[0], round(float(bo[1]), 3)]
                s2[sel[out]] = sc2[out]; choice2[f"{m}|{reg}|{k}"] = ch2
            fb = max((c for c in flat if c[0] in allowed["all"]), key=lambda c: flat[c].mean())
            final[f"{m}|{reg}"] = [list(fb[0]), fb[1], float(fb[2])]; final2[f"{m}|{reg}"] = final[f"{m}|{reg}"]
            v2, kind2 = flat[fb].mean(), ("REG",)
            if stk:
                ta = stk[("STACK", "all")]; bo = max(ta, key=lambda c: ta[c].mean())
                if ta[bo].mean() > v2:
                    v2, kind2 = ta[bo].mean(), ("STACK", bo)
            for cb in topc.get("all", []):
                if exp[cb]["all"].mean() > v2:
                    v2, kind2 = exp[cb]["all"].mean(), ("EXP", cb)
            if tmt:
                bo = max(tmt["all"], key=lambda c: tmt["all"][c].mean()); cbm = tmc["all"]; kind2 = ("TOPM",)
                final2[f"{m}|{reg}"] = [list(cbm), bo[0], float(bo[1])]
                if all(maps[c][1] is not None for c in cbm):
                    Xt2 = sum(maps[c][1] for c in cbm)[loct] / len(cbm)
                    for j, i in enumerate(selt):
                        B2[i] = ovt[i].astype(bool) if i in ovt else pp(Xt2[j] >= bo[1], bo[0])
            if kind2[0] == "STACK":
                bo = kind2[1]; final2[f"{m}|{reg}"] = [["STACK"], bo[0], float(bo[1])]
                for j, i in enumerate(selt):
                    B2[i] = ovt[i].astype(bool) if i in ovt else pp(STK["test"][i] >= bo[1], bo[0])
            elif kind2[0] == "EXP":
                cbx = kind2[1]; a_, b_ = expcal[cbx]["all"]; final2[f"{m}|{reg}"] = [list(cbx), "EXP", [a_, b_]]
                if all(maps[c][1] is not None for c in cbx):
                    Xt2 = sum(maps[c][1] for c in cbx)[loct] / len(cbx)
                    with Pool(a.procs) as pool:
                        dm = pool.map(_exp_test, [(Xt2[j], a_, b_) for j in range(len(selt))])
                    for j, i in enumerate(selt):
                        B2[i] = ovt[i].astype(bool) if i in ovt else dm[j]
            cb, o, t = fb
            if all(maps[c][1] is not None for c in cb):
                Xt = sum(maps[c][1] for c in cb)[loct] / len(cb)
                for j, i in enumerate(selt):
                    B[i] = ovt[i].astype(bool) if i in ovt else pp(Xt[j] >= t, o)
                    if kind2[0] == "REG":
                        B2[i] = B[i]
            else:
                print("WARNING no test maps for", cb)
            print(f"  {m}|{reg} n={len(sel)} nested {s[sel].mean():.4f} final {final[f'{m}|{reg}']} ({len(combos)} combos)"
                  + (f" | ADV {s2[sel].mean():.4f} final {final2[f'{m}|{reg}']}" if (stk or exp or tmt) else ""), flush=True)
    rep = summarize(s); rep.update(clean=True, kind="full_nested", cands=a.cands, boxes=a.boxes, thr_box=a.thr_box,
                                   restrict=a.restrict, choices=choice, final=final)
    if STK is not None or a.expdec or a.topm:   # gate (human rule): STACK / EXP count only if gain >= 0.003 AND >= 4/5 outer folds positive
        r2 = summarize(s2); d = [x - y for x, y in zip(r2["folds"], rep["folds"])]; gain = r2["cv_foldmean"] - rep["cv_foldmean"]
        gate = bool(gain >= a.stack_gate and sum(x > 0 for x in d) >= 4)
        info = {"channels": STK["channels"] if STK is not None else None, "expdec": a.expdec, "base_cv": rep["cv_foldmean"], "stack_cv": r2["cv_foldmean"], "gain": gain,
                "fold_diffs": d, "gate": gate, "cells_base": rep["cells"], "cells_stack": r2["cells"], "variants_tried": 1}
        print(f"ADV gate: base {rep['cv_foldmean']:.6f} -> with STACK/EXP {r2['cv_foldmean']:.6f} (gain {gain:+.6f}; fold diffs "
              + " ".join(f"{x:+.4f}" for x in d) + f") => {'KEEP advanced variant' if gate else 'drop (gate not met), base kept'}", flush=True)
        if gate:
            s, B, choice, final = s2, B2, choice2, final2
            rep = r2; rep.update(clean=True, kind="full_nested+stack", cands=a.cands, boxes=a.boxes, thr_box=a.thr_box,
                                 restrict=a.restrict, choices=choice, final=final)
        rep["stack"] = info
    (ROOT / "work/nested" / f"{a.name}.json").write_text(json.dumps(rep, indent=1)); np.save(ROOT / "work/nested" / f"{a.name}.npy", s)
    np.save(ROOT / "work/test" / f"{a.name}_bin.npy", B)
    print(f"{a.name}: nested CV {rep['cv_foldmean']:.6f} ± {rep['cv_std']:.4f} | " + " ".join(f"{x:.4f}" for x in rep["folds"]),
          "| test wm frac", round(float(B.mean()), 3))
    if a.write:
        from src.submit import write
        d = ROOT / "work/archive" / a.name; os.makedirs(d, exist_ok=True)
        pre = d / f"{a.name}.pt"                 # pre-created placeholder (project inode quota full): overwrite in place
        outp = write(B, a.name, f"{rep['cv_foldmean']:.4f}", out=pre if pre.exists() else None)
        if (d / f"{a.name}_test_bin.npy").exists() or not pre.exists():
            np.save(d / f"{a.name}_test_bin.npy", B)
        (d / "spec.json").write_text(json.dumps({"name": a.name, "kind": "full_nested", "final": final, "cands": a.cands,
            "boxes": a.boxes, "thr_box": a.thr_box, "restrict": a.restrict,
            "command": "python -m src.full_nested " + " ".join(sys.argv[1:]),
            "submission_file": str(outp.relative_to(ROOT)), "submission_md5": hashlib.md5(open(outp, "rb").read()).hexdigest()}, indent=1))
        print("archived", outp)


if __name__ == "__main__":
    main()
