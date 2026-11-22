"""Fully nested CV of a regime-routed ensemble.
For each outer fold k and each cell (method, PREDICTED regime, as at test time): the source combination (singles and
equal-weight pairs from the candidate pool, or a fixed spec) AND the threshold are chosen using ONLY the other 4 folds
(criterion: mean per-image mIoU on those 4 folds at their best threshold); the choice is then scored on fold k.
Outputs per-image held-out scores and a JSON report.
    python -m src.nested_cv --name C4 --cands F1_ss F1_tm ... snapDF
    python -m src.nested_cv --name C3 --fixed '<combine2 spec json>'
    python -m src.nested_cv --compare C4 C3
Reports: work/nested/<name>.json (+ .npy per-image scores, per-fold choices).
"""
import argparse
import itertools
import json

import numpy as np

from src.combine2 import RT, src
from src.data import METHODS, ROOT, folds, masks128, methods
from src.eval_cv import THRS
from src.metric import miou_batch

OUT = ROOT / "work" / "nested"


def cell_tables(m, reg, combos):
    """per combo: dict thr -> per-image scores (images of the cell, in train order)."""
    sel = (methods() == m) & (RT == (reg == "F")); loc = RT[methods() == m] == (reg == "F")
    G = masks128()[sel]; tabs = {}
    for cb in combos:
        X = sum(w * src(c, m)[0] for c, w in cb) / sum(w for _, w in cb)
        X = X[loc]
        tabs[cb] = {t: miou_batch(X >= t, G) for t in THRS}
    return sel, tabs


def _plain_obj_ancestor(a):
    """object items (shapes = full-regime holes) anywhere in the training lineage without fold exclusion.  Descendants
    inherit it: finetune_real and rescale_ft keep their parent only in base_args (the top-level flag is lost)."""
    if a.get("obj_items", 0) and not a.get("fold_excluded"):
        return True
    d = a
    while isinstance(d.get("base_args"), dict):
        if d.get("base_fold_clean"):                        # fine-tuned from <base>_xf{k}: base_args describes the plain twin
            return False
        d = d["base_args"]
        if d.get("obj_items", 0) and not d.get("fold_excluded") and d.get("exclude_fold", -1) < 0:
            return True
    return False


def run_regime(c):
    """training regime of a run ('full' for runs created before the --regime flag existed); snapDF = 'any'."""
    if c == "snapDF":
        return "any"
    a = json.loads((ROOT / "work/runs" / c / "args.json").read_text())
    if _plain_obj_ancestor(a):
        return "leaky"                                                  # full model that saw the full-regime hole shapes of all folds: no cell
    if a.get("fold_excluded") and not ("bs_real" in a and not a.get("base_fold_clean")):      # bs_real = src.finetune_real, whatever the run name
        return "any"                                                    # shapes of the scored fold never seen
    # FT_* of a plain synthetic base: the fine-tuning is out-of-fold, but the BASE saw the shape library of all folds
    # => same restriction as its base (full-trained only in F cells, small-trained only in S cells)
    r = a.get("regime", "full"); r = "small" if r in ("small2", "small3") else r                 # small2: same shape source (full holes)
    return r + ("_proc" if a.get("shape_lib") == "none" else "")      # procedural shapes only => no shape leakage


def run(name, cand=None, fixed=None, restrict=False):
    F = folds(); scores = np.full(510, np.nan); choices = {}
    for m in METHODS:
        for reg in ("F", "S"):
            if fixed is not None:
                combos = [tuple(sorted(fixed[m][reg].items()))]
            else:
                ok = [c for c in cand if c == "snapDF" or json.loads((ROOT / "work/runs" / c / "args.json").read_text())["method"] == m]
                if restrict:   # no shape leakage: full-trained models only in F cells, small-trained only in S cells
                    ok = [c for c in ok if run_regime(c) in ("any", {"F": "full", "S": "small"}[reg]) or run_regime(c).endswith("_proc")]
                combos = [((c, 1),) for c in ok] + [((a, 1), (b, 1)) for a, b in itertools.combinations(ok, 2)]
            sel, tabs = cell_tables(m, reg, combos)
            Fc = F[sel]; idx = np.where(sel)[0]
            for k in range(5):
                inner, outer = Fc != k, Fc == k
                best = None
                for cb, tab in tabs.items():
                    t = max(THRS, key=lambda t: tab[t][inner].mean()); v = tab[t][inner].mean()
                    if best is None or v > best[0]:
                        best = (v, cb, t)
                _, cb, t = best
                scores[idx[outer]] = tabs[cb][t][outer]
                choices[f"{m}|{reg}|fold{k}"] = {"combo": [c for c, _ in cb], "thr": round(float(t), 3), "inner": round(float(best[0]), 5)}
    OUT.mkdir(parents=True, exist_ok=True)
    np.save(OUT / f"{name}.npy", scores)
    rep = summarize(scores); rep["choices"] = choices; rep["mode"] = "fixed" if fixed is not None else {"cands": cand, "restrict": restrict}
    (OUT / f"{name}.json").write_text(json.dumps(rep, indent=1))
    return rep


def summarize(s):
    F = folds(); me = methods(); rg = RT
    fm = [float(s[F == k].mean()) for k in range(5)]
    cells = {}
    for m in METHODS:
        for reg in ("F", "S"):
            c = (me == m) & (rg == (reg == "F"))
            cells[f"{m}|{reg}"] = {"n": int(c.sum()), "mean": float(s[c].mean())}
    return {"cv_foldmean": float(np.mean(fm)), "cv_std": float(np.std(fm)), "image_mean": float(s.mean()), "folds": fm, "cells": cells}


def compare(a, b):
    sa, sb = np.load(OUT / f"{a}.npy"), np.load(OUT / f"{b}.npy"); F = folds()
    ra, rb = summarize(sa), summarize(sb)
    d = [ra["folds"][k] - rb["folds"][k] for k in range(5)]
    cell_d = {k: ra["cells"][k]["mean"] - rb["cells"][k]["mean"] for k in ra["cells"]}
    gain = ra["cv_foldmean"] - rb["cv_foldmean"]
    res = {"cand": a, "ref": b, "gain": gain, "fold_diffs": d, "n_pos_folds": int(sum(x > 0 for x in d)),
           "cell_diffs": cell_d, "worst_cell": min(cell_d.values()),
           "gate": bool(gain >= 0.005 and sum(x > 0 for x in d) >= 4 and min(cell_d.values()) >= -0.020)}
    print(f"nested CV {a}: {ra['cv_foldmean']:.6f} ± {ra['cv_std']:.4f} | {b}: {rb['cv_foldmean']:.6f} ± {rb['cv_std']:.4f}")
    print(f"global gain (unrounded) {gain:.6f} | paired fold diffs " + " ".join(f"{x:+.5f}" for x in d) + f" | positive {res['n_pos_folds']}/5")
    print("cell | n | " + a + " | " + b + " | diff")
    for k in ra["cells"]:
        print(f"{k:14s} | {ra['cells'][k]['n']:3d} | {ra['cells'][k]['mean']:.4f} | {rb['cells'][k]['mean']:.4f} | {cell_d[k]:+.4f}")
    print("GATE (gain>=0.005, >=4/5 folds, no cell < -0.020):", res["gate"])
    (OUT / f"compare_{a}_vs_{b}.json").write_text(json.dumps(res, indent=1))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name"); ap.add_argument("--cands", nargs="*"); ap.add_argument("--fixed")
    ap.add_argument("--compare", nargs=2); ap.add_argument("--restrict", action="store_true")
    a = ap.parse_args()
    if a.compare:
        compare(*a.compare); return
    rep = run(a.name, cand=a.cands, fixed=json.loads(a.fixed) if a.fixed else None, restrict=a.restrict)
    print(f"{a.name}: nested CV {rep['cv_foldmean']:.6f} ± {rep['cv_std']:.4f} folds " + " ".join(f"{x:.4f}" for x in rep["folds"]))


if __name__ == "__main__":
    main()
