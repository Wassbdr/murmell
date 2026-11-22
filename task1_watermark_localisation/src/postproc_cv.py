"""Structural post-processing chosen in fully nested CV (per method x predicted-regime cell, option AND threshold chosen
on the 4 inner folds). Base = OOF probability maps of a candidate (work/archive/<base>/<base>_oof.npy) + optional box
override (apply_boxes rule, applied after post-processing, as in the test pipeline).
Options: raw | largest component | bbox of largest component | drop components < 16 cells | fill holes < 16 cells
         | drop comps<16 + fill holes<16
    python -m src.postproc_cv --base C4r --name C6pp --boxes trustmark:S1_tm
"""
import argparse
import json

import numpy as np
from scipy import ndimage as ndi

from src.apply_boxes import cells, load_boxes
from src.combine2 import RE, RT
from src.data import METHODS, ROOT, folds, masks128, methods, names
from src.eval_cv import THRS
from src.metric import miou_batch
from src.nested_cv import summarize


def pp(b, opt):
    if opt == "raw" or not b.any():
        return b
    if opt in ("largest", "bbox"):
        lab, n = ndi.label(b)
        if n == 0:
            return b
        k = 1 + int(np.argmax(ndi.sum(b, lab, range(1, n + 1)))); c = lab == k
        if opt == "largest":
            return c
        ys, xs = np.where(c); o = np.zeros_like(b); o[ys.min():ys.max() + 1, xs.min():xs.max() + 1] = True; return o
    o = b.copy()
    if opt in ("dropsmall", "both"):
        lab, n = ndi.label(o); sz = ndi.sum(o, lab, range(1, n + 1))
        for j, v in enumerate(sz, 1):
            if v < 16:
                o[lab == j] = False
    if opt in ("fillsmall", "both"):
        lab, n = ndi.label(~o); sz = ndi.sum(~o, lab, range(1, n + 1))
        for j, v in enumerate(sz, 1):
            if v < 16:
                o[lab == j] = True
    return o


OPTS = ("raw", "largest", "bbox", "dropsmall", "fillsmall", "both")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--base"); ap.add_argument("--name"); ap.add_argument("--boxes", nargs="*", default=[])
    ap.add_argument("--thr_box", type=float, default=0.8); ap.add_argument("--relmax", type=int, default=0)
    ap.add_argument("--write", type=int, default=0)
    a = ap.parse_args()
    P = np.load(ROOT / "work/archive" / a.base / f"{a.base}_oof.npy").astype(np.float32)
    T = np.load(ROOT / "work/archive" / a.base / f"{a.base}_test.npy").astype(np.float32)
    G, Fo, me = masks128(), folds(), methods()
    ix = {n: i for i, n in enumerate(names())}; ixt = {n: i for i, n in enumerate(names("test"))}
    ovr, ovt = {}, {}
    for spec in a.boxes:
        m, mp = spec.split(":")
        ovr.update({ix[n]: cells(r["box"]) for n, r in load_boxes(m, mp, "train").items() if r["acc"] >= a.thr_box})
        ovt.update({ixt[n]: cells(r["box"]) for n, r in load_boxes(m, mp, "test").items() if r["acc"] >= a.thr_box})
    s = np.zeros(510); choice = {}; final = {}
    for m in METHODS:
        for reg in ("F", "S"):
            sel = np.where((me == m) & (RT == (reg == "F")))[0]
            tab = {}
            modes = [("abs", t) for t in THRS] + ([("rel", t) for t in THRS] if a.relmax else [])
            for o in OPTS:
                for mode, t in modes:
                    pred = np.stack([pp(P[i] >= (t * P[i].max() if mode == "rel" else t), o) for i in sel])
                    for jj, i in enumerate(sel):
                        if i in ovr:
                            pred[jj] = ovr[i]
                    tab[(o, mode, t)] = miou_batch(pred, G[sel])
            for k in range(5):
                inn, out = Fo[sel] != k, Fo[sel] == k
                best = max(tab, key=lambda c: tab[c][inn].mean())
                s[sel[out]] = tab[best][out]; choice[f"{m}|{reg}|{k}"] = [best[0], best[1], round(float(best[2]), 2)]
            final[f"{m}|{reg}"] = max(tab, key=lambda c: tab[c].mean())
    rep = summarize(s); rep.update(clean=True, kind="postproc", base=a.base, boxes=a.boxes, choices=choice,
                                   final={k: [v[0], v[1], float(v[2])] for k, v in final.items()})
    (ROOT / "work/nested" / f"{a.name}.json").write_text(json.dumps(rep, indent=1)); np.save(ROOT / "work/nested" / f"{a.name}.npy", s)
    print(f"{a.name}: nested CV {rep['cv_foldmean']:.6f} ± {rep['cv_std']:.4f} | " + " ".join(f"{x:.4f}" for x in rep["folds"]))
    for kk, v in rep["cells"].items():
        print(f"  {kk:14s} n={v['n']:3d} {v['mean']:.4f} final choice {rep['final'][kk]}")
    # test masks with the choice made on all train
    mt = methods("test"); B = np.zeros(T.shape, bool)
    for m in METHODS:
        for reg in ("F", "S"):
            o, mode, t = final[f"{m}|{reg}"]
            for i in np.where((mt == m) & (RE == (reg == "F")))[0]:
                B[i] = ovt[i].astype(bool) if i in ovt else pp(T[i] >= (t * T[i].max() if mode == "rel" else t), o)
    np.save(ROOT / "work/test" / f"{a.name}_bin.npy", B); print("test wm frac", round(float(B.mean()), 3))
    if a.write:      # submission file + archive (autosubmit candidate)
        import hashlib, os, sys
        from src.submit import write
        out = write(B, a.name, f"{rep['cv_foldmean']:.4f}"); d = ROOT / "work/archive" / a.name; os.makedirs(d, exist_ok=True)
        np.save(d / f"{a.name}_test_bin.npy", B)
        (d / "spec.json").write_text(json.dumps({"name": a.name, "base": a.base, "base_spec": f"work/archive/{a.base}/spec.json",
            "boxes": a.boxes, "thr_box": a.thr_box, "relmax": a.relmax, "final": rep["final"],
            "command": "python -m src.postproc_cv " + " ".join(sys.argv[1:]),
            "submission_file": str(out.relative_to(ROOT)), "submission_md5": hashlib.md5(open(out, "rb").read()).hexdigest()}, indent=1))
        print("archived", out)


if __name__ == "__main__":
    main()
