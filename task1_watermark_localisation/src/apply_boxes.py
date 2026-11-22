"""Candidate = base candidate + decoder-ranked box override on predicted-small images (GT-free rule: bit acc >= thr).
Nested CV: per-image held-out scores of the base (work/nested/<base>.npy) replaced, for overridden train images, by the
mIoU of the box mask (the rule has no parameter fitted on labels; thr fixed a priori).
    python -m src.apply_boxes --base C4r --name C5 --boxes trustmark:S1_tm [stegastamp:S1_ss] --thr 0.8
Writes work/nested/<name>.{npy,json}, work/test/<name>_bin.npy, submissions/<HHMM>_<name>_cv<cv>.pt, work/archive/<name>/spec.json.
"""
import argparse
import hashlib
import json
import os

import numpy as np

from src.data import ROOT, masks128, names
from src.metric import miou_image
from src.nested_cv import summarize
from src.submit import write


def cells(b):
    m = np.zeros((512, 512), np.float32); m[b[0]:b[1], b[2]:b[3]] = 1
    return (m.reshape(128, 4, 128, 4).mean((1, 3)) >= 0.5).astype(np.uint8)


def load_boxes(method, mp, split):
    d = {}
    for f in sorted((ROOT / "work/boxes").glob(f"{method}_{split}_{mp}_[0-9]*of*.json")):
        if f.stat().st_size > 0:                 # empty = pre-created placeholder (inode quota), not a result
            d.update(json.loads(f.read_text()))
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base"); ap.add_argument("--name"); ap.add_argument("--boxes", nargs="+"); ap.add_argument("--thr", type=float, default=0.8)
    a = ap.parse_args()
    G = masks128(); s = np.load(ROOT / "work/nested" / f"{a.base}.npy").copy()
    B = np.load(ROOT / "work/archive" / a.base / f"{a.base}_test_bin.npy").copy()
    tr_idx = {n: i for i, n in enumerate(names())}; te_idx = {n: i for i, n in enumerate(names("test"))}
    used = {}
    for spec in a.boxes:
        meth, mp = spec.split(":")
        bt = load_boxes(meth, mp, "train"); be = load_boxes(meth, mp, "test")
        ntr = nte = 0
        for n, r in bt.items():
            if r["acc"] >= a.thr:
                s[tr_idx[n]] = miou_image(cells(r["box"]), G[tr_idx[n]]); ntr += 1
        for n, r in be.items():
            if r["acc"] >= a.thr:
                B[te_idx[n]] = cells(r["box"]).astype(bool); nte += 1
        used[spec] = {"train_overridden": ntr, "train_candidates": len(bt), "test_overridden": nte, "test_candidates": len(be)}
    rep = summarize(s); rep.update(base=a.base, boxes=a.boxes, thr=a.thr, used=used, clean=True)
    (ROOT / "work/nested" / f"{a.name}.json").write_text(json.dumps(rep, indent=1)); np.save(ROOT / "work/nested" / f"{a.name}.npy", s)
    np.save(ROOT / "work/test" / f"{a.name}_bin.npy", B)
    print(f"{a.name}: nested CV {rep['cv_foldmean']:.6f} ± {rep['cv_std']:.4f} | folds " + " ".join(f"{x:.4f}" for x in rep["folds"]), used)
    for k, v in rep["cells"].items():
        print(f"  {k:14s} n={v['n']:3d} {v['mean']:.4f}")
    out = write(B, a.name, f"{rep['cv_foldmean']:.4f}")
    d = ROOT / "work/archive" / a.name; os.makedirs(d, exist_ok=True)
    np.save(d / f"{a.name}_test_bin.npy", B)
    (d / "spec.json").write_text(json.dumps({"name": a.name, "base": a.base, "base_spec": f"work/archive/{a.base}/spec.json",
        "boxes": a.boxes, "rule": f"override predicted-small images whose best decoder-ranked box has bit acc >= {a.thr}",
        "command": f"python -m src.apply_boxes --base {a.base} --name {a.name} --boxes {' '.join(a.boxes)} --thr {a.thr}",
        "submission_file": str(out.relative_to(ROOT)), "submission_md5": hashlib.md5(open(out, "rb").read()).hexdigest()}, indent=1))
    print("archived", out)


if __name__ == "__main__":
    main()
