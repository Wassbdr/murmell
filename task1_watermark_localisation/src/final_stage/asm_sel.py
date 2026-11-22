"""Bundle of a selector3 candidate for the snap stage (after the asm_T02.py): masks from the SEL3 runs written by
`python -m src.selector3 --base BASE --name TAG --out_root work/hunt/runs --write_runs 1`, plus the decoder boxes.
    python src/final_stage/asm_sel.py NAME TAG OUT.npz        (NAME = candidate with work/nested/NAME.{json,npy} and work/test/NAME_bin.npy)"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.getcwd())
from src.apply_boxes import cells, load_boxes
from src.data import METHODS, ROOT, masks128, methods, names
from src.metric import miou_image
SH = {"pixelseal": "ps", "trustmark": "tm", "stegastamp": "ss"}; name, tag, out = sys.argv[1], sys.argv[2], sys.argv[3]
G = masks128().astype(bool); me = methods(); met = methods("test")
rep = json.loads((ROOT / f"work/nested/{name}.json").read_text())


def ovr(split):
    ix = {n: i for i, n in enumerate(names(split))}; ov = {}
    for spec in rep["boxes"]:
        m, mp = spec.split(":")
        for n, r in load_boxes(m, mp, split).items():
            y0, y1, x0, x1 = r["box"]; h, w = y1 - y0, x1 - x0
            if r["acc"] < rep["thr_box"] or (m == "trustmark" and max(h, w) > 2.0 * min(h, w)):
                continue
            if ix[n] not in ov or r["acc"] >= ov[ix[n]][1]:
                ov[ix[n]] = (r["box"], r["acc"])
    return ov


P = np.zeros((510, 128, 128), bool); T = np.zeros((1020, 128, 128), bool)
for m in METHODS:
    P[me == m] = np.load(ROOT / f"work/hunt/runs/{tag}_{SH[m]}/train.npy") >= .5; T[met == m] = np.load(ROOT / f"work/hunt/runs/{tag}_{SH[m]}/test.npy") >= .5
boxed = np.zeros(510, bool); boxedt = np.zeros(1020, bool); bx = np.zeros((510, 4), int); bxt = np.zeros((1020, 4), int)
for i, (b, _) in ovr("train").items():
    P[i] = cells(b).astype(bool); boxed[i] = True; bx[i] = b
for i, (b, _) in ovr("test").items():
    T[i] = cells(b).astype(bool); boxedt[i] = True; bxt[i] = b
s = np.array([miou_image(P[i], G[i]) for i in range(510)]); ref = np.load(ROOT / f"work/nested/{name}.npy"); B = np.load(ROOT / f"work/test/{name}_bin.npy").astype(bool)
print(f"{name}: rebuilt OOF {s.mean():.6f} stored {ref.mean():.6f} max diff {np.abs(s - ref).max():.4f} ({int((np.abs(s - ref) > 1e-4).sum())} images differ) | test identical: {bool((B == T).all())} ({int((B != T).reshape(1020, -1).any(1).sum())} images differ)")
np.savez_compressed(out, P=P, s=s, boxed=boxed, bx=bx, T=B, boxedt=boxedt, bxt=bxt)
