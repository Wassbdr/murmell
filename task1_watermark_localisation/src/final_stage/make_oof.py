"""Bundle of a full_nested candidate for src/final_stage/snap_stage.py: out-of-fold train masks P, per-image nested scores s, test masks
T and the decoder boxes the pipeline applied (same rule as src.full_nested).  python src/final_stage/make_oof.py <NAME> <out.npz>"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.getcwd())
from src.apply_boxes import cells, load_boxes
from src.data import ROOT, masks128, names
from src.loss_analysis import rebuild
from src.metric import miou_image

name, out = sys.argv[1], sys.argv[2]
P, s, boxed0, rep = rebuild(name); P = np.asarray(P).astype(bool); G = masks128().astype(bool)
ref = np.load(ROOT / f"work/nested/{name}.npy"); T = np.load(ROOT / f"work/test/{name}_bin.npy").astype(bool)


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


boxed = np.zeros(510, bool); bx = np.zeros((510, 4), int); boxedt = np.zeros(1020, bool); bxt = np.zeros((1020, 4), int)
for i, (b, _) in ovr("train").items():
    boxed[i] = True; bx[i] = b
for i, (b, _) in ovr("test").items():
    boxedt[i] = True; bxt[i] = b
sc = np.array([miou_image(P[i], G[i]) for i in range(510)])
okb = all((P[i] == cells(bx[i]).astype(bool)).all() for i in np.where(boxed)[0]) and all((T[i] == cells(bxt[i]).astype(bool)).all() for i in np.where(boxedt)[0])
print(f"{name}: rebuilt OOF {sc.mean():.6f} stored {ref.mean():.6f} max diff {np.abs(sc - ref).max():.2e} | boxed train {int(boxed.sum())} test {int(boxedt.sum())} | boxed masks equal boxes: {okb}")
np.savez_compressed(out, P=P, s=ref, boxed=boxed, bx=bx, T=T, boxedt=boxedt, bxt=bxt)
