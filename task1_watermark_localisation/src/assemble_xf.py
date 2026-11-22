"""Assemble fold-excluded runs <exp>_xf0..4 into <exp>_xfall: train map of each image comes from the model whose shape
libraries excluded that image's fold (=> no shape leakage in ANY cell); test map = mean of the 5 models.
    python -m src.assemble_xf F1_ss F1_tm_long
"""
import json
import sys

import numpy as np

from src.data import ROOT, folds, methods

for exp in sys.argv[1:]:
    a = json.loads((ROOT / "work/runs" / f"{exp}_xf0" / "args.json").read_text())
    F = folds()[methods() == a["method"]]
    tr = np.load(ROOT / "work/runs" / f"{exp}_xf0" / "train.npy").copy(); te = []
    for k in range(5):
        d = ROOT / "work/runs" / f"{exp}_xf{k}"
        tr[F == k] = np.load(d / "train.npy")[F == k]; te.append(np.load(d / "test.npy").astype(np.float32))
    out = ROOT / "work/runs" / f"{exp}_xfall"; out.mkdir(parents=True, exist_ok=True)
    np.save(out / "train.npy", tr); np.save(out / "test.npy", np.mean(te, 0).astype(np.float16))
    a.update(exp=f"{exp}_xfall", fold_excluded=True); (out / "args.json").write_text(json.dumps(a))
    print("assembled", out)
