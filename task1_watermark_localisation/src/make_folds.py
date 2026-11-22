"""5 folds, seed 0, stratified by method x regime x area tercile (within method x regime). Writes work/folds.json."""
import json

import numpy as np

from src.data import ROOT, masks128, methods, names, regime


def main():
    n, me, rg, area = names(), methods(), regime(), masks128().reshape(510, -1).mean(1)
    rng = np.random.RandomState(0); fold = np.zeros(len(n), int)
    for m in set(me):
        for r in (False, True):
            idx = np.where((me == m) & (rg == r))[0]
            idx = idx[np.argsort(area[idx], kind="stable")]
            for t in np.array_split(idx, 3):                         # area terciles
                t = t[rng.permutation(len(t))]
                off = rng.randint(5)
                fold[t] = (np.arange(len(t)) + off) % 5
    out = {"seed": 0, "strata": "method x regime x area tercile", "fold": {k: int(f) for k, f in zip(n, fold)}}
    (ROOT / "work" / "folds.json").write_text(json.dumps(out))
    for f in range(5):
        s = fold == f
        print(f, s.sum(), {m: int((s & (me == m)).sum()) for m in sorted(set(me))}, "full", int((s & rg).sum()))


if __name__ == "__main__":
    main()
