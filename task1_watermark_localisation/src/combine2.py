"""Flexible regime-routed ensemble. For each (method, regime) a list of sources, averaged with weights.
Source names: a run in work/runs/<exp> (maps for its method only), 'snapDF' (frozen teammate decoder_feat), 'ones', 'zeros'.
Spec example (JSON):  {"pixelseal": {"F": {"snapDF": 1}, "S": {"snapDF": 1, "S1_ps_amp4": 1}}, ...}
    python -m src.combine2 --name C4 --spec '<json>' [--save]
    python -m src.combine2 --search          (tries each candidate source and pairwise averages per cell, prints CV)
Thresholds per (method, predicted regime), nested in CV (src.eval_cv).
"""
import argparse
import itertools
import json
from functools import lru_cache

import numpy as np

from src.combine import load_named
from src.data import METHODS, ROOT, masks128, methods
from src.eval_cv import THRS, evaluate
from src.metric import miou_batch

RT = np.load(ROOT / "work/oof/B0_regime_vec.npy"); RE = np.load(ROOT / "work/test/B0_regime_vec.npy")


@lru_cache(None)
def src(name, method):
    """(train maps of that method's 170 images, test maps of its 340 images)."""
    mtr, mte = methods() == method, methods("test") == method
    if name in ("ones", "zeros"):
        v = 1.0 if name == "ones" else 0.0
        return np.full((mtr.sum(), 128, 128), v, np.float32), np.full((mte.sum(), 128, 128), v, np.float32)
    if name == "snapDF":
        d = ROOT / "work/snap_mate"
        return (load_named(d / "decoder_feat_train.npz", "maps", "train")[mtr],
                load_named(d / "decoder_feat_test.npz", "maps", "test")[mte])
    d = ROOT / "work/runs" / name
    te = np.load(d / "test.npy").astype(np.float32) if (d / "test.npy").exists() else None
    return np.load(d / "train.npy").astype(np.float32), te


def build(spec):
    P = np.zeros((510, 128, 128), np.float32); T = np.zeros((1020, 128, 128), np.float32)
    for m in METHODS:
        mtr, mte = methods() == m, methods("test") == m
        for reg, rflag in (("F", True), ("S", False)):
            ws = spec[m][reg]; tot = sum(ws.values())
            a = sum(w * src(n, m)[0] for n, w in ws.items()) / tot
            b = None
            if all(src(n, m)[1] is not None for n in ws):
                b = sum(w * src(n, m)[1] for n, w in ws.items()) / tot
            sel = RT[mtr] == rflag
            idx = np.where(mtr)[0][sel]; P[idx] = a[sel]
            if b is not None:
                selt = RE[mte] == rflag; T[np.where(mte)[0][selt]] = b[selt]
    return P, T


def groups(split="train"):
    r = RT if split == "train" else RE
    return np.array([f"{m}|{'F' if x else 'S'}" for m, x in zip(methods(split), r)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="tmp"); ap.add_argument("--spec"); ap.add_argument("--save", action="store_true")
    ap.add_argument("--search", nargs="*", help="candidate sources")
    a = ap.parse_args()
    if a.search is not None:
        cands = a.search + ["snapDF"]
        G = masks128(); F = None
        from src.data import folds
        F = folds()
        for m in METHODS:
            for reg, rflag in (("F", True), ("S", False)):
                sel = (methods() == m) & (RT == rflag); loc = RT[methods() == m] == rflag
                rows = []
                ok = [c for c in cands if (ROOT / "work/runs" / c / "train.npy").exists() or c == "snapDF"]
                ok = [c for c in ok if c == "snapDF" or json.loads((ROOT / "work/runs" / c / "args.json").read_text())["method"] == m]
                combos = [(c,) for c in ok] + list(itertools.combinations(ok, 2))
                for cb in combos:
                    X = sum(src(c, m)[0] for c in cb)[loc] / len(cb)
                    # nested threshold within this cell
                    tab = {t: miou_batch(X >= t, G[sel]) for t in THRS}; s = np.zeros(sel.sum())
                    for f in range(5):
                        tr, te = F[sel] != f, F[sel] == f
                        t = max(THRS, key=lambda t: tab[t][tr].mean()); s[te] = tab[t][te]
                    rows.append((s.mean(), "+".join(cb)))
                rows.sort(reverse=True)
                print(f"{m:10s} {reg} n={sel.sum()} | " + " | ".join(f"{n} {v:.4f}" for v, n in rows[:5]), flush=True)
        return
    spec = json.loads(a.spec)
    P, T = build(spec)
    evaluate(P, name=a.name, groups=groups())
    if a.save:
        G = masks128(); g, gt = groups(), groups("test"); B = np.zeros(T.shape, bool); thr = {}
        for k in sorted(set(g)):
            s = g == k
            thr[k] = float(max(THRS, key=lambda t: miou_batch(P[s] >= t, G[s]).mean()))
            B[gt == k] = T[gt == k] >= thr[k]
        np.save(ROOT / "work/oof" / f"{a.name}.npy", P.astype(np.float16)); np.save(ROOT / "work/test" / f"{a.name}.npy", T.astype(np.float16))
        np.save(ROOT / "work/test" / f"{a.name}_bin.npy", B)
        (ROOT / "work/oof" / f"{a.name}.spec.json").write_text(json.dumps(spec))
        print("saved", a.name, {k: round(v, 2) for k, v in thr.items()}, "test wm frac", round(float(B.mean()), 3))


if __name__ == "__main__":
    main()
