"""CV evaluation of an OOF probability map (work/oof/<exp>.npy, float [510,128,128], train metadata order).
Decoding: threshold per method, tuned NESTED (on the 4 other folds) -> honest fold scores.
Reports mean +- std over the 5 folds, per method, per regime, overall; also the score at a fixed 0.5 threshold.
    python -m src.eval_cv <exp> [<exp2> ...]   |   python -m src.eval_cv --path some.npy
"""
import argparse
import sys

import numpy as np

from src.data import METHODS, ROOT, folds, masks128, methods, regime
from src.metric import miou_batch

THRS = np.concatenate([[0.02, 0.05], np.linspace(0.1, 0.9, 17), [0.95, 0.98]])


def per_image_at(P, G, t):
    return miou_batch(P >= t, G)


def evaluate(P, verbose=True, name="", groups=None):
    """groups: optional array of group labels (e.g. method|predicted-regime) for the nested thresholds; default = method."""
    P = np.asarray(P, np.float32); G = masks128(); F = folds(); me = methods(); rg = regime()
    grp = me if groups is None else np.asarray(groups)
    table = {t: per_image_at(P, G, t) for t in THRS}               # 510 per-image scores per threshold
    s_nested = np.zeros(len(P)); thr_used = {}
    for m in sorted(set(grp)):
        for f in range(5):
            tr = (grp == m) & (F != f); te = (grp == m) & (F == f)
            if not te.any():
                continue
            t = max(THRS, key=lambda t: table[t][tr].mean())
            s_nested[te] = table[t][te]; thr_used.setdefault(m, []).append(round(float(t), 2))
    fixed = table[THRS[np.argmin(abs(THRS - 0.5))]]
    res = {}

    def fold_stats(s, mask):
        v = [s[mask & (F == f)].mean() for f in range(5)]
        return float(np.mean(v)), float(np.std(v))

    all_ = np.ones(len(P), bool)
    res["all"] = fold_stats(s_nested, all_)
    for m in METHODS:
        res[m] = fold_stats(s_nested, me == m)
    res["small"] = fold_stats(s_nested, ~rg); res["full"] = fold_stats(s_nested, rg)
    res["fixed0.5"] = fold_stats(fixed, all_)
    if verbose:
        print(f"[{name}] nested per-{'group' if groups is not None else 'method'} thr | ALL {res['all'][0]:.4f}±{res['all'][1]:.4f} | "
              + " ".join(f"{m[:2].upper()} {res[m][0]:.4f}±{res[m][1]:.4f}" for m in METHODS)
              + f" | small {res['small'][0]:.4f} full {res['full'][0]:.4f} | @0.5 {res['fixed0.5'][0]:.4f} | thr {thr_used}")
    return res, s_nested


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("exps", nargs="*"); ap.add_argument("--path", nargs="*", default=[])
    a = ap.parse_args()
    for e in a.exps:
        evaluate(np.load(ROOT / "work" / "oof" / f"{e}.npy"), name=e)
    for p in a.path:
        evaluate(np.load(p), name=p)
    if not a.exps and not a.path:
        sys.exit("give exp ids or --path")


if __name__ == "__main__":
    main()
