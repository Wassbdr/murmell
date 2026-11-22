"""B0: regime classifier from global decode features (teammate's work/global: bitacc, agree) -> all-1 / all-0 map.
Per method, a 1-D threshold on one feature, fitted on the 4 training folds (nested), reported out-of-fold.
Also converts teammate maps (work/<dir>/train.npz) to the shared OOF format for comparison.
    python -m src.regime_b0
"""
import numpy as np

from src.data import METHODS, ROOT, folds, methods, names, regime
from src.eval_cv import evaluate


def load_named(path, key, split="train"):
    z = np.load(path); idx = {n: i for i, n in enumerate(z["names"].tolist())}
    return z[key][[idx[n] for n in names(split)]]


def fit_thr(x, y):
    c = np.unique(x); c = (c[1:] + c[:-1]) / 2 if len(c) > 1 else c
    acc = [((x > t) == y).mean() for t in c]
    return c[int(np.argmax(acc))]


def regime_oof(feat_tr, feat_te, me_tr, me_te, y, F):
    """returns OOF regime prediction on train and test prediction (thresholds fit on all train)."""
    oof = np.zeros(len(y), bool); te = np.zeros(len(feat_te), bool)
    for m in METHODS:
        for f in range(5):
            tr = (me_tr == m) & (F != f); ev = (me_tr == m) & (F == f)
            oof[ev] = feat_tr[ev] > fit_thr(feat_tr[tr], y[tr])
        s = me_te == m
        te[s] = feat_te[s] > fit_thr(feat_tr[me_tr == m], y[me_tr == m])
    return oof, te


def main():
    y, F, me, mt = regime(), folds(), methods(), methods("test")
    for key in ("bitacc", "agree"):
        a = load_named(ROOT / "work/global/train.npz", key); b = load_named(ROOT / "work/global/test.npz", key, "test")
        oof, te = regime_oof(a, b, me, mt, y, F)
        print(f"regime from {key}: OOF acc all {(oof == y).mean():.3f} | "
              + " ".join(f"{m} {(oof == y)[me == m].mean():.3f}" for m in METHODS)
              + f" | test full-frac {te.mean():.3f} (" + " ".join(f"{m[:2]} {te[mt == m].mean():.2f}" for m in METHODS) + ")")
        P = np.broadcast_to(oof[:, None, None], (510, 128, 128)).astype(np.float16)
        evaluate(P, name=f"B0_regime_{key}")
        if key == "agree":
            np.save(ROOT / "work/oof/B0_regime.npy", P)
            np.save(ROOT / "work/test/B0_regime.npy", np.broadcast_to(te[:, None, None], (1020, 128, 128)).astype(np.float16))
            np.save(ROOT / "work/oof/B0_regime_vec.npy", oof); np.save(ROOT / "work/test/B0_regime_vec.npy", te)
    # oracle regime -> const
    evaluate(np.broadcast_to(y[:, None, None], (510, 128, 128)).astype(np.float16), name="ORACLE_regime_const")
    # teammate maps on my CV (their train maps are OOF per their README)
    for d in ("decoder_feat", "decoder_perpos"):
        evaluate(load_named(ROOT / f"work/{d}/train.npz", "maps"), name=f"mate_{d}")


if __name__ == "__main__":
    main()
