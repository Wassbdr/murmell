"""Regime-routed combination: predicted FULL images -> full-regime hole map, predicted SMALL -> small map.
Thresholds per (method, predicted regime), nested in CV; final test thresholds fit on all train.
    python -m src.combine --name C1 --full ps=F1_ps,tm=F1_tm,ss=F1_ss [--small zeros|mate_decoder_feat|<oof exp>] [--save]
"""
import argparse

import numpy as np

from src.data import METHODS, ROOT, masks128, methods, names
from src.eval_cv import THRS, evaluate
from src.metric import miou_batch

SHORT = {"ps": "pixelseal", "tm": "trustmark", "ss": "stegastamp"}


def place_runs(spec):
    """spec 'ps=F1_ps,tm=F1_tm,...' -> (train [510], test [1020]) arrays; missing methods are NaN."""
    tr = np.full((510, 128, 128), np.nan, np.float32); te = np.full((1020, 128, 128), np.nan, np.float32)
    for kv in spec.split(","):
        k, e = kv.split("="); m = SHORT[k]
        d = ROOT / "work" / "runs" / e
        tr[methods() == m] = np.load(d / "train.npy")
        if (d / "test.npy").exists():
            te[methods("test") == m] = np.load(d / "test.npy")
    return tr, te


def load_named(path, key, split):
    z = np.load(path); idx = {n: i for i, n in enumerate(z["names"].tolist())}
    return z[key][[idx[n] for n in names(split)]].astype(np.float32)


def small_maps(kind):
    if kind == "zeros":
        return np.zeros((510, 128, 128), np.float32), np.zeros((1020, 128, 128), np.float32)
    if kind == "snapDF":                                  # frozen copy of teammate decoder_feat (17:58)
        d = ROOT / "work" / "snap_mate"
        return load_named(d / "decoder_feat_train.npz", "maps", "train"), load_named(d / "decoder_feat_test.npz", "maps", "test")
    if kind.startswith("mate_"):
        d = ROOT / "work" / kind[5:]
        return load_named(d / "train.npz", "maps", "train"), load_named(d / "test.npz", "maps", "test")
    return np.load(ROOT / "work/oof" / f"{kind}.npy").astype(np.float32), np.load(ROOT / "work/test" / f"{kind}.npy").astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name"); ap.add_argument("--full"); ap.add_argument("--small", default="zeros")
    ap.add_argument("--regime", default="B0_regime_vec"); ap.add_argument("--save", action="store_true")
    ap.add_argument("--full_fallback", default="ones", help="ones | small: map used on predicted-full images of methods without a full run")
    a = ap.parse_args()
    rtr = np.load(ROOT / "work/oof" / f"{a.regime}.npy"); rte = np.load(ROOT / "work/test" / f"{a.regime}.npy")
    ftr, fte = place_runs(a.full); str_, ste = small_maps(a.small)
    if a.full_fallback == "small":
        ftr = np.where(np.isnan(ftr), str_, ftr); fte = np.where(np.isnan(fte), ste, fte)
    else:
        ftr = np.where(np.isnan(ftr), 1.0, ftr); fte = np.where(np.isnan(fte), 1.0, fte)  # no full map => all-ones
    P = np.where(rtr[:, None, None], ftr, str_); T = np.where(rte[:, None, None], fte, ste)
    me, mt = methods(), methods("test")
    grp = np.array([f"{m}|{'F' if r else 'S'}" for m, r in zip(me, rtr)])
    gte = np.array([f"{m}|{'F' if r else 'S'}" for m, r in zip(mt, rte)])
    res, _ = evaluate(P, name=a.name, groups=grp)
    if a.save:
        G = masks128(); B = np.zeros(T.shape, bool); thr = {}
        for g in sorted(set(grp)):
            s = grp == g
            thr[g] = float(max(THRS, key=lambda t: miou_batch(P[s] >= t, G[s]).mean()))
            B[gte == g] = T[gte == g] >= thr[g]
        np.save(ROOT / "work/oof" / f"{a.name}.npy", P.astype(np.float16))
        np.save(ROOT / "work/test" / f"{a.name}.npy", T.astype(np.float16))
        np.save(ROOT / "work/test" / f"{a.name}_bin.npy", B)
        print("saved", a.name, "test thresholds", thr, "test wm frac", round(float(B.mean()), 3))


if __name__ == "__main__":
    main()
