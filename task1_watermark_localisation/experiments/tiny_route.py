"""tiny_route (Sun 10:40): use a tiny-hole specialist map only on the full-regime images whose STANDARD mask has a small clean
share (label-free routing).  For each method: standard chain = masks + nested scores of the current best full-cell chain;
specialist = mean of the RH_* runs (optionally averaged with the standard map).  (tau, source, post-proc, threshold) are chosen
in inner folds.  Writes DIR/TAG_train_scores.npy and DIR/TAG_test_bin.npy (TAG = Ftiny), a chain for src/final_stage/chains2.py.
    python experiments/tiny_route.py --dir DIR
"""
import argparse, os, sys
import numpy as np
sys.path.insert(0, os.getcwd())
from src.combine2 import RE, RT
from src.data import METHODS, ROOT, folds, masks128, methods
from src.eval_cv import THRS
from src.metric import miou_batch, miou_image
from src.postproc_cv import OPTS, pp

ap = argparse.ArgumentParser(); ap.add_argument("--dir", required=True); a = ap.parse_args(); O = a.dir
G = masks128().astype(bool); Fo = folds(); me = methods(); met = methods("test")
CAND = [(o, float(t)) for o in OPTS for t in THRS]; TAUS = [0.0, 0.02, 0.03, 0.05, 0.08]
SPEC = {"pixelseal": ("D7_ps_Fsub", ["RH_D5_ps_noamp_tta", "RH_D5_ps_amp4_tta"]), "trustmark": ("D7_tm_Fens", ["RH_RF4_D2_tm_32k_tta", "RH_RF4_D2_tm_tta"])}
STD = {"pixelseal": "Fsub", "trustmark": "T02"}                      # standard full-cell chain of B21
s_out = np.load("work/nested/B21.npy").copy(); T_out = np.load("work/test/B21_bin.npy").astype(bool).copy()
T02P = np.load("work/hunt/T02_oof.npz")["P"].astype(bool)


def nested_std(run, idx, tidx, sel, tsel):
    X = np.load(ROOT / "work/runs" / run / "train.npy").astype(np.float32)[sel]; Xt = np.load(ROOT / "work/runs" / run / "test.npy").astype(np.float32)[tsel]
    S = np.zeros((len(idx), len(CAND))); M = np.zeros((len(CAND), len(idx), 128, 128), bool)
    for c, (o, t) in enumerate(CAND):
        M[c] = np.stack([pp(x >= t, o) for x in X]); S[:, c] = miou_batch(M[c], G[idx])
    s = np.zeros(len(idx)); Pm = np.zeros((len(idx), 128, 128), bool)
    for k in range(5):
        c = int(np.argmax(S[Fo[idx] != k].mean(0))); te = Fo[idx] == k; s[te] = S[te, c]; Pm[te] = M[c][te]
    o, t = CAND[int(np.argmax(S.mean(0)))]
    return s, Pm, np.stack([pp(x >= t, o) for x in Xt])


for m, (xrun, zruns) in SPEC.items():
    mi = np.where(me == m)[0]; ti = np.where(met == m)[0]; sel = RT[mi]; idx = mi[sel]; tsel = RE[ti]; tidx = ti[tsel]
    if not all((ROOT / "work/runs" / r / "test.npy").stat().st_size > 0 for r in zruns):
        print(m, "specialist runs not ready"); continue
    if STD[m] == "Fsub":
        s_std, P_std, T_std = nested_std(xrun, idx, tidx, sel, tsel)
    else:
        s_std = np.load("work/nested/T02.npy")[idx]; P_std = T02P[idx]; T_std = np.load("work/test/T02_bin.npy").astype(bool)[tidx]
    X = np.load(ROOT / "work/runs" / xrun / "train.npy").astype(np.float32)[sel]; Xt = np.load(ROOT / "work/runs" / xrun / "test.npy").astype(np.float32)[tsel]
    Z = np.mean([np.load(ROOT / "work/runs" / r / "train.npy").astype(np.float32) for r in zruns], 0)[sel]; Zt = np.mean([np.load(ROOT / "work/runs" / r / "test.npy").astype(np.float32) for r in zruns], 0)[tsel]
    src = {"Z": (Z, Zt), "XZ": ((X + Z) / 2, (Xt + Zt) / 2)}
    cl = 1 - P_std.reshape(len(idx), -1).mean(1); clt = 1 - T_std.reshape(len(tidx), -1).mean(1)
    opts = [(tau, sn, c) for tau in TAUS for sn in src for c in range(len(CAND))] ; opts = [o for o in opts if o[0] > 0] + [(0.0, "Z", 0)]
    Ssrc = {sn: np.stack([miou_batch(np.stack([pp(x >= t, o) for x in src[sn][0]]), G[idx]) for o, t in CAND], 1) for sn in src}
    def score(opt):
        tau, sn, c = opt; r = cl < tau
        return np.where(r, Ssrc[sn][:, c], s_std)
    SC = np.stack([score(o) for o in opts], 1); out = np.zeros(len(idx)); ch = []
    for k in range(5):
        j = int(np.argmax(SC[Fo[idx] != k].mean(0))); out[Fo[idx] == k] = SC[Fo[idx] == k, j]; ch.append((opts[j][0], opts[j][1]) + CAND[opts[j][2]])
    j = int(np.argmax(SC.mean(0))); tau, sn, c = opts[j]; o, t = CAND[c]
    d = np.array([out[Fo[idx] == k].mean() - s_std[Fo[idx] == k].mean() for k in range(5)])
    tiny = cl < 0.03; gt_tiny = (1 - G[idx].reshape(len(idx), -1).mean(1)) < 0.03
    print(f"{m}|F n={len(idx)}: standard {s_std.mean():.4f} -> routed specialist (nested) {out.mean():.4f} ({out.mean() - s_std.mean():+.4f}; folds {np.round(d, 4).tolist()}; {int((d > 0).sum())}/5) | choices {ch} | test choice {(tau, sn, o, t)} | routed train {int((cl < tau).sum())} test {int((clt < tau).sum())}")
    print(f"    on images with TRUE clean share < 3 % (n={int(gt_tiny.sum())}): standard {s_std[gt_tiny].mean():.4f} -> {out[gt_tiny].mean():.4f} | specialist alone at its best fixed (pp, thr), all images: {Ssrc['Z'].mean(0).max():.4f}, true-tiny images: {Ssrc['Z'][gt_tiny].mean(0).max():.4f} (standard map alone on true-tiny: {s_std[gt_tiny].mean():.4f})")
    s_out[idx] = out; Tm = T_std.copy(); r = clt < tau; Tm[r] = np.stack([pp(x >= t, o) for x in src[sn][1][r]]) if r.any() else Tm[r]; T_out[tidx] = Tm
np.save(f"{O}/Ftiny_train_scores.npy", s_out); np.save(f"{O}/Ftiny_test_bin.npy", T_out); print("written Ftiny chain")
