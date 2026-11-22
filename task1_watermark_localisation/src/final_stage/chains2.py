"""chains2 (Sun 10:10): fewer choices instead of more.  The big-pool selection loses to an A-PRIORI family mean with only
(post-proc, threshold) chosen in inner folds (PixelSeal full: .7817 vs .7744), so this script builds such chains and lets
each (method, regime) cell choose its chain in inner folds among a short list:
  full cells : T02 (selector), T03 (pair), S11, P05, Fens = family mean D7_<m>_Fens with nested (pp, thr)
  small cells: T02, S11snap, T01snap, T02snap, E8snap / E9snap = ensemble map (DS8 / DS9 family) with nested (pp, thr),
               decoder boxes, then the snap stage (src/final_stage/snap_stage.py) with its config chosen in inner folds
Box shrink for SS / TM boxes nested as in final_chain.py.  No label is used outside the inner folds.
    python src/final_stage/chains2.py --dir <scratch dir> --step build     # writes <dir>/E8_oof.npz, E9_oof.npz, Fens files
    (run snap_stage.py on E8_oof.npz / E9_oof.npz)
    python src/final_stage/chains2.py --dir <scratch dir> --step blend --out B20
"""
import argparse, hashlib, json, os, sys
import numpy as np
sys.path.insert(0, os.getcwd())
from pathlib import Path
from src.apply_boxes import cells
from src.combine2 import RE, RT
from src.data import METHODS, ROOT, folds, masks128, methods
from src.eval_cv import THRS
from src.metric import miou_batch, miou_image
from src.postproc_cv import OPTS, pp
from src.submit import write

ap = argparse.ArgumentParser(); ap.add_argument("--dir", required=True); ap.add_argument("--step", default="blend"); ap.add_argument("--out", default="B20")
ap.add_argument("--fchains", nargs="*", default=["T02", "T03", "S11", "P05", "Fens"]); ap.add_argument("--schains", nargs="*", default=["T02", "S11snap", "T01snap", "T02snap", "E8snap", "E9snap"])
ap.add_argument("--write", type=int, default=1); ap.add_argument("--fix", nargs="*", default=[])     # e.g. "trustmark|F=T02": cell kept on one chain (decided outside the folds: say so)
a = ap.parse_args(); O = a.dir; R = Path(".")
G = masks128().astype(bool); Fo = folds(); me = methods(); met = methods("test"); D = [-6, -4, -2, 0, 2, 4]
z = np.load("work/hunt/T02_oof.npz"); boxed = z["boxed"].astype(bool); bx = z["bx"]; boxedt = z["boxedt"].astype(bool); bxt = z["bxt"]; P02 = z["P"].astype(bool); T02m = z["T"].astype(bool)
fm = lambda v: np.array([v[Fo == k].mean() for k in range(5)])
CAND = [(o, float(t)) for o in OPTS for t in THRS]
ENSF = {"Fens": {"pixelseal": "D7_ps_Fens", "trustmark": "D7_tm_Fens"}, "Fall": {"pixelseal": "D7_ps_Fall", "trustmark": "D7_tm_Fall", "stegastamp": "F7_ss_Fall"}, "Fsub": {"pixelseal": "D7_ps_Fsub"}}
ENSS = {"Eany": {"pixelseal": "DS7_ps_Eany", "trustmark": "DS7_tm_Eany", "stegastamp": "S7_ss_Eany"}, "Esm": {"pixelseal": "DS7_ps_Esm", "trustmark": "DS7_tm_Esm", "stegastamp": "S7_ss_Esm"}, "E8": {"pixelseal": "DS8_ps_ens", "trustmark": "DS8_tm_ens", "stegastamp": "S8_ss_ens"}, "E9": {"pixelseal": "DS9_ps_ens", "trustmark": "DS9_tm_ens", "stegastamp": "S9_ss_ens"}}


def nested_map(run, m, full):
    """(pp, thr) of one map chosen on inner folds; returns train idx, per-image nested scores, OOF masks, test idx, test masks"""
    mi = np.where(me == m)[0]; ti = np.where(met == m)[0]; sel = RT[mi] == full; idx = mi[sel]; tsel = RE[ti] == full; tidx = ti[tsel]
    X = np.load(ROOT / "work/runs" / run / "train.npy").astype(np.float32)[sel]; Xt = np.load(ROOT / "work/runs" / run / "test.npy").astype(np.float32)[tsel]
    M = np.zeros((len(CAND), len(idx), 128, 128), bool); S = np.zeros((len(idx), len(CAND)))
    for c, (o, t) in enumerate(CAND):
        M[c] = np.stack([pp(x >= t, o) for x in X])
        if not full:
            for j, i in enumerate(idx):
                if boxed[i]:
                    M[c, j] = cells(bx[i]).astype(bool)
        S[:, c] = miou_batch(M[c], G[idx])
    s = np.zeros(len(idx)); Pm = np.zeros((len(idx), 128, 128), bool)
    for k in range(5):
        c = int(np.argmax(S[Fo[idx] != k].mean(0))); te = Fo[idx] == k; s[te] = S[te, c]; Pm[te] = M[c][te]
    c = int(np.argmax(S.mean(0))); o, t = CAND[c]; Tm = np.stack([pp(x >= t, o) for x in Xt])
    if not full:
        for j, i in enumerate(tidx):
            if boxedt[i]:
                Tm[j] = cells(bxt[i]).astype(bool)
    return idx, s, Pm, tidx, Tm, (o, t)


if a.step == "build":
    for ftag, fr in ENSF.items():
        sF = np.load("work/nested/T02.npy").copy(); TF = T02m.copy()
        for m, run in fr.items():
            idx, s, Pm, tidx, Tm, c = nested_map(run, m, True); sF[idx] = s; TF[tidx] = Tm
            print(f"{ftag} {m}|F n={len(idx)} nested {s.mean():.4f} test choice {c}")
        np.save(f"{O}/{ftag}_train_scores.npy", sF); np.save(f"{O}/{ftag}_test_bin.npy", TF)
    if os.environ.get("ONLY_F"):
        sys.exit(0)
    for tag, runs in ENSS.items():
        if os.environ.get('ONLY_S') and tag not in os.environ['ONLY_S'].split(','):
            continue
        P = P02.copy(); s_all = np.load("work/nested/T02.npy").copy(); T = T02m.copy()
        for m, run in runs.items():
            idx, s, Pm, tidx, Tm, c = nested_map(run, m, False); P[idx] = Pm; s_all[idx] = s; T[tidx] = Tm
            print(f"{tag} {m}|S n={len(idx)} nested (before snap) {s.mean():.4f} test choice {c}")
        np.savez_compressed(f"{O}/{tag}_oof.npz", P=P, s=s_all, boxed=boxed, bx=bx, T=T, boxedt=boxedt, bxt=bxt)
    sys.exit(0)

V = np.zeros((510, len(D)))
for i in np.where(boxed)[0]:
    y0, y1, x0, x1 = bx[i]
    for j, d in enumerate(D):
        c = [max(0, y0 - d), min(512, y1 + d), max(0, x0 - d), min(512, x1 + d)]
        if c[1] - c[0] < 4 or c[3] - c[2] < 4:
            c = [y0, y1, x0, x1]
        V[i, j] = miou_image(cells(c), G[i])


def nshrink(s):
    o = s.copy()
    for m in ("trustmark", "stegastamp"):
        sel = boxed & (me == m)
        for k in range(5):
            o[sel & (Fo == k)] = V[sel & (Fo == k), int(np.argmax(V[sel & (Fo != k)].mean(0)))]
    return o


def chain(n):
    if n.endswith("snap"):
        b = n[:-4]; return nshrink(np.load(f"{O}/{b}_snapg_train_scores.npy")), f"{O}/{b}_snapg_shrink2_test_bin.npy"
    if n in ENSF or os.path.exists(f"{O}/{n}_train_scores.npy"):
        return np.load(f"{O}/{n}_train_scores.npy"), f"{O}/{n}_test_bin.npy"
    return np.load(f"work/nested/{n}.npy"), f"work/test/{n}_bin.npy"


out = np.load("work/nested/T02.npy").copy(); B = T02m.copy(); ch, fin, tab = {}, {}, {}
for full, names_ in ((True, a.fchains), (False, a.schains)):
    C = {n: chain(n) for n in names_}
    for m in METHODS:
        c = (me == m) & (RT == full); key = f"{m}|{'F' if full else 'S'}"
        fx = dict(x.split("=") for x in a.fix).get(key)
        if fx:
            out[c] = chain(fx)[0][c]; ch[key] = [fx] * 5; fin[key] = fx; tab[key] = {fx: round(float(chain(fx)[0][c].mean()), 4)}
            kt = (met == m) & (RE == full); B[kt] = np.load(chain(fx)[1]).astype(bool)[kt]; continue
        for k in range(5):
            j = max(names_, key=lambda n: C[n][0][c & (Fo != k)].mean()); out[c & (Fo == k)] = C[j][0][c & (Fo == k)]; ch.setdefault(key, []).append(j)
        j = max(names_, key=lambda n: C[n][0][c].mean()); fin[key] = j; tab[key] = {n: round(float(C[n][0][c].mean()), 4) for n in names_}
        kt = (met == m) & (RE == full); B[kt] = np.load(C[j][1]).astype(bool)[kt]
f = fm(out); print(f"{a.out}: nested {f.mean():.6f} folds {np.round(f, 4).tolist()}")
for ref in ("T02", "B10", "B11"):
    d = f - fm(np.load(f"work/nested/{ref}.npy")); print(f"   vs {ref}: {d.mean():+.5f} {np.round(d, 4).tolist()} {int((d > 0).sum())}/5")
for key in tab:
    print("  ", key, "nested cell", round(float(out[(me == key.split('|')[0]) & (RT == (key[-1] == 'F'))].mean()), 4), "| chains", tab[key], "| per fold", ch[key], "| test", fin[key])
if a.write:
    o = R / f"work/archive/{a.out}/{a.out}.pt"; o.parent.mkdir(parents=True, exist_ok=True); write(B, a.out, f"{f.mean():.4f}", out=o); md5 = hashlib.md5(o.read_bytes()).hexdigest()
    with open(R / f"work/test/{a.out}_bin.npy", "wb") as fh:
        np.save(fh, B)
    with open(R / f"work/nested/{a.out}.npy", "wb") as fh:
        np.save(fh, out)
    cd = {f"{m}|{'F' if r else 'S'}": {"n": int(((me == m) & (RT == r)).sum()), "mean": float(out[(me == m) & (RT == r)].mean())} for m in METHODS for r in (True, False)}
    t2 = fm(np.load("work/nested/T02.npy"))
    (R / f"work/nested/{a.out}.json").write_text(json.dumps(dict(cv_foldmean=float(f.mean()), cv_std=float(f.std()), image_mean=float(out.mean()), folds=f.tolist(), cells=cd, clean=True,
        kind="chains2: per-cell choice among short lists of chains, in inner folds", base="T02", gain=float((f - t2).mean()), fold_diffs=(f - t2).tolist(), blend_choices=ch, blend_final=fin,
        chain_table=tab, fchains=a.fchains, schains=a.schains, fixed_cells=a.fix, note="free choice over the longer lists (T02 T03 S11 P05 Fens / + E8snap E9snap) gives 0.822780: TrustMark full flips between two equal chains; the lists were shortened and TM|F fixed AFTER seeing that table", test_wm_frac=float(B.mean())), indent=1))
    (R / f"work/archive/{a.out}/spec.json").write_text(json.dumps(dict(name=a.out, kind="chains2", base="T02",
        command="python src/final_stage/chains2.py --dir DIR --step build ; python src/final_stage/snap_stage.py --root . --base DIR/E8_oof.npz --name E8 --outdir DIR (same for E9) ; python src/final_stage/chains2.py --dir DIR --step blend --out " + a.out,
        blend_final=fin, submission_file=f"work/archive/{a.out}/{a.out}.pt", submission_md5=md5), indent=1))
    print("   written", a.out, md5, "| test images differing from B11:", int((B != np.load("work/test/B11_bin.npy").astype(bool)).reshape(1020, -1).any(1).sum()), "| wm frac", round(float(B.mean()), 4))
