"""Final assembly (Sun 09:25): per-method choice of the chain for the predicted-SMALL cells, in inner folds, among existing
chains and the chains of a NEW base candidate (full_nested with the SAM-object runs RS_* in its pool).  Full cells stay on T02.
Box shrink (SS / TM decoder boxes) chosen in inner folds for the train scores; test uses -2 px (the choice on all 510).
    python src/final_stage/final_chain.py --new P05 --out B15 --dir <scratch dir with *_snapg_* files> [--chains T02 S11snap ...]
Writes work/archive/<out>, work/nested/<out>.{json,npy}, work/test/<out>_bin.npy.  Prints the fold table vs T02, B10 and B11.
"""
import argparse, hashlib, json, os, sys
import numpy as np
sys.path.insert(0, os.getcwd())
from pathlib import Path
from src.apply_boxes import cells
from src.combine2 import RE, RT
from src.data import METHODS, folds, masks128, methods
from src.metric import miou_image
from src.submit import write

ap = argparse.ArgumentParser(); ap.add_argument("--new", default=""); ap.add_argument("--out", required=True); ap.add_argument("--dir", required=True)
ap.add_argument("--chains", nargs="*", default=None); ap.add_argument("--write", type=int, default=1)
a = ap.parse_args(); O = a.dir; R = Path(".")
G = masks128().astype(bool); Fo = folds(); me = methods(); met = methods("test"); D = [-6, -4, -2, 0, 2, 4]
z = np.load("work/hunt/S11_oof.npz"); boxed = z["boxed"].astype(bool); bx = z["bx"]; V = np.zeros((510, len(D)))
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


fm = lambda v: np.array([v[Fo == k].mean() for k in range(5)])
C = {"T02": (np.load("work/nested/T02.npy"), "work/test/T02_bin.npy")}
for n in ("S11", "T01", "T02") + ((a.new,) if a.new else ()):
    C[n + "snap"] = (nshrink(np.load(f"{O}/{n}_snapg_train_scores.npy")), f"{O}/{n}_snapg_shrink2_test_bin.npy")
if a.new:
    C[a.new] = (np.load(f"work/nested/{a.new}.npy"), f"work/test/{a.new}_bin.npy")
names_ = a.chains or list(C); S = [C[n][0] for n in names_]
out = C["T02"][0].copy(); ch = {}
for m in METHODS:
    c = (me == m) & ~RT
    for k in range(5):
        j = int(np.argmax([s[c & (Fo != k)].mean() for s in S])); out[c & (Fo == k)] = S[j][c & (Fo == k)]; ch.setdefault(m, []).append(names_[j])
fin = {m: names_[int(np.argmax([s[(me == m) & ~RT].mean() for s in S]))] for m in METHODS}
f = fm(out); print(f"{a.out}: nested {f.mean():.6f} folds {np.round(f, 4).tolist()}")
for ref in ("T02", "B10", "B11"):
    d = f - fm(np.load(f"work/nested/{ref}.npy")); print(f"   vs {ref}: {d.mean():+.5f} {np.round(d, 4).tolist()} {int((d > 0).sum())}/5")
print("   choices per fold", ch); print("   final (test)", fin)
print("   small cells:", {m[:2]: {n: round(float(s[(me == m) & ~RT].mean()), 4) for n, s in zip(names_, S)} for m in METHODS})
if a.write:
    B = np.load("work/test/T02_bin.npy").astype(bool).copy()
    for m, n in fin.items():
        if n != "T02":
            A_ = np.load(C[n][1]).astype(bool); k = (met == m) & ~RE; B[k] = A_[k]
    o = R / f"work/archive/{a.out}/{a.out}.pt"; o.parent.mkdir(parents=True, exist_ok=True); write(B, a.out, f"{f.mean():.4f}", out=o); md5 = hashlib.md5(o.read_bytes()).hexdigest()
    with open(R / f"work/test/{a.out}_bin.npy", "wb") as fh:
        np.save(fh, B)
    with open(R / f"work/nested/{a.out}.npy", "wb") as fh:
        np.save(fh, out)
    cellsd = {f"{m}|{'F' if r else 'S'}": {"n": int(((me == m) & (RT == r)).sum()), "mean": float(out[(me == m) & (RT == r)].mean())} for m in METHODS for r in (True, False)}
    t2 = fm(C["T02"][0])
    (R / f"work/nested/{a.out}.json").write_text(json.dumps(dict(cv_foldmean=float(f.mean()), cv_std=float(f.std()), image_mean=float(out.mean()), folds=f.tolist(), cells=cellsd,
        clean=True, kind="blend_small over chains " + " ".join(names_), base="T02", new_base=a.new, gain=float((f - t2).mean()), fold_diffs=(f - t2).tolist(),
        blend_choices=ch, blend_final=fin, test_wm_frac=float(B.mean())), indent=1))
    (R / f"work/archive/{a.out}/spec.json").write_text(json.dumps(dict(name=a.out, kind="blend_small", base="T02", new_base=a.new,
        command=f"python src/final_stage/make_oof.py {a.new} <dir>/{a.new}_oof.npz ; python src/final_stage/snap_stage.py --root . --base <dir>/{a.new}_oof.npz --name {a.new} --outdir <dir> ; python src/final_stage/final_chain.py --new {a.new} --out {a.out} --dir <dir>",
        blend_final=fin, submission_file=f"work/archive/{a.out}/{a.out}.pt", submission_md5=md5), indent=1))
    print("   written", a.out, md5, "| test images differing from B11:", int((B != np.load("work/test/B11_bin.npy").astype(bool)).reshape(1020, -1).any(1).sum()), "| wm frac", round(float(B.mean()), 4))
