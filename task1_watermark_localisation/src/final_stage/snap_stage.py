"""snap_stage: SAM snap as a LAST stage on any base candidate. CPU only (~1 min). GT-free rule.
Base = npz with P [510] train out-of-fold masks, s [510] per-image nested scores, boxed/bx, T [1020] test masks, boxedt/bxt.
For a predicted-small (RT / RE), unboxed image with a non-empty mask p:
   family A<ov>  : c = union of SAM segments q with |q & p| >= ov |q|
   family single : c = the SAM segment with the best IoU with p
   accepted iff  rectangularity(p) < RG  and  IoU(c, p) >= GATE  and  area(p) <= AMAX ; otherwise p is kept.
(family, RG, GATE, AMAX) chosen on the 4 inner folds (max mean score), applied to the 5th  -> nested per-image scores.
Test: the config chosen on all 510 images.  --shrink 1: StegaStamp / TrustMark decoder boxes shrunk by 2 px (chosen on all 510,
so the *_shrink2 train scores are NOT nested for this step; see work/nested/B10.json "correction").
  python snap_stage.py --root <task dir> --base work/hunt/T02_oof.npz --name T02 --outdir out
"""
import argparse, glob, itertools, json, sys
import numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); ap.add_argument("--base", required=True); ap.add_argument("--name", required=True)
ap.add_argument("--outdir", required=True); ap.add_argument("--families", default="A0.4,A0.5,A0.6,A0.7,A0.8,single")
a = ap.parse_args(); sys.path.insert(0, a.root)
from src.apply_boxes import cells
from src.combine2 import RE, RT
from src.data import ROOT, folds, masks128, methods
from src.metric import miou_image

G = masks128().astype(bool); Fo = folds(); me = methods(); met = methods("test")
z = np.load(a.base); P = z["P"].astype(bool); s = z["s"].astype(float); boxed = z["boxed"]; T = z["T"].astype(bool); boxedt = z["boxedt"]
FAM = a.families.split(","); GATES = [0, .2, .3, .4, .5, .6]; RGS = [.8, .9, 1.01]; AMAX = [.25, .4, 1.01]     # AMAX: no snap if the mask covers more than this fraction
CFG = list(itertools.product(FAM, RGS, GATES, AMAX))


def load_seg(split):
    seg = {}
    for f in sorted(glob.glob(str(ROOT / f"work/sam/{split}_*of4.npz"))):
        d = np.load(f); bits = np.unpackbits(d["bits"], axis=1).reshape(-1, 128, 128).astype(bool)
        for i in np.unique(d["idx"]):
            seg[int(i)] = bits[d["idx"] == i]
    return seg


def iou(x, y):
    u = (x | y).sum(); return (x & y).sum() / u if u else 1.0


def rectness(p):
    ys, xs = np.where(p); return p[ys.min():ys.max() + 1, xs.min():xs.max() + 1].mean()


def snap(Q, p, fam):
    if len(Q) == 0:
        return np.zeros_like(p)
    Qf = Q.reshape(len(Q), -1); ar = Qf.sum(1); inter = (Qf & p.ravel()).sum(1)
    if fam == "single":
        v = inter / np.maximum(1, ar + p.sum() - inter); return Q[int(np.argmax(v))]
    k = (ar > 0) & (inter >= float(fam[1:]) * ar)
    return Q[k].any(0) if k.any() else np.zeros_like(p)


def stage(p, Q, cfg):
    fam, rg, gate, amax = cfg
    if not p.any() or rectness(p) >= rg or p.mean() > amax:
        return None
    c = snap(Q, p, fam)
    return c if (c.any() and iou(c, p) >= gate) else None


seg = load_seg("train"); cand = [i for i in range(510) if not RT[i] and not boxed[i] and P[i].any()]
S = np.tile(s[:, None], (1, len(CFG)))
for i in cand:
    cache = {}
    for j, cfg in enumerate(CFG):
        if cfg[0] not in cache:
            c = snap(seg[i], P[i], cfg[0]); cache[cfg[0]] = (c, miou_image(c, G[i]) if c.any() else None, iou(c, P[i]), rectness(P[i]))
        c, v, io, rc = cache[cfg[0]]
        if v is not None and rc < cfg[1] and io >= cfg[2] and P[i].mean() <= cfg[3]:
            S[i, j] = v
fm = lambda v: np.array([v[Fo == k].mean() for k in range(5)])
out = s.copy(); ch = []
for k in range(5):
    j = int(np.argmax(S[Fo != k].mean(0))); ch.append(CFG[j]); out[Fo == k] = S[Fo == k, j]
d = fm(out) - fm(s); jall = int(np.argmax(S.mean(0))); cfg = CFG[jall]
rep = {"base": a.name, "base_nested": float(fm(s).mean()), "snap_nested": float(fm(out).mean()), "gain": float(d.mean()), "fold_diffs": d.tolist(),
       "folds_up": int((d > 0).sum()), "choices_per_fold": [list(c) for c in ch], "test_config": list(cfg), "gain_in_sample_test_config": float(S[:, jall].mean() - s.mean()),
       "train_candidates": len(cand), "train_changed": int((S[:, jall] != s).sum())}
print(f"{a.name}: nested {rep['base_nested']:.6f} -> {rep['snap_nested']:.6f} (gain {d.mean():+.5f}; folds " + " ".join(f"{x:+.4f}" for x in d) + f"; {rep['folds_up']}/5) choices {ch} | test config {cfg} in-sample {rep['gain_in_sample_test_config']:+.5f}")
for m in ("pixelseal", "trustmark", "stegastamp"):
    k = np.array([i for i in cand if me[i] == m]); print(f"   {m}: n={len(k)} base {s[k].mean():.4f} -> nested {out[k].mean():.4f}")
# shrink on train (fixed -2 px for SS / TM boxes)
out2 = out.copy()
for i in np.where(boxed)[0]:
    if me[i] in ("stegastamp", "trustmark"):
        b = z["bx"][i]; out2[i] = miou_image(cells([b[0] + 2, b[1] - 2, b[2] + 2, b[3] - 2]), G[i])
d2 = fm(out2) - fm(s); rep.update(snap_shrink_nested=float(fm(out2).mean()), gain_shrink=float(d2.mean()), fold_diffs_shrink=d2.tolist())
print(f"   + shrink 2 px (SS/TM boxes): {fm(out2).mean():.6f} (gain {d2.mean():+.5f}; folds " + " ".join(f"{x:+.4f}" for x in d2) + f"; {int((d2 > 0).sum())}/5)")
np.save(f"{a.outdir}/{a.name}_snapg_train_scores.npy", out); np.save(f"{a.outdir}/{a.name}_snapg_shrink2_train_scores.npy", out2)
# nested train masks (fold k uses the config chosen without fold k) incl. shrunk boxes: usable as an out-of-fold source
PM = P.copy()
for i in cand:
    c = stage(P[i], seg[i], ch[int(Fo[i])])
    if c is not None:
        PM[i] = c
for i in np.where(boxed)[0]:
    if me[i] in ("stegastamp", "trustmark"):
        b = z["bx"][i]; PM[i] = cells([b[0] + 2, b[1] - 2, b[2] + 2, b[3] - 2]).astype(bool)
chk = np.array([miou_image(PM[i], G[i]) for i in range(510)]); assert np.abs(chk - out2).max() < 1e-9, np.abs(chk - out2).max()
np.save(f"{a.outdir}/{a.name}_snapg_shrink2_train_masks.npy", PM)
# test
segt = load_seg("test"); B = T.copy(); n = 0
for i in range(1020):
    if RE[i] or boxedt[i] or i not in segt:
        continue
    c = stage(T[i], segt[i], cfg)
    if c is not None:
        B[i] = c; n += 1
np.save(f"{a.outdir}/{a.name}_snapg_test_bin.npy", B)
B2 = B.copy(); ns = 0
for i in np.where(boxedt)[0]:
    if met[i] in ("stegastamp", "trustmark"):
        b = z["bxt"][i]; c = [b[0] + 2, b[1] - 2, b[2] + 2, b[3] - 2]
        if c[1] - c[0] >= 8 and c[3] - c[2] >= 8:
            B2[i] = cells(c).astype(bool); ns += 1
np.save(f"{a.outdir}/{a.name}_snapg_shrink2_test_bin.npy", B2)
rep.update(test_changed=n, test_boxes_shrunk=ns, test_wm_frac=[float(T.mean()), float(B.mean()), float(B2.mean())])
json.dump(rep, open(f"{a.outdir}/{a.name}_snapg_report.json", "w"), indent=1)
print(f"   test: {n} images changed of 1020 ({rep['train_changed']} of 510 in train), {ns} boxes shrunk, wm frac {T.mean():.4f} -> {B.mean():.4f} -> {B2.mean():.4f}")
