"""TrustMark small-regime diagnostic: box proposals from OOF maps, ranked by the official decoder with the KNOWN message.
GT is used ONLY to measure (best available IoU among proposals, IoU of the chosen box, final mIoU vs C3).
Proposals per image (predicted-small TM images): for map in {S1_tm, C3 map}, thresholds {.2,.3,.4,.5,.6,.7} ->
connected components (8-conn) -> bbox, plus bbox of the top-k cells (k = 0.5/1/2/4/8 % of the frame), each also
scaled x{0.85, 1.0, 1.15}. Score = agreement mean((2m-1) tanh(logit)) of the decoder on the crop (resized internally).
Null: the same ranking with K permuted messages (bits shuffled).
    python -m src.tm_boxes --method trustmark --device cuda:1 [--limit 0]
Writes work/diag/boxes_<method>.npz and prints the tables.
"""
import argparse
import json
import sys

import numpy as np
import torch
from scipy import ndimage as ndi

from src.combine2 import RT, src
from src.data import ROOT, folds, image, mask512, masks128, message, methods, regime
from src.metric import miou_batch

sys.path.insert(0, str(ROOT / "kit"))


def boxes_from_map(M):
    out = set()
    for t in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7):
        lab, n = ndi.label(M >= t, structure=np.ones((3, 3)))
        for sl in ndi.find_objects(lab):
            if sl is not None:
                out.add((sl[0].start, sl[0].stop, sl[1].start, sl[1].stop))
    flat = np.argsort(-M.ravel())
    for frac in (0.005, 0.01, 0.02, 0.04, 0.08):
        k = max(1, int(frac * M.size)); ys, xs = np.unravel_index(flat[:k], M.shape)
        out.add((ys.min(), ys.max() + 1, xs.min(), xs.max() + 1))
    res = set()
    for (y0, y1, x0, x1) in out:
        cy, cx, h, w = (y0 + y1) / 2, (x0 + x1) / 2, y1 - y0, x1 - x0
        for s in (0.85, 1.0, 1.15):
            hh, ww = max(2, h * s), max(2, w * s)
            b = (int(max(0, round(cy - hh / 2))), int(min(128, round(cy + hh / 2))), int(max(0, round(cx - ww / 2))), int(min(128, round(cx + ww / 2))))
            if b[1] - b[0] >= 2 and b[3] - b[2] >= 2:
                res.add(b)
    return sorted(res)


def box_mask(b):
    m = np.zeros((128, 128), np.uint8); m[b[0]:b[1], b[2]:b[3]] = 1; return m


def iou1(a, g):
    u = (a | g).sum(); return 1.0 if u == 0 else (a & g).sum() / u


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--method", default="trustmark"); ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--limit", type=int, default=0); ap.add_argument("--nperm", type=int, default=8)
    a = ap.parse_args()
    from wmloc.watermarks import load_watermark
    wm = load_watermark(a.method, models_dir=ROOT / "models", device=a.device)
    me, G = methods(), masks128(); mi = np.where(me == a.method)[0]
    s1 = src({"trustmark": "S1_tm", "stegastamp": "S1_ss", "pixelseal": "S1_ps_amp4"}[a.method], a.method)[0]
    c3 = np.load(ROOT / "work/archive/C3/C3_oof.npy").astype(np.float32)[mi]
    sel = [k for k, i in enumerate(mi) if not RT[i]]                      # predicted-small images (as at test)
    if a.limit:
        sel = sel[:a.limit]
    rng = np.random.RandomState(0); rows = []
    for k in sel:
        i = mi[k]; img = torch.from_numpy(image("train", int(i))).permute(2, 0, 1)[None].float().div(255)
        props = sorted(set(boxes_from_map(s1[k])) | set(boxes_from_map(c3[k])))
        crops = [torch.nn.functional.interpolate(img[:, :, 4 * b[0]:4 * b[1], 4 * b[2]:4 * b[3]], size=(256, 256), mode="bilinear",
                                                 align_corners=False, antialias=True) for b in props]
        lg = np.concatenate([wm.logits(torch.cat(crops[j:j + 64])).cpu().numpy() for j in range(0, len(crops), 64)])
        m = message("train", int(i)).astype(np.float32); th = np.tanh(lg)
        ag = (th * (2 * m - 1)).mean(1); acc = ((lg > 0) == (m > 0)).mean(1)
        g = G[i].astype(bool); ious = np.array([iou1(box_mask(b).astype(bool), g) for b in props])
        j = int(np.argmax(ag))
        perm_ious = []
        for _ in range(a.nperm):
            mp = rng.permutation(m); perm_ious.append(ious[int(np.argmax((th * (2 * mp - 1)).mean(1)))])
        rows.append(dict(i=int(i), n=len(props), best_iou=float(ious.max()), chosen_iou=float(ious[j]), chosen_acc=float(acc[j]),
                         chosen_ag=float(ag[j]), perm_iou=float(np.mean(perm_ious)), box=props[j], area=float(g.mean()),
                         rect=bool(mask512(int(i)).sum() >= 0.97 * np.prod(np.ptp(np.argwhere(mask512(int(i))), 0) + 1)) if g.any() else False,
                         true_small=bool(not regime()[i])))
        r = rows[-1]
        print(f"{i:3d} n={r['n']:3d} best {r['best_iou']:.3f} chosen {r['chosen_iou']:.3f} (acc {r['chosen_acc']:.2f}) perm {r['perm_iou']:.3f} area {r['area']:.3f} rect {r['rect']}", flush=True)
    (ROOT / "work/diag").mkdir(parents=True, exist_ok=True)
    json.dump(rows, open(ROOT / "work/diag" / f"boxes_{a.method}.json", "w"))
    R = rows
    print(f"\nSUMMARY {a.method} predicted-small n={len(R)}: mean best-available IoU {np.mean([r['best_iou'] for r in R]):.3f} | "
          f"chosen (true msg) {np.mean([r['chosen_iou'] for r in R]):.3f} | chosen (permuted msgs) {np.mean([r['perm_iou'] for r in R]):.3f} | "
          f"frac best>=0.7 {np.mean([r['best_iou'] >= 0.7 for r in R]):.2f}")


if __name__ == "__main__":
    main()
