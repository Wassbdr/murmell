"""Decoder-ranked box search for SMALL-regime images (GT-free; GT only used afterwards for diagnostics).
For each predicted-small image of <method>:
  proposals = boxes derived from the OOF / test map (thresholded components, top-k cells, x{0.85,1,1.15})
            ∪ a multi-scale grid centred on the 10 strongest map peaks (8 sizes 24..360 px, aspects {1/2,1,2})
  score     = agreement mean((2m-1) tanh(logit)) of the OFFICIAL decoder on the crop, with the KNOWN message
  refine    = coordinate ascent on the 4 edges of the 5 best boxes (steps 16, 8, 4, 2 px)
Output: work/boxes/<method>_<split>.json : per image name -> box (y0,y1,x0,x1 in px), agreement, bit acc, n proposals.
    python -m src.box_search --method trustmark --split train --map S1_tm --device cuda:0 [--shard 0 --nshards 1]
"""
import argparse
import json
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from src.combine2 import RE, RT, src
from src.data import ROOT, image, message, methods, names
from src.tm_boxes import boxes_from_map

sys.path.insert(0, str(ROOT / "kit"))


def peaks(M, k=10, r=6):
    M = M.copy(); out = []
    for _ in range(k):
        j = int(np.argmax(M)); y, x = divmod(j, M.shape[1]); out.append((y, x))
        M[max(0, y - r):y + r + 1, max(0, x - r):x + r + 1] = -1
    return out


ARS = (0.5, 1.0, 2.0)          # peak-grid aspects (w/h); --ars overrides (official doc: PS/SS up to 16:1, TM 2:1)
FARS = (0.6, 1.0, 1.6)         # full-frame grid aspects
MAXASP = 0.0                   # >0: refinement rejects boxes with aspect > MAXASP (TrustMark: 2)


def grid_boxes(M):
    out = set()
    for (y, x) in peaks(M):
        cy, cx = 4 * y + 2, 4 * x + 2
        for side in (24, 36, 54, 80, 120, 180, 260, 360):
            for ar in ARS:
                h, w = side / np.sqrt(ar), side * np.sqrt(ar)
                out.add((int(cy - h / 2), int(cy + h / 2), int(cx - w / 2), int(cx + w / 2)))
    return out


def full_grid():
    """frame-wide coarse grid for medium patches the map misses (TrustMark tolerates ~±8 px / ±10 %)."""
    out = set()
    for side in (40, 56, 80, 112):
        for ar in FARS:
            h, w = int(side / np.sqrt(ar)), int(side * np.sqrt(ar)); st = max(8, side // 2)
            for y in range(0, 512 - h + 1, st):
                for x in range(0, 512 - w + 1, st):
                    out.add((y, y + h, x, x + w))
    return out


def clip(b):
    y0, y1, x0, x1 = b
    y0, x0 = max(0, int(y0)), max(0, int(x0)); y1, x1 = min(512, int(y1)), min(512, int(x1))
    if y1 - y0 < 12 or x1 - x0 < 12:
        return None
    if MAXASP and max(y1 - y0, x1 - x0) > MAXASP * min(y1 - y0, x1 - x0):
        return None
    return (y0, y1, x0, x1)


class Scorer:
    def __init__(self, wm, img, msg):
        self.wm, self.img, self.sgn = wm, img, torch.from_numpy(2 * msg.astype(np.float32) - 1)
        self.cache = {}

    def __call__(self, boxes):
        todo = [b for b in boxes if b not in self.cache]
        for j in range(0, len(todo), 64):
            ch = todo[j:j + 64]
            x = torch.cat([F.interpolate(self.img[:, :, b[0]:b[1], b[2]:b[3]], size=(256, 256), mode="bilinear",
                                         align_corners=False, antialias=True) for b in ch])
            lg = self.wm.logits(x).float().cpu()
            ag = (torch.tanh(lg) * self.sgn).mean(1); acc = ((lg > 0).float() == (self.sgn > 0).float()).float().mean(1)
            for b, a_, c_ in zip(ch, ag.tolist(), acc.tolist()):
                self.cache[b] = (a_, c_)
        return [self.cache[b] for b in boxes]


def refine(S, b):
    best = b; bv = S([b])[0][0]
    for st in (16, 8, 4, 2):
        improved = True
        while improved:
            improved = False
            cand = []
            for i in range(4):
                for sg in (-1, 1):
                    c = list(best); c[i] += sg * st; c = clip(tuple(c))
                    if c:
                        cand.append(c)
            for c, (v, _) in zip(cand, S(cand)):
                if v > bv:
                    bv, best, improved = v, c, True
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method"); ap.add_argument("--split", default="train"); ap.add_argument("--map")
    ap.add_argument("--device", default="cuda:0"); ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0); ap.add_argument("--grid", type=int, default=0)
    ap.add_argument("--ars", default=""); ap.add_argument("--fars", default=""); ap.add_argument("--maxasp", type=float, default=0)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    global ARS, FARS, MAXASP
    if a.ars:
        ARS = tuple(float(v) for v in a.ars.split(","))
    if a.fars:
        FARS = tuple(float(v) for v in a.fars.split(","))
    MAXASP = a.maxasp
    from wmloc.watermarks import load_watermark
    wm = load_watermark(a.method, models_dir=ROOT / "models", device=a.device)
    me = methods(a.split); mi = np.where(me == a.method)[0]
    maps = src(a.map, a.method)[0 if a.split == "train" else 1]
    reg = RT if a.split == "train" else RE
    sel = [k for k, i in enumerate(mi) if not reg[i]][a.shard::a.nshards]
    if a.limit:
        sel = sel[:a.limit]
    out = {}; t0 = time.time()
    for n_done, k in enumerate(sel):
        i = int(mi[k]); M = maps[k].astype(np.float32)
        img = torch.from_numpy(image(a.split, i)).permute(2, 0, 1)[None].float().div(255).to(a.device)
        S = Scorer(wm, img, message(a.split, i))
        props = {clip((4 * b[0], 4 * b[1], 4 * b[2], 4 * b[3])) for b in boxes_from_map(M)} | {clip(b) for b in grid_boxes(M)}
        if a.grid:
            props |= full_grid()
        props = sorted(p for p in props if p)
        sc = S(props); order = np.argsort([-s[0] for s in sc])[:5]
        ref = [refine(S, props[j]) for j in order]
        rs = S(ref); j = int(np.argmax([r[0] for r in rs]))
        out[names(a.split)[i]] = {"box": list(ref[j]), "ag": rs[j][0], "acc": rs[j][1], "n_props": len(props), "n_eval": len(S.cache)}
        if n_done % 10 == 0:
            print(f"{a.method} {a.split} {n_done}/{len(sel)} acc {rs[j][1]:.2f} evals {len(S.cache)} {time.time() - t0:.0f}s", flush=True)
    d = ROOT / "work/boxes"; d.mkdir(parents=True, exist_ok=True)
    (d / f"{a.method}_{a.split}_{a.map}{'_grid' if a.grid else ''}{a.tag}_{a.shard}of{a.nshards}.json").write_text(json.dumps(out))
    print("done", len(out), f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
