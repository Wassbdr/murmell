"""PROMPTED SAM (Sun 07:10): instead of hoping that the automatic masks contain the transplanted object, ask SAM for the
object under our own evidence.  For each image of a method, the prompt comes from the method's ensemble map E
(run DS9_ps_ens / DS9_tm_ens / S9_ss_ens = label-free mean of the S-usable runs, see src/sam_ens.py):
  blob  = largest connected component of E >= 0.5 * max(E)
  P1    = box of the blob + positive point at the maximum of the smoothed E inside it      -> 3 masks (SAM multimask)
  P2    = positive point alone                                                              -> 3 masks
  P3    = box alone                                                                         -> 1 mask
Masks are stored at cell level with the rule of the ground truth, in the format of src/sam_segments.py:
    work/sam/prompt_<split>_<k>of<n>.npz : idx, bits [M, 2048], area, piou (SAM's own score), stab (prompt id 1/2/3)
No label is read.  The choice between these masks, the automatic ones and no snapping at all is made later (src.sam_ens).
    python -m src.sam_prompt --split train --shard 0 --nshards 2 --device cuda:0
"""
import argparse
import time

import numpy as np
import torch
from scipy import ndimage as ndi

from src.data import METHODS, ROOT, image, methods
from src.sam_ens import ENS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train"); ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--device", default="cuda:0"); ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    from segment_anything import SamPredictor, sam_model_registry
    sam = sam_model_registry["vit_h"](checkpoint=str(ROOT / "work/torch_cache/sam_vit_h_4b8939.pth")).to(a.device).eval()
    pred = SamPredictor(sam); me = methods(a.split)
    E = np.zeros((len(me), 128, 128), np.float32)
    for m in METHODS:
        E[me == m] = np.load(ROOT / "work/runs" / ENS[m] / f"{a.split}.npy").astype(np.float32)
    ids = [i for i in range(len(me)) if i % a.nshards == a.shard]
    if a.limit:
        ids = ids[:a.limit]
    idx, bits, area, piou, pid = [], [], [], [], []; t0 = time.time()
    for j, i in enumerate(ids):
        p = E[i]; b = p >= 0.5 * float(p.max()); lab, n = ndi.label(b)
        if n == 0:
            continue
        sz = ndi.sum(b, lab, range(1, n + 1)); c = lab == 1 + int(np.argmax(sz)); ys, xs = np.where(c)
        ps = ndi.gaussian_filter(p, 2.0) * c; py, px = np.unravel_index(int(np.argmax(ps)), ps.shape)
        box = np.array([4 * xs.min(), 4 * ys.min(), 4 * xs.max() + 4, 4 * ys.max() + 4], np.float32)
        pt = np.array([[4 * px + 2, 4 * py + 2]], np.float32); lb = np.array([1])
        with torch.no_grad():
            pred.set_image(image(a.split, i))
            outs = [(1, pred.predict(point_coords=pt, point_labels=lb, box=box, multimask_output=True)),
                    (2, pred.predict(point_coords=pt, point_labels=lb, multimask_output=True)),
                    (3, pred.predict(box=box, multimask_output=False))]
        for k, (ms, sc, _) in outs:
            for s, q in zip(ms, sc):
                cm = s.reshape(128, 4, 128, 4).sum((1, 3)) >= 8
                if cm.sum() < 2:
                    continue
                idx.append(i); bits.append(np.packbits(cm.ravel())); area.append(int(s.sum())); piou.append(float(q)); pid.append(k)
        if j % 50 == 0:
            print(f"{a.split} shard {a.shard}: {j + 1}/{len(ids)} images, {len(idx)} masks, {time.time() - t0:.0f}s", flush=True)
    out = ROOT / "work/sam" / f"prompt_{a.split}_{a.shard}of{a.nshards}.npz"
    with open(out, "wb") as f:
        np.savez(f, idx=np.array(idx, np.int32), bits=np.stack(bits), area=np.array(area, np.int32),
                 piou=np.array(piou, np.float32), stab=np.array(pid, np.float32))
    print(f"DONE {a.split} shard {a.shard}: {len(ids)} images, {len(idx)} masks, {time.time() - t0:.0f}s -> {out}", flush=True)


if __name__ == "__main__":
    main()
