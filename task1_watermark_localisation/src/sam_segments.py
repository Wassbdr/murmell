"""SAM segments = label-free OBJECT-BOUNDARY prior (human list 23:45, item 5; started Sun 06:45 after the loss analysis of S11:
PixelSeal / TrustMark have only 4-7 images per cell above 0.95 -> the masks are found but their contours are blurred).
Automatic masks of the public SAM ViT-H for every image, stored at CELL level with the rule of the ground truth
(cell = 1 if >= 50 % of its 4x4 px belong to the segment).  No label, no watermark information: usable on train and test alike.
    python -m src.sam_segments --split train --shard 0 --nshards 4 --device cuda:0
-> work/sam/<split>_<k>of<n>.npz : idx [M] image index (metadata order), bits [M, 2048] packed 128x128 cell masks,
   area [M] area in px at 512, piou [M], stab [M]
"""
import argparse
import time

import numpy as np
import torch

from src.data import ROOT, image, names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train"); ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--device", default="cuda:0"); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--out", default="")
    ap.add_argument("--pps", type=int, default=32)
    a = ap.parse_args()
    from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
    sam = sam_model_registry["vit_h"](checkpoint=str(ROOT / "work/torch_cache/sam_vit_h_4b8939.pth")).to(a.device).eval()
    gen = SamAutomaticMaskGenerator(sam, points_per_side=a.pps, points_per_batch=128, pred_iou_thresh=0.86, stability_score_thresh=0.92,
                                    crop_n_layers=0, min_mask_region_area=0)
    N = len(names(a.split)); ids = [i for i in range(N) if i % a.nshards == a.shard]
    if a.limit:
        ids = ids[:a.limit]
    idx, bits, area, piou, stab = [], [], [], [], []; t0 = time.time()
    for j, i in enumerate(ids):
        with torch.no_grad():
            ms = gen.generate(image(a.split, i))
        for m in ms:
            s = m["segmentation"]
            c = s.reshape(128, 4, 128, 4).sum((1, 3)) >= 8
            if c.sum() < 2:
                continue
            idx.append(i); bits.append(np.packbits(c.ravel())); area.append(int(s.sum())); piou.append(m["predicted_iou"]); stab.append(m["stability_score"])
        if j % 20 == 0:
            print(f"{a.split} shard {a.shard}: {j + 1}/{len(ids)} images, {len(idx)} segments, {time.time() - t0:.0f}s", flush=True)
    out = a.out or str(ROOT / "work/sam" / f"{a.split}_{a.shard}of{a.nshards}.npz")
    with open(out, "wb") as f:                      # the file is pre-created (inode quota): write into it
        np.savez(f, idx=np.array(idx, np.int32), bits=np.stack(bits), area=np.array(area, np.int32),
                 piou=np.array(piou, np.float32), stab=np.array(stab, np.float32))
    print(f"DONE {a.split} shard {a.shard}: {len(ids)} images, {len(idx)} segments, {time.time() - t0:.0f}s -> {out}", flush=True)


if __name__ == "__main__":
    main()
