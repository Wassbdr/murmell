"""Global decode of every image with its method's extractor; agreement with the known message.
Writes $W/work/global/{train,test}.npz: names, method, agree (mean (2m-1)*tanh(logit)), bitacc, logits list.
    PYTHONPATH=$W/kit python global_decode.py --data $W/release --out $W/work/global
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from wmloc.data import LocalisationDataset
from wmloc.utils import to_tensor
from wmloc.watermarks import load_watermark


def run(split, data, wms, bs=16):
    ds = LocalisationDataset(data, split)
    names, meths, agree, bitacc = [], [], [], []
    by = {}
    for i, m in enumerate(ds.meta):
        by.setdefault(m["method"], []).append(i)
    for meth, idx in by.items():
        wm = wms[meth]
        for s in range(0, len(idx), bs):
            items = [ds[i] for i in idx[s:s + bs]]
            x = torch.cat([to_tensor(it["image"]) for it in items])
            lg = wm.logits(x).cpu().float().numpy()
            for it, l in zip(items, lg):
                sgn = it["message"].astype(np.float32) * 2 - 1
                names.append(it["image_name"]); meths.append(meth)
                agree.append(float(np.mean(sgn * np.tanh(l))))
                bitacc.append(float(np.mean((l > 0) == (it["message"] > 0))))
        print(split, meth, "done", flush=True)
    return dict(names=np.array(names), method=np.array(meths), agree=np.array(agree), bitacc=np.array(bitacc))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data"); ap.add_argument("--out")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    wms = {m: load_watermark(m, device=a.device) for m in ("pixelseal", "trustmark", "stegastamp")}
    for split in ("train", "test"):
        np.savez(out / f"{split}.npz", **run(split, a.data, wms))


if __name__ == "__main__":
    main()
