"""TrustMark >= 0.9 official box detector localize() as extra TrustMark box proposals (human item 5).
Runs with the PRIVATE env work/venv_tm09 (trustmark 0.9.2 --no-deps; other deps read from the main .venv via a .pth;
the main venv keeps trustmark 0.8.0).  For each predicted-small TrustMark image: all localize() boxes (+ the top one)
are scored with the official kit decoder and the KNOWN message (box_search.Scorer), the best is edge-refined
(aspect <= 2:1, official doc), saved as work/boxes/trustmark_<split>_loc09_0of1.json (same format as box_search).
    work/venv_tm09/bin/python -m src.tm_localize --split train --device cuda:0 [--limit 5]
"""
import argparse
import json
import time

import numpy as np
import torch
from PIL import Image

import src.box_search as bs
from src.combine2 import RE, RT
from src.data import ROOT, image, message, methods, names


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--split", default="train"); ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    from trustmark import TrustMark
    from wmloc.watermarks import load_watermark
    tm = TrustMark(verbose=False, model_type="Q", loadRemover=False, loadBBoxDetector=True, device=a.device)
    wm = load_watermark("trustmark", models_dir=ROOT / "models", device=a.device)
    bs.MAXASP = 2.0
    me = methods(a.split); reg = RT if a.split == "train" else RE
    idx = [int(i) for i in np.where((me == "trustmark") & ~reg)[0]]
    if a.limit:
        idx = idx[:a.limit]
    out, t0, nbox = {}, time.time(), []
    for n_, i in enumerate(idx):
        im = image(a.split, i); H, W = im.shape[:2]
        with torch.no_grad():
            bx = tm.localize(Image.fromarray(im), return_all=True) or []
        props = []
        for x1, y1, x2, y2 in bx:
            b = bs.clip((y1 * H, y2 * H, x1 * W, x2 * W))
            if b:
                props.append(b)
        nbox.append(len(props))
        if not props:
            continue
        S = bs.Scorer(wm, torch.from_numpy(im).permute(2, 0, 1)[None].float().div(255).to(a.device), message(a.split, i))
        sc = S(props); order = np.argsort([-s[0] for s in sc])[:3]
        ref = [bs.refine(S, props[j]) for j in order]; rs = S(ref); j = int(np.argmax([r[0] for r in rs]))
        out[names(a.split)[i]] = {"box": list(ref[j]), "ag": rs[j][0], "acc": rs[j][1], "n_props": len(props),
                                  "raw_best": list(props[order[0]]), "raw_acc": sc[order[0]][1]}
        if n_ % 20 == 0:
            print(f"{a.split} {n_}/{len(idx)} props {len(props)} acc {rs[j][1]:.2f} {time.time() - t0:.0f}s", flush=True)
    print("images", len(idx), "with boxes", len(out), "mean props", np.mean(nbox) if nbox else 0, flush=True)
    if not a.limit:
        (ROOT / "work/boxes" / f"trustmark_{a.split}_loc09_0of1.json").write_text(json.dumps(out))
    else:
        from src.data import masks128
        from src.apply_boxes import cells
        from src.metric import miou_image
        G = masks128(); ix = {n: k for k, n in enumerate(names())}
        for n, r in out.items():
            if a.split == "train":
                print(n, "acc %.2f" % r["acc"], "IoU-cells %.3f" % miou_image(cells(r["box"]), G[ix[n]]), "raw acc %.2f" % r["raw_acc"])
    print("done", f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
