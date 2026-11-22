"""Shift test-time augmentation (human 01:50): predict on copies of the frame shifted by 0 / 4 / 8 px in y and x
(multiples of the 4-px cell, NO flips: the watermark is not symmetric), move each 128x128 map back by the same number of
cells and average the probabilities over the shifts that cover a cell.
    python -m src.tta --exp D2_tm --device cuda:0        -> work/runs/D2_tm_tta/{train.npy,test.npy,args.json}
The output is an ordinary run (same regime / flags as its base), so src.night evaluates it like any other source and the
nested selection decides; the gain is judged with the usual gate (+0.003 nested, >= 4/5 outer folds).
Works for runs with a single model.pt (plain runs).  Shifted frame: reflect-padded on the top/left, cropped to 512.
"""
import argparse
import json

import numpy as np
import torch
import torch.nn.functional as F

import src.train_full as tf
from src.data import ROOT, image, methods


@torch.no_grad()
def predict_tta(net, split, idx, dev, shifts, bs=8):
    net.eval(); out = []
    for s in range(0, len(idx), bs):
        x = torch.from_numpy(np.stack([image(split, int(i)) for i in idx[s:s + bs]])).permute(0, 3, 1, 2).to(dev)
        acc = torch.zeros(len(x), 128, 128, device=dev); cnt = torch.zeros(128, 128, device=dev)
        for dy in shifts:
            for dx in shifts:
                xs = F.pad(x.float(), (dx, 0, dy, 0), mode="reflect")[:, :, :512, :512].round().to(torch.uint8) if (dy or dx) else x
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    p = torch.sigmoid(net(xs).float())[:, 0]
                cy, cx = dy // 4, dx // 4                                  # original cell (y, x) = shifted cell (y + cy, x + cx)
                acc[:, :128 - cy, :128 - cx] += p[:, cy:, cx:]; cnt[:128 - cy, :128 - cx] += 1
        out.append((acc / cnt).cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--exp"); ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--shifts", default="0,4,8"); ap.add_argument("--tag", default="_tta")
    ap.add_argument("--xf", type=int, default=0)     # 1: fold-clean set <exp>_xf0..4 -> <exp>_xfall_tta (train map of an image
    a = ap.parse_args(); dev = a.device; shifts = [int(v) for v in a.shifts.split(",")]   # from the piece that excluded its fold)
    if a.xf:
        from src.data import folds
        b = json.loads((ROOT / "work/runs" / f"{a.exp}_xf0" / "args.json").read_text())
        tf.INPUT = b.get("input", "rgbhp"); tf.INIT = "none" if b.get("init", "imagenet") == "dec" else "imagenet"
        m = b["method"]; tr = np.where(methods() == m)[0]; te = np.where(methods("test") == m)[0]; Fo = folds()[tr]
        P = np.zeros((len(tr), 128, 128), np.float32); T = np.zeros((len(te), 128, 128), np.float32)
        for k in range(5):
            net = tf.Net(b.get("encoder", "resnet34")).to(dev).to(memory_format=torch.channels_last)
            net.load_state_dict(torch.load(ROOT / "work/runs" / f"{a.exp}_xf{k}" / "model.pt", map_location=dev))
            P[Fo == k] = predict_tta(net, "train", tr[Fo == k], dev, shifts); T += predict_tta(net, "test", te, dev, shifts) / 5
            print(f"piece {k} done", flush=True)
        out = ROOT / "work/runs" / f"{a.exp}_xfall{a.tag}"; out.mkdir(parents=True, exist_ok=True)
        np.save(out / "train.npy", P.astype(np.float16)); np.save(out / "test.npy", T.astype(np.float16))
        b.update(exp=out.name, fold_excluded=True, tta_shifts=shifts, tta_of=f"{a.exp}_xfall"); (out / "args.json").write_text(json.dumps(b))
        print("DONE", out.name, flush=True); return
    d = ROOT / "work/runs" / a.exp; b = json.loads((d / "args.json").read_text())
    tf.INPUT = b.get("input", "rgbhp"); tf.INIT = "none" if b.get("init", "imagenet") == "dec" else "imagenet"
    net = tf.Net(b.get("encoder", "resnet34")).to(dev).to(memory_format=torch.channels_last)
    net.load_state_dict(torch.load(d / "model.pt", map_location=dev))
    m = b["method"]; tr = np.where(methods() == m)[0]; te = np.where(methods("test") == m)[0]
    out = ROOT / "work/runs" / f"{a.exp}{a.tag}"; out.mkdir(parents=True, exist_ok=True)
    P = predict_tta(net, "train", tr, dev, shifts); np.save(out / "train.npy", P.astype(np.float16))
    ref = np.load(d / "train.npy").astype(np.float32)
    print(f"{a.exp}: mean |TTA - base| on train maps {np.abs(P - ref).mean():.4f}; identity-only check uses the same weights", flush=True)
    np.save(out / "test.npy", predict_tta(net, "test", te, dev, shifts).astype(np.float16))
    (out / "args.json").write_text(json.dumps(dict(b, exp=f"{a.exp}{a.tag}", tta_shifts=shifts, tta_of=a.exp)))
    print("DONE", out.name, flush=True)


if __name__ == "__main__":
    main()
