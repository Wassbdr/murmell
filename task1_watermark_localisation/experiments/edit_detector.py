"""E1: EDITED-ZONE detector common to the three methods, trained fold by fold on REAL train images (human 23:45).
Why: 35/85 PixelSeal holes are missed, 28/43 PixelSeal rectangles are unreadable by the decoder and transplanted objects
cannot be decoded on their box => we need a signal that does not depend on the watermark: inpainting / outpainting /
pasting leave traces (noise, spectrum, 8-px VAE grid, seams).
Targets (128x128 cells; GT mask + true regime are TRAINING labels only):
  ch0 = edited zone : full regime -> the hole (~mask) | small rectangle (outpainting) -> outside the rectangle (~mask)
                      | small object (transplant) -> the pasted object (mask)
  ch1 = watermark mask itself (same supervision; directly usable as a source by the nested selection)
Input: full-resolution RGB + 5x5 high-pass residual x8 (6 ch) + method one-hot (3 constant planes).
Model: smp U-Net resnet34 (ImageNet), logits 512 -> avg-pool 4 -> 128; loss per channel = BCE + soft mIoU.
FIXED hyper-parameters (set in advance, never tuned): 3000 steps, bs 12, AdamW 3e-4 OneCycle, augmentation = h-flip +
roll by multiples of 8 px (keeps a possible 8-px grid aligned), seed = fold.
Fold k: train on the real images of ALL methods with fold != k (408), predict fold k (out-of-fold) and the test set.
    python -m experiments.edit_detector --fold 0 --device cuda:0      -> work/runs/E1_f0/{train.npy [510,2,128,128] NaN outside
                                                                 fold 0, test.npy [1020,2,128,128], args.json}
    python -m experiments.edit_detector --assemble 1                  -> runs E1w_<m> (ch1) and E1n_<m> (1 - ch0), train = OOF,
                                                                 test = mean of the 5 folds, fold_excluded=true
"""
import argparse
import json
import time

import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F

import src.train_full as tf
from src.data import METHODS, ROOT, folds, image, masks128, methods, regime

SHORT = {"pixelseal": "ps", "trustmark": "tm", "stegastamp": "ss"}


def is_rect(g):
    if not g.any():
        return True
    ys, xs = np.where(g)
    return g.sum() / ((np.ptp(ys) + 1) * (np.ptp(xs) + 1)) >= 0.95


def targets():
    """[510, 2, 128, 128] float32: ch0 edited zone, ch1 watermark mask; types: 0 full, 1 small rect, 2 small object."""
    G, rg = masks128().astype(bool), regime(); T = np.zeros((510, 2, 128, 128), np.float32); typ = np.zeros(510, int)
    for i in range(510):
        if rg[i]:
            e = ~G[i]
        elif is_rect(G[i]):
            e = ~G[i]; typ[i] = 1
        else:
            e = G[i]; typ[i] = 2
        T[i, 0] = e; T[i, 1] = G[i]
    return T, typ


class Net(nn.Module):
    def __init__(self):
        super().__init__(); self.m = smp.Unet("resnet34", encoder_weights="imagenet", in_channels=9, classes=2)

    def forward(self, x_u8, mid):
        x = tf.make_input(x_u8)                                              # rgb + high-pass (INPUT = rgbhp)
        oh = F.one_hot(mid, 3).float()[:, :, None, None].expand(-1, -1, x.shape[2], x.shape[3])
        return F.avg_pool2d(self.m(torch.cat([x, oh], 1)), 4)                # 2 x 128 x 128 logits


def loss2(lg, y):
    return tf.loss_fn(lg[:, :1], y[:, :1]) + tf.loss_fn(lg[:, 1:], y[:, 1:])


@torch.no_grad()
def predict(net, X, mid, dev, bs=24):
    net.eval(); out = []
    for s in range(0, len(X), bs):
        x = torch.from_numpy(np.stack(X[s:s + bs])).permute(0, 3, 1, 2).to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out.append(torch.sigmoid(net(x, torch.from_numpy(mid[s:s + bs]).to(dev)).float()).cpu().numpy())
    net.train(); return np.concatenate(out)


def auc(v, y):
    from scipy.stats import rankdata
    y = y.ravel() > 0.5; n1 = y.sum(); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    return (rankdata(v.ravel())[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def report(P, T, typ, me, idx, tag):
    """P: [len(idx), 2, 128, 128] predictions for images idx."""
    from src.metric import miou_batch
    names = ["full(hole)", "small-rect", "small-object"]
    for m in METHODS:
        for t in range(3):
            sel = [j for j, i in enumerate(idx) if me[i] == m and typ[i] == t]
            if not sel:
                continue
            a0 = np.nanmean([auc(P[j, 0], T[idx[j], 0]) for j in sel]); a1 = np.nanmean([auc(P[j, 1], T[idx[j], 1]) for j in sel])
            mi = miou_batch(P[sel, 1] >= 0.5, T[[idx[j] for j in sel], 1].astype(np.uint8)).mean()
            print(f"{tag} {m:10s} {names[t]:12s} n={len(sel):3d} | AUC edited-zone {a0:.3f} | AUC wm-mask {a1:.3f} | mIoU(wm ch >= .5) {mi:.3f}", flush=True)


def train_fold(a):
    dev = a.device; k = a.fold; torch.manual_seed(k); rng = np.random.RandomState(k)
    tf.INPUT = "rgbhp"
    T, typ = targets(); me, Fo = methods(), folds(); mid_all = np.array([METHODS.index(m) for m in me])
    tr = np.where(Fo != k)[0]; ev = np.where(Fo == k)[0]
    imgs = {int(i): image("train", int(i)) for i in range(510)}
    net = Net().to(dev).to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps, pct_start=0.05)
    t0 = time.time(); run = 0
    for step in range(1, a.steps + 1):
        b = rng.choice(tr, a.bs); xs, ys = [], []
        for i in b:
            x, y = imgs[int(i)], T[i]
            dy, dx = 8 * rng.randint(64), 8 * rng.randint(64)
            x = np.roll(x, (dy, dx), (0, 1)); y = np.roll(y, (dy // 4, dx // 4), (1, 2))
            if rng.rand() < 0.5:
                x = x[:, ::-1]; y = y[:, :, ::-1]
            xs.append(np.ascontiguousarray(x)); ys.append(np.ascontiguousarray(y))
        x = torch.from_numpy(np.stack(xs)).permute(0, 3, 1, 2).to(dev); y = torch.from_numpy(np.stack(ys)).to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg = net(x, torch.from_numpy(mid_all[b]).to(dev))
        loss = loss2(lg.float(), y)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
        run = 0.98 * run + 0.02 * loss.item() if step > 1 else loss.item()
        if step % 250 == 0:
            print(f"fold {k} step {step} loss {run:.4f} {time.time() - t0:.0f}s", flush=True)
        if step in (1000, 2000) or step == a.steps:
            Pe = predict(net, [imgs[int(i)] for i in ev], mid_all[ev], dev)
            report(Pe, T, typ, me, ev, f"HELD-OUT fold {k} step {step}")
    out = ROOT / "work/runs" / f"E1{a.tag}_f{k}"; out.mkdir(parents=True, exist_ok=True)
    P = np.full((510, 2, 128, 128), np.nan, np.float16); P[ev] = Pe.astype(np.float16)
    np.save(out / "train.npy", P)
    mt = methods("test"); midt = np.array([METHODS.index(m) for m in mt]); Pt = []
    for s in range(0, 1020, 60):
        Pt.append(predict(net, [image("test", i) for i in range(s, min(1020, s + 60))], midt[s:s + 60], dev))
    np.save(out / "test.npy", np.concatenate(Pt).astype(np.float16))
    (out / "args.json").write_text(json.dumps(dict(vars(a), kind="edit_fold", exp=f"E1_f{k}")))
    print(f"DONE fold {k} {time.time() - t0:.0f}s", flush=True)


def assemble():
    me, mt, Fo = methods(), methods("test"), folds()
    P = np.full((510, 2, 128, 128), np.nan, np.float32); Pt = np.zeros((1020, 2, 128, 128), np.float32)
    for k in range(5):
        d = ROOT / "work/runs" / f"E1_f{k}"; p = np.load(d / "train.npy").astype(np.float32)
        P[Fo == k] = p[Fo == k]; Pt += np.load(d / "test.npy").astype(np.float32) / 5
    assert np.isfinite(P).all(), "missing folds"
    T, typ = targets(); report(P, T, typ, me, np.arange(510), "OOF all folds")
    for m in METHODS:
        for tag, tr_, te_ in (("E1w", P[me == m, 1], Pt[mt == m, 1]), ("E1n", 1 - P[me == m, 0], 1 - Pt[mt == m, 0])):
            d = ROOT / "work/runs" / f"{tag}_{SHORT[m]}"; d.mkdir(parents=True, exist_ok=True)
            np.save(d / "train.npy", tr_.astype(np.float16)); np.save(d / "test.npy", te_.astype(np.float16))
            (d / "args.json").write_text(json.dumps({"method": m, "exp": d.name, "regime": "full", "fold_excluded": True,
                                                     "kind": "edit_detector", "channel": "watermark" if tag == "E1w" else "1 - edited zone"}))
    print("assembled E1w_* / E1n_*", flush=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--fold", type=int, default=0); ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--steps", type=int, default=3000); ap.add_argument("--bs", type=int, default=12)
    ap.add_argument("--lr", type=float, default=3e-4); ap.add_argument("--assemble", type=int, default=0)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    assemble() if a.assemble else train_fold(a)


if __name__ == "__main__":
    main()
