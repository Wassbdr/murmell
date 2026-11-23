"""Small 128x128 stacking combiner on OUT-OF-FOLD maps, evaluated in fully nested CV.
Channels per image (restricted like C4r: full-trained map only on predicted-full images, small-trained only on
predicted-small): [regime-appropriate synthetic map, snapDF map, box mask (TM decoder-ranked, acc>=0.8), box acc (const),
method one-hot (3), predicted regime (const)].  Net: 5 conv layers (32 ch, dilations 1,2,4,8,1), BCE + soft per-image mIoU.
Outer fold k: train on the 4 other folds (inner 4-fold cross-fit gives out-of-sample predictions for the threshold per
method x predicted regime), predict fold k. Test: model trained on all 510 (5 seeds averaged), thresholds from outer OOF.
    python -m experiments.stack --name ST1 --device cuda:1
"""
import argparse
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.apply_boxes import cells, load_boxes
from src.combine2 import RE, RT, src
from src.data import METHODS, ROOT, folds, masks128, methods, names
from src.eval_cv import THRS
from src.metric import miou_batch
from src.nested_cv import summarize

FULL = {"pixelseal": "F2_ps_amp4", "trustmark": "F1_tm_long", "stegastamp": "F1_ss"}
SMALL = {"pixelseal": "S1_ps_amp4", "trustmark": "S1_tm", "stegastamp": "S1_ss"}
BOXES = {"trustmark": "S1_tm"}


def features(split):
    me = methods(split); reg = RT if split == "train" else RE; N = len(me); X = np.zeros((N, 9, 128, 128), np.float32)
    nm = names(split); ix = {n: i for i, n in enumerate(nm)}
    for m in METHODS:
        mi = np.where(me == m)[0]; k = 0 if split == "train" else 1
        fmap, smap, dmap = src(FULL[m], m)[k], src(SMALL[m], m)[k], src("snapDF", m)[k]
        r = reg[mi]
        X[mi, 0] = np.where(r[:, None, None], fmap, smap); X[mi, 1] = dmap
        X[mi, 4 + METHODS.index(m)] = 1.0
        if m in BOXES:
            for n, b in load_boxes(m, BOXES[m], split).items():
                i = ix[n]
                if b["acc"] >= 0.8:
                    X[i, 2] = cells(b["box"]); X[i, 3] = b["acc"]
    X[:, 7] = reg[:, None, None].astype(np.float32)
    X[:, 8] = 0.0
    return X


class Net(nn.Module):
    def __init__(self, c=9, w=32):
        super().__init__()
        L = []; cin = c
        for d in (1, 2, 4, 8, 1):
            L += [nn.Conv2d(cin, w, 3, padding=d, dilation=d), nn.GroupNorm(4, w), nn.GELU()]; cin = w
        self.body = nn.Sequential(*L); self.head = nn.Conv2d(w + c, 1, 1)

    def forward(self, x):
        return self.head(torch.cat([self.body(x), x], 1))


def loss_fn(lg, y):
    p = torch.sigmoid(lg)
    i1 = (p * y).sum((1, 2, 3)); u1 = (p + y - p * y).sum((1, 2, 3))
    i0 = ((1 - p) * (1 - y)).sum((1, 2, 3)); u0 = ((1 - p) + (1 - y) - (1 - p) * (1 - y)).sum((1, 2, 3))
    return F.binary_cross_entropy_with_logits(lg, y) + (1 - 0.5 * (i1 / (u1 + 1) + i0 / (u0 + 1))).mean()


def fit_predict(Xtr, Ytr, Xev, dev, seed=0, epochs=60):
    torch.manual_seed(seed); net = Net().to(dev); opt = torch.optim.AdamW(net.parameters(), 2e-3, weight_decay=1e-4)
    xt, yt = torch.from_numpy(Xtr).to(dev), torch.from_numpy(Ytr[:, None].astype(np.float32)).to(dev)
    n = len(xt); steps = epochs * ((n + 15) // 16); sch = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=steps)
    g = torch.Generator(device="cpu").manual_seed(seed)
    for _ in range(epochs):
        perm = torch.randperm(n, generator=g)
        for j in range(0, n, 16):
            b = perm[j:j + 16]; xb, yb = xt[b], yt[b]
            if torch.rand(1, generator=g).item() < 0.5:          # flips are fine at the map level
                xb, yb = xb.flip(-1), yb.flip(-1)
            loss = loss_fn(net(xb), yb); opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    net.eval()
    with torch.no_grad():
        return torch.sigmoid(net(torch.from_numpy(Xev).to(dev))).cpu().numpy()[:, 0]


def groups(split):
    reg = RT if split == "train" else RE
    return np.array([f"{m}|{'F' if r else 'S'}" for m, r in zip(methods(split), reg)])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--name", default="ST1"); ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--epochs", type=int, default=60)
    a = ap.parse_args(); dev = a.device
    X, G, Fo, grp = features("train"), masks128(), folds(), groups("train")
    P = np.zeros((510, 128, 128), np.float32); s = np.zeros(510)
    for k in range(5):
        tr, te = np.where(Fo != k)[0], np.where(Fo == k)[0]
        # inner cross-fit on the 4 training folds => out-of-sample predictions for threshold selection
        Pin = np.zeros((len(tr), 128, 128), np.float32)
        for k2 in sorted(set(Fo[tr])):
            a_, b_ = tr[Fo[tr] != k2], np.where(Fo[tr] == k2)[0]
            Pin[b_] = fit_predict(X[a_], G[a_], X[tr[b_]], dev, epochs=a.epochs)
        P[te] = fit_predict(X[tr], G[tr], X[te], dev, epochs=a.epochs)
        for gname in sorted(set(grp)):
            gi, go = grp[tr] == gname, grp[te] == gname
            if not go.any():
                continue
            t = max(THRS, key=lambda t: miou_batch(Pin[gi] >= t, G[tr][gi]).mean())
            s[te[go]] = miou_batch(P[te][go] >= t, G[te][go])
        print(f"outer fold {k}: {s[te].mean():.4f}", flush=True)
    rep = summarize(s); rep.update(clean=True, kind="stack", inputs="restricted maps + snapDF + TM boxes")
    (ROOT / "work/nested" / f"{a.name}.json").write_text(json.dumps(rep, indent=1)); np.save(ROOT / "work/nested" / f"{a.name}.npy", s)
    np.save(ROOT / "work/oof" / f"{a.name}.npy", P.astype(np.float16))
    print(f"{a.name}: nested CV {rep['cv_foldmean']:.6f} ± {rep['cv_std']:.4f} | " + " ".join(f"{x:.4f}" for x in rep["folds"]))
    for kk, v in rep["cells"].items():
        print(f"  {kk:14s} n={v['n']:3d} {v['mean']:.4f}")
    # test: train on all 510 (3 seeds), thresholds per group from the outer OOF predictions
    Xt = features("test"); T = np.mean([fit_predict(X, G, Xt, dev, seed=sd, epochs=a.epochs) for sd in range(3)], 0)
    gt = groups("test"); B = np.zeros(T.shape, bool); thr = {}
    for gname in sorted(set(grp)):
        sel = grp == gname; thr[gname] = float(max(THRS, key=lambda t: miou_batch(P[sel] >= t, G[sel]).mean()))
        B[gt == gname] = T[gt == gname] >= thr[gname]
    np.save(ROOT / "work/test" / f"{a.name}.npy", T.astype(np.float16)); np.save(ROOT / "work/test" / f"{a.name}_bin.npy", B)
    print("test thresholds", thr, "wm frac", round(float(B.mean()), 3))


if __name__ == "__main__":
    main()
