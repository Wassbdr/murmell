"""Per-fold fine-tuning of a synthetic-trained U-Net on REAL train images (human list 20:25).
For outer fold k: start from work/runs/<base>/model.pt, fine-tune at low LR on real images of <method> and of the base's
regime (full / small, TRUE regime label: these are training targets) from the 4 other folds, mixed 50/50 with the base's
synthetic stream (same pools / input / amp=1), whose shape library EXCLUDES fold k.  Predict fold k only.
=> every train prediction is out-of-fold (no fold-k pixel, mask or shape seen) => args fold_excluded=true ("any" cell).
Test map = mean of the 5 fold models.  Real augmentation: random roll by multiples of 4 px (keeps the 4x4 cell grid).
    python -m src.finetune_real --base F2_ps_amp4 --exp FT_F2_ps_amp4 --steps 1500 --lr 3e-5 --device cuda:0
"""
import argparse
import json
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

import src.train_full as tf
from src.data import ROOT, folds, image, masks128, methods, regime


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base"); ap.add_argument("--exp"); ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=3e-5); ap.add_argument("--bs_real", type=int, default=6)
    ap.add_argument("--bs_syn", type=int, default=6); ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--workers", type=int, default=12); ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--tta", type=int, default=0)       # 1: held-out and test predictions with the 0/4/8-px shift TTA (src.tta)
    ap.add_argument("--base_xf", type=int, default=0)   # 1: fold k starts from <base>_xf<k>/model.pt (base never saw fold-k shapes)
    a = ap.parse_args(); dev = a.device
    b = json.loads((ROOT / "work/runs" / a.base / "args.json").read_text())
    tf.POOL_TAG = b.get("pool_tag", ""); tf.SHAPE_LIB = b.get("shape_lib", "train"); tf.INPUT = b.get("input", "rgbhp")
    tf.RESCALE_AUG = b.get("rescale_aug", 0); tf.INIT = "none" if b.get("init", "imagenet") == "dec" else "imagenet"   # weights come from model.pt
    method, reg = b["method"], b.get("regime", "full")       # runs older than the --regime flag are full-regime
    out = ROOT / "work/runs" / a.exp; out.mkdir(parents=True, exist_ok=True)
    args = dict(vars(a), method=method, regime=reg, base_args=b, fold_excluded=True, base_fold_clean=bool(a.base_xf), pool_tag=tf.POOL_TAG,
                input=tf.INPUT, shape_lib=tf.SHAPE_LIB)
    (out / "args.json").write_text(json.dumps(args))
    me, rg, G, Fo = methods(), regime(), masks128(), folds()
    mi = np.where(me == method)[0]; te_idx = np.where(methods("test") == method)[0]
    reg_ok = {"full": rg, "small": ~rg, "small2": ~rg, "small3": ~rg, "mix": np.ones_like(rg)}[reg]
    P = np.full((len(mi), 128, 128), np.nan, np.float32); T = np.zeros((len(te_idx), 128, 128), np.float32)
    fl = [int(x) for x in a.folds.split(",")]
    imgs = {int(i): image("train", int(i)) for i in mi}
    t0 = time.time()
    for k in fl:
        torch.manual_seed(k)
        tf.EXCLUDE_FOLD = k
        pool = [int(i) for i in mi if Fo[i] != k and reg_ok[i]]
        DS = {"full": tf.SynthFull, "small": tf.SynthSmall, "small2": tf.SynthSmall2, "small3": tf.SynthSmall3, "mix": tf.SynthMix}[reg]
        ds = DS(method, n_epoch_items=a.steps * a.bs_syn, seed=100 + k, amp_start=1.0, amp_frac=0.5)
        dl = iter(DataLoader(ds, batch_size=a.bs_syn, num_workers=a.workers, shuffle=False, drop_last=True,
                             prefetch_factor=4))
        net = tf.Net(b.get("encoder", "resnet34")).to(dev).to(memory_format=torch.channels_last)
        net.load_state_dict(torch.load(ROOT / "work/runs" / (f"{a.base}_xf{k}" if a.base_xf else a.base) / "model.pt", map_location=dev))
        opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps, pct_start=0.1)
        rng = np.random.RandomState(1000 + k); run = 0
        for step in range(1, a.steps + 1):
            xs, ys = next(dl)
            xr, yr = [], []
            for i in rng.choice(pool, a.bs_real):
                dy, dx = rng.randint(128), rng.randint(128)
                xr.append(np.roll(imgs[i], (4 * dy, 4 * dx), (0, 1))); yr.append(np.roll(G[i], (dy, dx), (0, 1)))
            xr = torch.from_numpy(np.stack(xr)).permute(0, 3, 1, 2); yr = torch.from_numpy(np.stack(yr).astype(np.float32))[:, None]
            x = torch.cat([xs, xr]).to(dev); y = torch.cat([ys, yr]).to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg = net(x)
            loss = tf.loss_fn(lg.float(), y)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
            run = 0.98 * run + 0.02 * loss.item() if step > 1 else loss.item()
            if step % 250 == 0:
                print(f"fold {k} step {step} loss {run:.4f} {time.time() - t0:.0f}s", flush=True)
        ev = np.where(Fo[mi] == k)[0]
        if a.tta:
            from src.tta import predict_tta
            P[ev] = predict_tta(net, "train", mi[ev], dev, [0, 4, 8]); T += predict_tta(net, "test", te_idx, dev, [0, 4, 8]) / len(fl)
        else:
            P[ev] = tf.predict(net, "train", mi[ev], dev)
            T += tf.predict(net, "test", te_idx, dev) / len(fl)
        sel = ev[reg_ok[mi[ev]]]
        s, t, _ = tf.val_report(P[sel], G[mi[sel]])
        print(f"FOLD {k} held-out {reg}-regime mIoU {s:.4f} @thr {t:.2f} (n={len(sel)}) {time.time() - t0:.0f}s", flush=True)
        del dl, ds
    done = ~np.isnan(P[:, 0, 0])
    sel = np.where(done & reg_ok[mi])[0]
    s, t, _ = tf.val_report(P[sel], G[mi[sel]])
    print(f"DONE all-fold held-out {reg}-regime mIoU {s:.4f} @thr {t:.2f} {time.time() - t0:.0f}s", flush=True)
    if done.all():
        np.save(out / "train.npy", P.astype(np.float16)); np.save(out / "test.npy", T.astype(np.float16))


if __name__ == "__main__":
    main()
