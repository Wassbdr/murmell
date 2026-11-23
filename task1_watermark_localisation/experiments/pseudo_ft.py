"""PL_*: self-training on the REAL TEST images.

Why: the remaining loss of PixelSeal / TrustMark is a synthetic -> real gap (unknown low-pass of the real pipeline, soft
hole borders).  The rescale fine-tuning (R_*) closes part of it with a proxy augmentation.  The 1,020 test images ARE the
real distribution: fine-tune on them with pseudo-labels from the best SYNTHETIC-ONLY models, ignoring uncertain cells.
  real stream  = test images of the method routed to the base's regime (B0 regime vector), label = mean test map of the
                 --teacher runs: 1 if p >= hi, 0 if p <= lo, ignored in between (masked BCE); random roll by multiples of
                 4 px.  Small-regime bases: decoder-verified boxes (acc >= 0.8, GT-free) replace the teacher label.
  synth stream = the base's own stream with the same wrapper as src.rescale_ft (rescale / feather / low-pass options)
No train label is used anywhere (teachers must be synthetic-only runs: R_*, D5_*, DT5_*, D2_* ... NOT FT_*)
=> the predictions on ALL real train images stay out-of-sample; same cell restriction as the base (args inherited).
"VAL step" lines = real train images of the base's regime, as src.train_full.
    python -m experiments.pseudo_ft --base R_D2_ps_amp4 --teacher R_D2_ps_amp4_tta R_D2_ps_noamp_tta --exp PL_R_D2_ps_amp4 --steps 2000 --device cuda:0
    python -m experiments.pseudo_ft --base RF_D2_tm --teacher RF_D2_tm D2_tm_tta --exp PL_RF_D2_tm --p_rescale 0 --p_feather 0.8 --feather_max 4 ...
Outputs <out_root>/<exp>/{train.npy, test.npy, model.pt, args.json}.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import src.rescale_ft as rft                    # imported BEFORE tf.to128 is patched (it keeps the original to128)
import src.train_full as tf
from src.combine2 import RE
from src.data import ROOT, image, masks128, methods, names, regime


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base"); ap.add_argument("--teacher", nargs="+"); ap.add_argument("--exp")
    ap.add_argument("--steps", type=int, default=2000); ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--bs_real", type=int, default=6); ap.add_argument("--bs_syn", type=int, default=6)
    ap.add_argument("--lo", type=float, default=0.2); ap.add_argument("--hi", type=float, default=0.8)
    ap.add_argument("--w_real", type=float, default=1.0)
    ap.add_argument("--p_rescale", type=float, default=0.85); ap.add_argument("--thin", type=int, default=0)
    ap.add_argument("--p_feather", type=float, default=0.0); ap.add_argument("--feather_max", type=float, default=5.0)
    ap.add_argument("--p_down", type=float, default=0.0); ap.add_argument("--p_blur", type=float, default=0.0); ap.add_argument("--blur_max", type=float, default=0.6)
    ap.add_argument("--p_jpeg", type=float, default=0.0); ap.add_argument("--jpeg_min", type=int, default=90)
    ap.add_argument("--boxes", nargs="*", default=[])      # small-regime bases: box sets of the method, e.g. S1_ps_amp4_grid S1_ps_amp4_grid_g2
    ap.add_argument("--device", default="cuda:0"); ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--val_every", type=int, default=500); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_root", default=str(ROOT / "work/runs")); ap.add_argument("--smoke", type=int, default=0)
    a = ap.parse_args(); dev = a.device
    b = json.loads((ROOT / "work/runs" / a.base / "args.json").read_text()); method = b["method"]; reg = b.get("regime", "full")
    tf.POOL_TAG = b.get("pool_tag", ""); tf.SHAPE_LIB = b.get("shape_lib", "train"); tf.INPUT = b.get("input", "rgbhp")
    tf.EXCLUDE_FOLD = b.get("exclude_fold", -1); tf.RESCALE_AUG = 0; tf.INIT = "none" if b.get("init", "imagenet") == "dec" else "imagenet"
    rft.PF, rft.FMAX = a.p_feather, a.feather_max
    rft.LP.update(p_down=a.p_down, p_blur=a.p_blur, blur_max=a.blur_max, p_jpeg=a.p_jpeg, jpeg_min=a.jpeg_min)
    torch.manual_seed(a.seed); rng = np.random.RandomState(a.seed)
    out = Path(a.out_root) / a.exp; out.mkdir(parents=True, exist_ok=True)
    for t in a.teacher:
        tj = json.loads((ROOT / "work/runs" / t / "args.json").read_text())
        assert tj["method"] == method, f"teacher {t} is not a {method} run"
        if t.startswith("FT") or tj.get("kind") in ("selector", "selector3", "router"):
            print(f"WARNING: teacher {t} used train labels -> the train predictions of {a.exp} are NOT strictly out-of-sample", flush=True)
    args = dict(b, exp=a.exp, base=a.base, kind_ft="pseudo_ft", teacher=a.teacher, ft_steps=a.steps, ft_lr=a.lr, lo=a.lo, hi=a.hi,
                p_rescale=a.p_rescale, p_feather=a.p_feather, lowpass=dict(rft.LP), thin=a.thin, boxes=a.boxes, base_args=b)
    (out / "args.json").write_text(json.dumps(args))
    me, mt, rg, G = methods(), methods("test"), regime(), masks128()
    tr_idx = np.where(me == method)[0]; te_idx = np.where(mt == method)[0]
    small = reg in ("small", "small2", "small3")
    val_sel = (~rg[tr_idx]) if small else (rg[tr_idx] if reg == "full" else np.ones(len(tr_idx), bool))
    # ---- pseudo-labels on the test images of the base's regime
    Tm = np.mean([np.load(ROOT / "work/runs" / t / "test.npy").astype(np.float32) for t in a.teacher], 0)      # [n_test_method,128,128]
    use = np.where(RE[te_idx] == (not small))[0] if reg != "mix" else np.arange(len(te_idx))
    Y = np.full((len(te_idx), 128, 128), -1, np.int8); Y[Tm >= a.hi] = 1; Y[Tm <= a.lo] = 0
    nbox = 0
    if small and a.boxes:
        from src.apply_boxes import cells, load_boxes
        pos = {n: j for j, n in enumerate(np.array(names("test"))[te_idx])}; best = {}
        for mp in a.boxes:
            for n, r in load_boxes(method, mp, "test").items():
                y0, y1, x0, x1 = r["box"]; h, w = y1 - y0, x1 - x0
                if r["acc"] < 0.8 or (method == "trustmark" and max(h, w) > 2.0 * min(h, w)):
                    continue
                if n not in best or r["acc"] >= best[n][1]:
                    best[n] = (cells(r["box"]), r["acc"])
        for n, (bm, _) in best.items():
            Y[pos[n]] = bm.astype(np.int8); nbox += 1
    if a.smoke:
        use = use[:8]
    lab = Y[use]; print(f"pseudo-labels: {len(use)} test images of {method} ({'small' if small else 'full'}-routed), teacher {a.teacher}: "
                        f"wm {100 * (lab == 1).mean():.1f} % clean {100 * (lab == 0).mean():.1f} % ignored {100 * (lab == -1).mean():.1f} % | boxes {nbox}", flush=True)
    imgs = {int(j): image("test", int(te_idx[j])) for j in use}
    net = tf.Net(b.get("encoder", "resnet34"))
    net.load_state_dict(torch.load(ROOT / "work/runs" / a.base / "model.pt", map_location="cpu"))
    net = net.to(dev).to(memory_format=torch.channels_last)
    vi = tr_idx[:6] if a.smoke else tr_idx; vs = val_sel[:6] if a.smoke else val_sel

    def val(step, t0):
        P = tf.predict(net, "train", vi, dev)
        s, t, _ = tf.val_report(P[vs], G[vi][vs]) if vs.any() else (float("nan"), 0, 0)
        print(f"VAL step {step} {reg}-regime mIoU {s:.4f} @thr {t:.2f} {time.time() - t0:.0f}s", flush=True)
        return P, s
    t0 = time.time(); _, s0 = val(0, t0)
    extra = rft.thin_pool(method, a.thin, dev, a.seed) if a.thin else None
    DS = {"full": tf.SynthFull, "small": tf.SynthSmall, "small3": tf.SynthSmall3, "mix": tf.SynthMix}[reg]
    if a.p_feather > 0:
        DS = {"full": rft.FeatherFull, "small": rft.FeatherSmall, "small3": tf.SynthSmall3}[reg]
    tf.to128 = lambda M: M.astype(np.float32)        # wrapped datasets return their 512-px mask (workers fork after this)
    inner = DS(method, n_epoch_items=a.steps * a.bs_syn, seed=2000 + a.seed, amp_start=1.0, amp_frac=0.5)
    if a.p_feather > 0 and reg == "small3":
        inner.rect = rft.FeatherSmall(method, a.steps * a.bs_syn, 2000 + a.seed, 1.0, 0.5); inner.rect.lib = []
    ds = rft.Rescaled(inner, a.p_rescale, a.seed, extra)
    dl = iter(DataLoader(ds, batch_size=a.bs_syn, num_workers=a.workers, shuffle=False, drop_last=True, prefetch_factor=4))
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps, pct_start=0.1)
    net.train(); run = rr = 0; s = s0
    for step in range(1, a.steps + 1):
        xs, ys = next(dl); xr, yr = [], []
        for j in rng.choice(use, a.bs_real):
            dy, dx = rng.randint(128), rng.randint(128)
            xr.append(np.roll(imgs[int(j)], (4 * dy, 4 * dx), (0, 1))); yr.append(np.roll(Y[j], (dy, dx), (0, 1)))
        xr = torch.from_numpy(np.stack(xr)).permute(0, 3, 1, 2); yr = torch.from_numpy(np.stack(yr))[:, None].to(dev)
        x = torch.cat([xs, xr]).to(dev); ys = ys.to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled="cuda" in dev):
            lg = net(x).float()
        ls = tf.loss_fn(lg[:len(xs)], ys)
        w = (yr >= 0).float(); lr_ = (F.binary_cross_entropy_with_logits(lg[len(xs):], yr.clamp(min=0).float(), reduction="none") * w).sum() / w.sum().clamp(min=1)
        loss = ls + a.w_real * lr_
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
        run = 0.98 * run + 0.02 * ls.item() if step > 1 else ls.item(); rr = 0.98 * rr + 0.02 * lr_.item() if step > 1 else lr_.item()
        if step % 100 == 0 or a.smoke:
            print(f"step {step} synth loss {run:.4f} pseudo loss {rr:.4f} {time.time() - t0:.0f}s", flush=True)
        if step % a.val_every == 0 or step == a.steps:
            P, s = val(step, t0)
            if not a.smoke:
                np.save(out / "train.npy", P.astype(np.float16))
            torch.save(net.state_dict(), out / "model.pt")
    if not a.smoke:
        np.save(out / "test.npy", tf.predict(net, "test", te_idx, dev).astype(np.float16))
    print(f"DONE {a.exp}: {reg}-regime mIoU base {s0:.4f} -> {s:.4f} {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
