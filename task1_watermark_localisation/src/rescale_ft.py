"""R_*: fine-tune a synthetic-trained model on its OWN synthetic stream + the GLOBAL CROP & RESCALE of the real images.

Finding: real images went through a global crop + resize (scale 1.03-1.15, teammate's geometry fit).  The synthetic
pools reproduce the geometry (canvas / own-size embedding) but NOT the resampling, and every PS/TM run has rescale_aug=0.
Applying that resampling to synthetic items makes the existing models fall exactly to their real-image level:
  D2_ps_amp4, synthetic full items with the REAL hole masks: mIoU .823 -> .707 with rescale; real = .708 (AUC .979/.939/.937)
  D2_tm                                                    : mIoU .769 -> .732 with rescale; real = .726
  DT2_ps_amp4, synthetic thin rectangles                   : AUC  .874 -> .766 with rescale; real = .741
So the synthetic->real gap of PixelSeal / TrustMark is the resampling blur, and it can be trained away without labels.

This script = same data stream as the base run (same pools, hole sampler, shape library, exclude_fold => same leakage
status and same run_regime() restriction as the base) with, for a fraction p of the items, a random crop of side 512/s,
s ~ U[1.0, 1.2], resized back to 512 with a random kernel (bilinear / bicubic / lanczos); the 512-px mask follows.
Optional --thin N (small-regime bases): N extra rectangle items generated at start-up with the official process
(rectangle embedded at its own size, area log-U[0.2 %, 25 %], aspect up to 16:1 for PS/SS, 2:1 TM), mixed 50/50.
Synthetic-only: no real label is used => predictions on all real train images stay out-of-sample.

    python -m src.rescale_ft --base D2_ps_amp4 --exp R_D2_ps_amp4 --steps 2000 --device cuda:0
    python -m src.rescale_ft --base DT2_ps_amp4 --exp R_DT2_ps_amp4 --steps 2000 --thin 3000 --device cuda:0
    python -m src.rescale_ft --base D2_ps_amp4_xf3 --exp R_D2_ps_amp4_xf3 ...     (fold-clean: exclude_fold is inherited)
    python -m src.rescale_ft --base D2_tm --exp RF_D2_tm --steps 2000 --p_rescale 0 --p_feather 0.8     (soft hole borders)
Outputs <out_root>/<exp>/{train.npy, test.npy, model.pt, args.json}; log lines "VAL step ..." as src.train_full.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

import src.train_full as tf
from src.data import ROOT, image, masks128, methods, regime
from src.metric import miou_batch

KER = [Image.Resampling.BILINEAR, Image.Resampling.BICUBIC, Image.Resampling.LANCZOS]
_to128 = tf.to128


def crop_rescale(y, M, rng, smin=1.0, smax=1.2):
    """y uint8 HxWx3, M bool HxW -> same after a random crop of side 512/s resized back to 512 (random kernel)."""
    sc = rng.uniform(smin, smax); c = int(round(512 / sc)); oy, ox = rng.randint(0, 513 - c), rng.randint(0, 513 - c)
    y = np.asarray(Image.fromarray(np.ascontiguousarray(y[oy:oy + c, ox:ox + c])).resize((512, 512), KER[rng.randint(3)]))
    M = np.asarray(Image.fromarray((M[oy:oy + c, ox:ox + c] * 255).astype(np.uint8)).resize((512, 512), Image.Resampling.BILINEAR)) > 127
    return y, M


LP = dict(p_down=0.0, p_blur=0.0, blur_max=0.6, p_jpeg=0.0, jpeg_min=90)     # extra low-pass family, set from the CLI (all off by default)
DOWN = [Image.Resampling.BOX, Image.Resampling.BILINEAR, Image.Resampling.BICUBIC, Image.Resampling.LANCZOS]


def lowpass(y, rng):
    """mild low-pass family applied to the IMAGE only (masks unchanged).  Audit on fresh embeddings with the real hole masks:
    D2_ps_amp4 .80 -> .73 (blur sigma .5), .68 (jpeg q95), .48-.68 (2x larger original downscaled, by kernel); real .708.
    D2_tm .75-.78 -> .72-.75 (blur .8), .71-.72 (jpeg q95); real .726.  The real operation is unknown => train on the family."""
    if rng.rand() < LP["p_down"]:                       # resampling round trip: up by f, back down with a random kernel
        f = rng.uniform(1.3, 2.0); S = int(round(512 * f))
        y = np.asarray(Image.fromarray(np.ascontiguousarray(y)).resize((S, S), Image.Resampling.BICUBIC).resize((512, 512), DOWN[rng.randint(4)]))
    if rng.rand() < LP["p_blur"]:
        from scipy import ndimage as ndi
        sg = rng.uniform(0.2, LP["blur_max"])
        y = np.clip(ndi.gaussian_filter(y.astype(np.float32), (sg, sg, 0)), 0, 255).round().astype(np.uint8)
    if rng.rand() < LP["p_jpeg"]:
        import io
        bio = io.BytesIO(); Image.fromarray(np.ascontiguousarray(y)).save(bio, "JPEG", quality=int(rng.randint(LP["jpeg_min"], 100)))
        y = np.asarray(Image.open(io.BytesIO(bio.getvalue())).convert("RGB"))
    return y


class Rescaled(Dataset):
    """wraps a train_full dataset (which must return its 512-px mask: tf.to128 is replaced by identity below) and an
    optional in-memory pool of extra rectangle items; applies the global crop & rescale with probability p."""

    def __init__(self, ds, p, seed, extra=None):
        self.ds, self.p, self.seed, self.extra = ds, p, seed, extra

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        rng = np.random.RandomState((self.seed * 9176 + i * 7919 + 5) % 2 ** 31)
        if self.extra is not None and rng.rand() < 0.5:
            k = rng.randint(len(self.extra[0])); y, M = self.extra[0][k], self.extra[1][k]
        else:
            x, m = self.ds[i]; y = x.permute(1, 2, 0).numpy(); M = m[0].numpy() > 0.5
        if rng.rand() < self.p:
            y, M = crop_rescale(y, M, rng)
        y = lowpass(y, rng)
        return torch.from_numpy(np.ascontiguousarray(y)).permute(2, 0, 1), torch.from_numpy(_to128(M))[None]


PF, FMAX = 0.0, 5.0        # feathered compositing: probability and max sigma (px); set from --p_feather / --feather_max


def _feather(b, yk, M, rng):
    """soft compositing y = b + a (yk - b), a = gaussian-blurred mask (sigma ~ U[0.5, FMAX] px); the label stays the hard mask.
    Audit (TrustMark full, D2_tm, real hole masks): hard compositing .776, sigma 2 px .735, sigma 4 px .726; REAL .726, and
    the p-profile across the hole border of real images matches sigma 2-4 px."""
    from scipy import ndimage as ndi
    al = ndi.gaussian_filter(M.astype(np.float32), rng.uniform(0.5, FMAX))[..., None]
    return np.clip(b.astype(np.float32) + al * (yk.astype(np.float32) - b), 0, 255).round().astype(np.uint8)


class FeatherFull(tf.SynthFull):
    """SynthFull with soft hole borders for a fraction PF of the items (amp curriculum not used: fine-tuning at amp 1)."""

    def __getitem__(self, i):
        rng = np.random.RandomState((self.seed * 7919 + i * 104729) % 2 ** 31)
        pi, k = self.idx[rng.randint(len(self.idx))]; _, Y, sp, nm = self.pools[pi]; key = (str(sp[k]), str(nm[k]))
        if key not in self.bcache:
            if len(self.bcache) > 400:
                self.bcache.clear()
            self.bcache[key] = image(*key)
        b = self.bcache[key]; yk = np.asarray(Y[k]); H = tf.sample_holes(rng, self.lib); M = ~H
        y = _feather(b, yk, M, rng) if rng.rand() < PF else np.where(H[..., None], b, yk)
        return torch.from_numpy(np.ascontiguousarray(y)).permute(2, 0, 1), torch.from_numpy(M.astype(np.float32))[None]


class FeatherSmall(tf.SynthSmall):
    """SynthSmall (rectangle / object shape inside the watermarked rectangle) with soft borders for a fraction PF."""

    def __getitem__(self, i):
        rng = np.random.RandomState((self.seed * 7919 + i * 104729 + 17) % 2 ** 31)
        pi, k = self.idx[rng.randint(len(self.idx))]; Y, sp, nm, rect = self.pools[pi]; key = (str(sp[k]), str(nm[k]))
        if key not in self.bcache:
            if len(self.bcache) > 400:
                self.bcache.clear()
            self.bcache[key] = image(*key)
        b = self.bcache[key]; yk = np.asarray(Y[k]); x0, y0, w, h = [int(v) for v in rect[k]]
        M = np.zeros((512, 512), bool)
        if rng.rand() < 0.5 or not self.lib:
            M[y0:y0 + h, x0:x0 + w] = True
        else:
            o = self.lib[rng.randint(len(self.lib))]
            if rng.rand() < 0.5:
                o = o[:, ::-1]
            o = np.asarray(torch.nn.functional.interpolate(torch.from_numpy(o[None, None].astype(np.float32)).contiguous(), size=(h, w),
                                                           mode="bilinear")[0, 0] > 0.5)
            M[y0:y0 + h, x0:x0 + w] = o
            if M.sum() < 16:
                M[y0:y0 + h, x0:x0 + w] = True
        y = _feather(b, yk, M, rng) if rng.rand() < PF else np.where(M[..., None], yk, b)
        return torch.from_numpy(np.ascontiguousarray(y)).permute(2, 0, 1), torch.from_numpy(M.astype(np.float32))[None]


class ObjMix(Dataset):
    """With probability p_obj the item is an IN-PLACE transplanted object: an object-shaped fragment (0.3 %-35 % of the
    frame, shapes = hole_library(), i.e. full-regime hole components; no small-regime GT shape) of the WHOLE-IMAGE
    watermark of a full pool item on its own clean base.  Same watermark scale as the full-regime items, opposite area
    regime => one model for 'whole-image-scale watermark present here?' at every watermarked fraction.
    Why: on real small OBJECTS the full-type models beat the small-type ones (TM .736 vs .664, PS .75 vs .68) although they
    never saw a mostly-clean image; small3 models must also learn rectangles, whose watermark lives at another scale."""

    def __init__(self, inner, method, n, seed, p_obj):
        self.inner, self.p, self.seed = inner, p_obj, seed
        self.t3 = tf.SynthSmall3(method, n, seed, 1.0, 0.5)

    def __len__(self):
        return len(self.inner)

    def __getitem__(self, i):
        rng = np.random.RandomState((self.seed * 3571 + i * 15485863 + 11) % 2 ** 31)
        if rng.rand() >= self.p:
            return self.inner[i]
        t3 = self.t3; pi, k = t3.idx[rng.randint(len(t3.idx))]; Y, sp, nm = t3.pools[pi]
        b = t3._base(sp[k], nm[k]); yk = np.asarray(Y[k]); om, y0, x0 = t3._shape(rng, 0.003, 0.35); h, w = om.shape
        M = np.zeros((512, 512), bool); M[y0:y0 + h, x0:x0 + w] = om
        y = _feather(b, yk, M, rng) if rng.rand() < PF else np.where(M[..., None], yk, b)
        return torch.from_numpy(np.ascontiguousarray(y)).permute(2, 0, 1), torch.from_numpy(M.astype(np.float32))[None]


class SamObjMix(Dataset):
    """Sun 09:20. With probability p the item is a transplanted object whose silhouette is a SAM segment of a real image
    (label-free: work/sam/*.npz = SAM ViT-H automatic masks of the train and test IMAGES, no ground truth): half IN PLACE
    (the whole-image watermark of a full pool item kept only inside one segment of its own clean base), half PASTED (that
    segment of the watermarked image pasted on another clean image, with a clean distractor segment, label 0).
    Why: the hole-library blobs of SynthSmall3 never coincide with a visible object, whereas a real transplanted object is
    one visible object (SAM snapping as a post-process gains +0.06 to +0.13 on such images); here the segmenter itself
    learns that the watermark boundary follows the object contour."""

    def __init__(self, inner, method, n, seed, p):
        from src.data import names
        self.inner, self.p, self.seed = inner, p, seed
        self.t3 = tf.SynthSmall3(method, n, seed, 1.0, 0.5)
        self.ix = {sp: {nm: i for i, nm in enumerate(names(sp))} for sp in ("train", "test")}; self.seg = {}
        for sp in ("train", "test"):
            D = [np.load(ROOT / "work/sam" / f"{sp}_{k}of4.npz") for k in range(4)]
            idx = np.concatenate([d["idx"] for d in D]); bits = np.concatenate([d["bits"] for d in D]); ar = np.concatenate([d["area"] for d in D]) / 262144.0
            keep = (ar >= 0.008) & (ar <= 0.40); idx, bits = idx[keep], bits[keep]; o = np.argsort(idx, kind="stable"); idx, bits = idx[o], bits[o]
            N = len(self.ix[sp]); self.seg[sp] = (bits, np.searchsorted(idx, np.arange(N)), np.searchsorted(idx, np.arange(N), side="right"))
        print("sam items: segments", {sp: len(v[0]) for sp, v in self.seg.items()}, "p", p, flush=True)

    def __len__(self):
        return len(self.inner)

    def _seg(self, rng, sp, nm):
        from scipy import ndimage as ndi
        bits, st, en = self.seg[str(sp)]; i = self.ix[str(sp)].get(str(nm), -1)
        if i < 0 or en[i] <= st[i]:
            return None
        c = np.unpackbits(bits[rng.randint(st[i], en[i])]).reshape(128, 128).astype(np.float32)
        if rng.rand() < 0.5:                                                    # mild bias towards larger objects
            c2 = np.unpackbits(bits[rng.randint(st[i], en[i])]).reshape(128, 128).astype(np.float32)
            c = c2 if c2.sum() > c.sum() else c
        return ndi.gaussian_filter(c.repeat(4, 0).repeat(4, 1), 1.5) > 0.5      # 4-px cells -> smooth 512-px silhouette

    def __getitem__(self, i):
        rng = np.random.RandomState((self.seed * 2909 + i * 49979687 + 17) % 2 ** 31)
        if rng.rand() >= self.p:
            return self.inner[i]
        t3 = self.t3; pi, k = t3.idx[rng.randint(len(t3.idx))]; Y, sp, nm = t3.pools[pi]
        M = self._seg(rng, sp[k], nm[k])
        if M is None or M.sum() < 64:
            return self.inner[i]
        b = t3._base(sp[k], nm[k]); yk = np.asarray(Y[k])
        if rng.rand() < 0.5:
            y = np.where(M[..., None], yk, b)
        else:
            pj, kj = t3.idx[rng.randint(len(t3.idx))]; y = np.where(M[..., None], yk, t3._base(t3.pools[pj][1][kj], t3.pools[pj][2][kj]))
            if rng.rand() < 0.7:
                M2 = self._seg(rng, sp[k], nm[k])
                if M2 is not None and not (M2 & M).any():
                    y = np.where(M2[..., None], b, y)
        return torch.from_numpy(np.ascontiguousarray(y)).permute(2, 0, 1), torch.from_numpy(M.astype(np.float32))[None]


def tiny_holes(p_tiny):
    """Sun 10:35. Diagnosis: on PixelSeal / TrustMark full images whose holes cover < 3 % of the frame the evidence map
    itself fails (per-image oracle threshold .70-.73, StegaStamp .90).  With probability p_tiny the full-regime item gets
    1-3 holes of total area 0.2 %-3 % (same shape sources as tf.sample_holes), else the usual sampler."""
    base = tf.sample_holes

    def f(rng, lib):
        if rng.rand() >= p_tiny:
            return base(rng, lib)
        H = np.zeros((512, 512), bool); n = int(rng.choice([1, 2, 3], p=[0.6, 0.3, 0.1])); tot = float(np.exp(rng.uniform(np.log(0.002), np.log(0.03))))
        for _ in range(n):
            a = tot / n * rng.uniform(0.6, 1.4) * 512 * 512
            if lib and rng.rand() < 0.6:
                o = lib[rng.randint(len(lib))]; sc = np.sqrt(a / max(o.sum(), 1))
                sh = (max(6, min(512, int(o.shape[0] * sc))), max(6, min(512, int(o.shape[1] * sc))))
                o = np.asarray(torch.nn.functional.interpolate(torch.from_numpy(np.ascontiguousarray(o)[None, None].astype(np.float32)), size=sh, mode="bilinear")[0, 0] > 0.5)
            elif rng.rand() < 0.5:
                ar = np.exp(rng.uniform(-0.8, 0.8)); o = np.ones((max(6, min(512, int(np.sqrt(a * ar)))), max(6, min(512, int(np.sqrt(a / ar))))), bool)
            else:
                ar = np.exp(rng.uniform(-0.6, 0.6)); hh, ww = max(8, int(np.sqrt(a * ar * 4 / np.pi))), max(8, int(np.sqrt(a / ar * 4 / np.pi)))
                yy, xx = np.mgrid[-1:1:hh * 1j, -1:1:ww * 1j]; o = (xx ** 2 + yy ** 2) < 1
            y0 = rng.randint(0, 513 - o.shape[0]); x0 = rng.randint(0, 513 - o.shape[1]); H[y0:y0 + o.shape[0], x0:x0 + o.shape[1]] |= o
        return H
    return f


def thin_pool(method, n, device, seed=0):
    """n rectangle items, official process: rectangle of a clean base (image of another method) embedded as a whole image
    at its own size.  Kept in memory (n x 0.8 MB)."""
    import sys
    sys.path.insert(0, str(ROOT / "kit"))
    from wmloc.watermarks import load_watermark
    from src.synth_full import bases_for
    from src.synth_small2 import sample_wh
    wm = load_watermark(method, models_dir=ROOT / "models", device=device); bases = bases_for(method, ("train", "test"))
    rng = np.random.RandomState(4000 + seed); Y = np.zeros((n, 512, 512, 3), np.uint8); M = np.zeros((n, 512, 512), bool); t0 = time.time()
    for k in range(n):
        sp, nm = bases[rng.randint(len(bases))]; b = image(sp, nm).copy(); bits = rng.randint(0, 2, wm.nbits)
        w, h = sample_wh(rng, method, 0.002, 0.25); x0, y0 = rng.randint(0, 513 - w), rng.randint(0, 513 - h)
        b[y0:y0 + h, x0:x0 + w] = wm.embed(np.ascontiguousarray(b[y0:y0 + h, x0:x0 + w]), bits); Y[k] = b; M[k, y0:y0 + h, x0:x0 + w] = True
    del wm
    if "cuda" in str(device):
        torch.cuda.empty_cache()
    print(f"thin-rectangle pool: {n} items in {time.time() - t0:.0f}s", flush=True)
    return Y, M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base"); ap.add_argument("--exp"); ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=5e-5); ap.add_argument("--bs", type=int, default=12)
    ap.add_argument("--p_rescale", type=float, default=0.85); ap.add_argument("--thin", type=int, default=0)
    ap.add_argument("--device", default="cuda:0"); ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--val_every", type=int, default=500); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_root", default=str(ROOT / "work/runs"))
    ap.add_argument("--p_feather", type=float, default=0.0); ap.add_argument("--feather_max", type=float, default=5.0)
    ap.add_argument("--obj_items", type=float, default=0.0)     # full-regime bases: share of in-place transplanted-object items
    ap.add_argument("--tiny_holes", type=float, default=0.0)    # full bases: share of items with 1-3 holes of total area 0.2-3 %
    ap.add_argument("--sam_items", type=float, default=0.0)     # any base: share of transplanted-object items with a SAM-segment silhouette
    ap.add_argument("--p_down", type=float, default=0.0); ap.add_argument("--p_blur", type=float, default=0.0); ap.add_argument("--blur_max", type=float, default=0.6)
    ap.add_argument("--p_jpeg", type=float, default=0.0); ap.add_argument("--jpeg_min", type=int, default=90)
    a = ap.parse_args(); dev = a.device
    global PF, FMAX; PF, FMAX = a.p_feather, a.feather_max
    b = json.loads((ROOT / "work/runs" / a.base / "args.json").read_text()); method = b["method"]; reg = b.get("regime", "full")
    tf.POOL_TAG = b.get("pool_tag", ""); tf.SHAPE_LIB = b.get("shape_lib", "train"); tf.INPUT = b.get("input", "rgbhp")
    tf.EXCLUDE_FOLD = b.get("exclude_fold", -1); tf.RESCALE_AUG = 0; tf.INIT = "none" if b.get("init", "imagenet") == "dec" else "imagenet"
    torch.manual_seed(a.seed)
    out = Path(a.out_root) / a.exp; out.mkdir(parents=True, exist_ok=True)
    LP.update(p_down=a.p_down, p_blur=a.p_blur, blur_max=a.blur_max, p_jpeg=a.p_jpeg, jpeg_min=a.jpeg_min)
    args = dict(b, exp=a.exp, base=a.base, kind_ft="rescale_ft", lowpass=dict(LP), ft_steps=a.steps, ft_lr=a.lr, p_rescale=a.p_rescale, thin=a.thin,
                p_feather=a.p_feather, feather_max=a.feather_max, obj_items=a.obj_items, sam_items=a.sam_items, tiny_holes=a.tiny_holes,
                ft_seed=a.seed, base_args=b)         # method / regime / exclude_fold / shape_lib inherited => same cell restriction
    (out / "args.json").write_text(json.dumps(args))
    me, rg, G = methods(), regime(), masks128()
    tr_idx = np.where(me == method)[0]; te_idx = np.where(methods("test") == method)[0]
    val_sel = {"full": rg[tr_idx], "small": ~rg[tr_idx], "small2": ~rg[tr_idx], "small3": ~rg[tr_idx], "mix": np.ones(len(tr_idx), bool)}[reg]
    net = tf.Net(b.get("encoder", "resnet34"))
    net.load_state_dict(torch.load(ROOT / "work/runs" / a.base / "model.pt", map_location="cpu"))
    net = net.to(dev).to(memory_format=torch.channels_last)

    Gb = G[tr_idx].astype(bool); sm_ = ~rg[tr_idx]
    isobj = np.array([g.any() and g.sum() / ((np.ptp(np.where(g)[0]) + 1) * (np.ptp(np.where(g)[1]) + 1)) < 0.95 for g in Gb]) & sm_

    def val(step, t0):
        P = tf.predict(net, "train", tr_idx, dev); s, t, _ = tf.val_report(P[val_sel], G[tr_idx][val_sel])
        print(f"VAL step {step} {reg}-regime mIoU {s:.4f} @thr {t:.2f} {time.time() - t0:.0f}s", flush=True)
        if a.obj_items > 0 or a.sam_items > 0:           # second read-out: real SMALL-regime images (objects / rectangles), best threshold on the small images
            thr = np.linspace(0.05, 0.95, 19); sc = np.stack([tf.miou_batch(P[sm_] >= t_, G[tr_idx][sm_]) for t_ in thr]); j = int(np.argmax(sc.mean(1)))
            print(f"VAL2 step {step} small-regime mIoU {sc[j].mean():.4f} @thr {thr[j]:.2f} | objects {sc[j][isobj[sm_]].mean():.4f} (n={int(isobj.sum())}) "
                  f"rectangles {sc[j][~isobj[sm_]].mean():.4f} | objects at their own best thr {sc[:, isobj[sm_]].mean(1).max():.4f}", flush=True)
        return P, s
    t0 = time.time(); _, s0 = val(0, t0)
    if a.tiny_holes > 0:
        assert reg == 'full'; tf.sample_holes = tiny_holes(a.tiny_holes)
    extra = thin_pool(method, a.thin, dev, a.seed) if a.thin else None
    assert reg != "small2", "small2 pools store 128-px masks: use a small / small3 / full base"
    DS = {"full": tf.SynthFull, "small": tf.SynthSmall, "small3": tf.SynthSmall3, "mix": tf.SynthMix}[reg]
    if a.p_feather > 0:
        assert reg in ("full", "small", "small3"), "feather: full / small / small3 bases only"
        DS = {"full": FeatherFull, "small": FeatherSmall, "small3": tf.SynthSmall3}[reg]
    tf.to128 = lambda M: M.astype(np.float32)        # the wrapped datasets now return their 512-px mask (workers fork after this)
    inner = DS(method, n_epoch_items=a.steps * a.bs, seed=1000 + a.seed, amp_start=1.0, amp_frac=0.5)
    if a.p_feather > 0 and reg == "small3":          # soft borders on the rectangle half only (transplants stay hard)
        inner.rect = FeatherSmall(method, a.steps * a.bs, 1000 + a.seed, 1.0, 0.5); inner.rect.lib = []
    if a.obj_items > 0:
        assert reg == "full", "--obj_items is for full-regime bases"
        inner = ObjMix(inner, method, a.steps * a.bs, 1000 + a.seed, a.obj_items)
    if a.sam_items > 0:
        inner = SamObjMix(inner, method, a.steps * a.bs, 1000 + a.seed, a.sam_items)
    ds = Rescaled(inner, a.p_rescale, a.seed, extra)
    dl = DataLoader(ds, batch_size=a.bs, num_workers=a.workers, shuffle=False, drop_last=True, persistent_workers=True, prefetch_factor=4)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps, pct_start=0.1)
    net.train(); run = 0; s = s0
    for step, (x, y) in enumerate(dl, 1):
        x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg = net(x)
        loss = tf.loss_fn(lg.float(), y); opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
        run = 0.98 * run + 0.02 * loss.item() if step > 1 else loss.item()
        if step % 100 == 0:
            print(f"step {step} loss {run:.4f} {time.time() - t0:.0f}s", flush=True)
        if step % a.val_every == 0 or step == a.steps:
            P, s = val(step, t0)
            np.save(out / "train.npy", P.astype(np.float16)); torch.save(net.state_dict(), out / "model.pt")   # LAST checkpoint kept
        if step >= a.steps:
            break
    np.save(out / "test.npy", tf.predict(net, "test", te_idx, dev).astype(np.float16))
    print(f"DONE {a.exp}: {reg}-regime mIoU base {s0:.4f} -> {s:.4f} {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
