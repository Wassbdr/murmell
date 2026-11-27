"""Full-regime hole segmenter trained ONLY on synthetic data (src/synth_full.py pools + on-the-fly holes).
Real train images are never used for training => predictions on all real train images are out-of-sample.
Validation = real FULL-regime train images of the method (85): mIoU at best thr, and vs all-ones.
    python -m src.train_full --method trustmark --exp F1_tm --steps 6000 --device cuda:0
Outputs work/runs/<exp>/{train,test}.npy: float16 maps for the images of <method> (metadata order of that method),
        work/runs/<exp>/model.pt, log lines "VAL step ...".
"""
import argparse
import glob
import json
import time

import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage as ndi
from torch.utils.data import DataLoader, Dataset

from src.data import ROOT, folds, image, mask512, masks128, meta, methods, regime
from src.metric import miou_batch


RESCALE_AUG = 0        # 1: small items get the dataset's post-embedding crop+rescale (zoom s~U[1,1.2], random kernel)
EXCLUDE_FOLD = -1      # >=0: shape libraries exclude ALL train images of this CV fold (model valid for that fold only)
SHAPE_LIB = "train"    # "none" => procedural shapes only (blobs/rects), no train-mask shapes
POOL_TAG = ""          # set from --pool_tag ("" = bases from train+test of other methods, "_tr" = train-only bases)

# ---------------------------------------------------------------- holes
def object_library():
    """object/irregular shapes from small-regime train masks (not used as eval targets of this model)."""
    lib = []
    rg = regime(); pr = np.load(ROOT / "work/oof/B0_regime_vec.npy"); fo = folds()
    for i, m in enumerate(meta("train")):
        if rg[i]:
            continue
        if EXCLUDE_FOLD >= 0:
            if fo[i] == EXCLUDE_FOLD:
                continue
        elif pr[i]:                 # skip small images PREDICTED full: they are scored in F cells by F models
            continue
        mk = mask512(i).astype(bool)
        lab, n = ndi.label(mk)
        for j in range(1, n + 1):
            c = lab == j
            ys, xs = np.where(c)
            if len(ys) < 200:
                continue
            crop = c[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
            if crop.mean() < 0.95:                                   # skip plain rectangles (handled separately)
                lib.append(crop)
    return lib


def hole_library():
    """object shapes = connected components of the clean holes of FULL-regime train masks (used by the SMALL model,
    whose evaluation targets are small-regime masks => no overlap)."""
    lib = []
    rg = regime(); pr = np.load(ROOT / "work/oof/B0_regime_vec.npy"); fo = folds()
    sel = (rg & (fo != EXCLUDE_FOLD)) if EXCLUDE_FOLD >= 0 else (rg & pr)   # else: skip full images PREDICTED small
    for i in np.where(sel)[0]:
        hole = ~mask512(int(i)).astype(bool)
        lab, n = ndi.label(hole)
        for j in range(1, n + 1):
            c = lab == j
            ys, xs = np.where(c)
            if len(ys) < 200:
                continue
            crop = c[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
            if crop.mean() < 0.95:
                lib.append(crop)
    return lib


def sample_holes(rng, lib):
    """bool 512x512, True = hole (restored clean)."""
    r = rng.rand()
    h = rng.uniform(0, 0.05) if r < 0.4 else (rng.uniform(0.05, 0.25) if r < 0.85 else rng.uniform(0.25, 0.5))
    H = np.zeros((512, 512), bool)
    kind = rng.rand()
    if kind < 0.45:                                                  # smooth random blobs
        sig = rng.uniform(10, 60)
        f = ndi.gaussian_filter(rng.randn(128, 128).astype(np.float32), sig / 4)
        f = np.kron(f, np.ones((4, 4), np.float32)); f = ndi.gaussian_filter(f, 2)
        H = f > np.quantile(f, 1 - h)
    else:
        tries = 0
        while H.mean() < h and tries < 8:
            tries += 1
            if kind < 0.8 and lib:
                o = lib[rng.randint(len(lib))]
                sc = rng.uniform(0.5, 2.0) * np.sqrt(max(h, 0.003) * 512 * 512 / max(o.sum(), 1)) / 2
                sh = (max(4, min(512, int(o.shape[0] * sc))), max(4, min(512, int(o.shape[1] * sc))))
                o = np.asarray(torch.nn.functional.interpolate(torch.from_numpy(o[None, None].astype(np.float32)), size=sh,
                                                               mode="bilinear")[0, 0] > 0.5)
                if rng.rand() < 0.5:
                    o = o[:, ::-1]
            else:
                a = h * 512 * 512 * rng.uniform(0.3, 1.0); ar = np.exp(rng.uniform(-1, 1))
                o = np.ones((max(4, min(512, int(np.sqrt(a * ar)))), max(4, min(512, int(np.sqrt(a / ar))))), bool)
            y0 = rng.randint(-o.shape[0] // 3, 512 - o.shape[0] * 2 // 3); x0 = rng.randint(-o.shape[1] // 3, 512 - o.shape[1] * 2 // 3)
            ys, xs = max(0, y0), max(0, x0); ye, xe = min(512, y0 + o.shape[0]), min(512, x0 + o.shape[1])
            H[ys:ye, xs:xe] |= o[ys - y0:ye - y0, xs - x0:xe - x0]
    return H


def to128(wm512):
    return (wm512.reshape(128, 4, 128, 4).mean((1, 3)) >= 0.5).astype(np.float32)


class SynthFull(Dataset):
    def __init__(self, method, n_epoch_items=10 ** 7, seed=0, amp_start=1.0, amp_frac=0.5):
        self.pools = []
        for p in sorted(glob.glob(str(ROOT / "work" / f"synth_full{POOL_TAG}" / f"{method}_s*.npy"))):
            Y = np.load(p, mmap_mode="r"); z = np.load(p[:-4] + ".npz")
            self.pools.append((p, Y, z["split"], z["name"]))
        self.idx = [(pi, k) for pi, (_, Y, _, _) in enumerate(self.pools) for k in range(len(Y))]
        self.amp_start, self.amp_frac = amp_start, amp_frac   # curriculum: watermark x amp, annealed to 1
        self.n = n_epoch_items; self.seed = seed; self.lib = object_library() if SHAPE_LIB == "train" else []; self.bcache = {}
        print("pool items", len(self.idx), "object shapes", len(self.lib), flush=True)

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        rng = np.random.RandomState((self.seed * 7919 + i * 104729) % 2 ** 31)
        pi, k = self.idx[rng.randint(len(self.idx))]
        _, Y, sp, nm = self.pools[pi]
        key = (str(sp[k]), str(nm[k]))
        if key not in self.bcache:
            if len(self.bcache) > 400:
                self.bcache.clear()
            self.bcache[key] = image(*key)
        b = self.bcache[key]; yk = np.asarray(Y[k])
        H = sample_holes(rng, self.lib)
        prog = i / max(1, self.n * self.amp_frac)
        amp = 1.0 if prog >= 1 else self.amp_start + (1.0 - self.amp_start) * prog
        if amp != 1.0:
            yk = np.clip(b.astype(np.float32) + amp * (yk.astype(np.float32) - b), 0, 255).round().astype(np.uint8)
        y = np.where(H[..., None], b, yk)
        return torch.from_numpy(y).permute(2, 0, 1), torch.from_numpy(to128(~H))[None]


class SynthSmall(Dataset):
    """y = where(shape inside rect R, Yk, base); shape = full rect (50 %) or a stretched object shape (50 %)."""

    def __init__(self, method, n_epoch_items=10 ** 7, seed=0, amp_start=1.0, amp_frac=0.5):
        self.pools = []
        for p in sorted(glob.glob(str(ROOT / "work" / f"synth_small{POOL_TAG}" / f"{method}_s*.npy"))):
            Y = np.load(p, mmap_mode="r"); z = np.load(p[:-4] + ".npz")
            self.pools.append((Y, z["split"], z["name"], z["rect"]))
        self.idx = [(pi, k) for pi, (Y, _, _, _) in enumerate(self.pools) for k in range(len(Y))]
        self.n, self.seed, self.amp_start, self.amp_frac = n_epoch_items, seed, amp_start, amp_frac
        self.lib = hole_library() if SHAPE_LIB == "train" else []; self.bcache = {}
        print("small pool items", len(self.idx), "object shapes", len(self.lib), flush=True)

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        rng = np.random.RandomState((self.seed * 7919 + i * 104729 + 17) % 2 ** 31)
        pi, k = self.idx[rng.randint(len(self.idx))]
        Y, sp, nm, rect = self.pools[pi]
        key = (str(sp[k]), str(nm[k]))
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
            o = np.asarray(F.interpolate(torch.from_numpy(o[None, None].astype(np.float32)).contiguous(), size=(h, w),
                                         mode="bilinear")[0, 0] > 0.5)
            M[y0:y0 + h, x0:x0 + w] = o
            if M.sum() < 16:
                M[y0:y0 + h, x0:x0 + w] = True
        prog = i / max(1, self.n * self.amp_frac)
        amp = 1.0 if prog >= 1 else self.amp_start + (1.0 - self.amp_start) * prog
        if amp != 1.0:
            yk = np.clip(b.astype(np.float32) + amp * (yk.astype(np.float32) - b), 0, 255).round().astype(np.uint8)
        y = np.where(M[..., None], yk, b)
        if RESCALE_AUG:
            from PIL import Image as _I
            sc = rng.uniform(1.0, 1.2); c = int(round(512 / sc)); oy, ox = rng.randint(0, 512 - c + 1), rng.randint(0, 512 - c + 1)
            ker = [_I.Resampling.BILINEAR, _I.Resampling.BICUBIC, _I.Resampling.LANCZOS][rng.randint(3)]
            y = np.asarray(_I.fromarray(np.ascontiguousarray(y[oy:oy + c, ox:ox + c])).resize((512, 512), ker))
            M = np.asarray(_I.fromarray((M[oy:oy + c, ox:ox + c] * 255).astype(np.uint8)).resize((512, 512), _I.Resampling.BILINEAR)) > 127
        return torch.from_numpy(y).permute(2, 0, 1), torch.from_numpy(to128(M))[None]


class SynthSmall2(Dataset):
    """pools of src/synth_small2.py (official process: embed at own resolution + resize, aspect up to 16:1, transplanted
    objects at random scale); mask stored per item; amp curriculum relative to the stored clean base."""

    def __init__(self, method, n_epoch_items=10 ** 7, seed=0, amp_start=1.0, amp_frac=0.5):
        self.pools = []
        for p in sorted(glob.glob(str(ROOT / "work" / f"synth_small2{POOL_TAG}" / f"{method}_s*[0-9].npy"))):
            Y = np.load(p, mmap_mode="r"); Mk = np.load(p[:-4] + "_m.npy", mmap_mode="r"); z = np.load(p[:-4] + ".npz")
            self.pools.append((Y, Mk, z["split"], z["name"]))
        self.idx = [(pi, k) for pi, (Y, _, _, _) in enumerate(self.pools) for k in range(len(Y))]
        self.n, self.seed, self.amp_start, self.amp_frac = n_epoch_items, seed, amp_start, amp_frac; self.bcache = {}
        print("small2 pool items", len(self.idx), flush=True)

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        rng = np.random.RandomState((self.seed * 7919 + i * 104729 + 29) % 2 ** 31)
        pi, k = self.idx[rng.randint(len(self.idx))]
        Y, Mk, sp, nm = self.pools[pi]
        y = np.asarray(Y[k]); m = np.asarray(Mk[k]).astype(np.float32)
        prog = i / max(1, self.n * self.amp_frac)
        amp = 1.0 if prog >= 1 else self.amp_start + (1.0 - self.amp_start) * prog
        if amp != 1.0:
            key = (str(sp[k]), str(nm[k]))
            if key not in self.bcache:
                if len(self.bcache) > 400:
                    self.bcache.clear()
                self.bcache[key] = image(*key)
            b = self.bcache[key].astype(np.float32)
            y = np.clip(b + amp * (y.astype(np.float32) - b), 0, 255).round().astype(np.uint8)
        return torch.from_numpy(np.ascontiguousarray(y)).permute(2, 0, 1), torch.from_numpy(m)[None]


T_PRECT = 0.5         # small3: share of plain rectangle items (0 => transplant-only "object specialist")
T_PSCALE = 0.7        # small3: among PASTED transplants, share pasted at a random scale (0 => always native scale)


class SynthSmall3(Dataset):
    """'transplant' small items (official doc: watermarked object transplanted into a clean image, mask = the object):
    an object-shaped FRAGMENT of a WHOLE-IMAGE watermark (synth_full pools) on a clean image, mixed 50/50 with plain
    rectangle items of SynthSmall (outpainting: the rectangle carries a whole watermark at its own size).
    Transplant item: half 'in place' (y = where(M, Y_full, base)), half 'pasted' on ANOTHER clean image at a random scale
    g in [0.6, 1.6], with a CLEAN distractor object pasted from the clean source (label 0) so that paste seams are not
    the cue.  Shapes = hole_library() (components of full-regime holes; small-regime GT shapes are never used)."""

    def __init__(self, method, n_epoch_items=10 ** 7, seed=0, amp_start=1.0, amp_frac=0.5):
        self.rect = SynthSmall(method, n_epoch_items, seed, amp_start, amp_frac)
        self.lib = self.rect.lib; self.rect.lib = []                      # rect items only from SynthSmall
        self.pools = []
        for p in sorted(glob.glob(str(ROOT / "work" / f"synth_full{POOL_TAG}" / f"{method}_s*.npy"))):
            Y = np.load(p, mmap_mode="r"); z = np.load(p[:-4] + ".npz"); self.pools.append((Y, z["split"], z["name"]))
        self.idx = [(pi, k) for pi, (Y, _, _) in enumerate(self.pools) for k in range(len(Y))]
        self.n, self.seed, self.amp_start, self.amp_frac = n_epoch_items, seed, amp_start, amp_frac; self.bcache = {}
        print("small3: full-pool items", len(self.idx), "shapes", len(self.lib), flush=True)

    def __len__(self):
        return self.n

    def _base(self, sp, nm):
        key = (str(sp), str(nm))
        if key not in self.bcache:
            if len(self.bcache) > 300:
                self.bcache.clear()
            self.bcache[key] = image(*key)
        return self.bcache[key]

    def _shape(self, rng, amin, amax):
        from PIL import Image as _I
        if self.lib and rng.rand() < 0.85:
            o = self.lib[rng.randint(len(self.lib))]
            if rng.rand() < 0.5:
                o = o[:, ::-1]
        else:                                                           # procedural blob (ellipse)
            yy, xx = np.mgrid[-1:1:64j, -1:1:64j]; o = (xx ** 2 + yy ** 2) < 1
        a = np.exp(rng.uniform(np.log(amin), np.log(amax))) * 512 * 512 / max(0.2, o.mean())
        ar = (o.shape[1] / o.shape[0]) * np.exp(rng.uniform(-0.35, 0.35))
        w, h = int(np.clip(round(np.sqrt(a * ar)), 12, 512)), int(np.clip(round(np.sqrt(a / ar)), 12, 512))
        om = np.asarray(_I.fromarray((np.ascontiguousarray(o) * 255).astype(np.uint8)).resize((w, h), _I.Resampling.BILINEAR)) > 127
        if om.sum() < 16:
            om[:] = True
        return om, rng.randint(0, 513 - h), rng.randint(0, 513 - w)

    def _cut(self, rng, srcimg, h, w, g):
        from PIL import Image as _I
        sh, sw = int(np.clip(round(h / g), 8, 512)), int(np.clip(round(w / g), 8, 512))
        sy, sx = rng.randint(0, 513 - sh), rng.randint(0, 513 - sw)
        c = np.ascontiguousarray(srcimg[sy:sy + sh, sx:sx + sw])
        if (sh, sw) != (h, w):
            ker = [_I.Resampling.BILINEAR, _I.Resampling.BICUBIC, _I.Resampling.LANCZOS][rng.randint(3)]
            c = np.asarray(_I.fromarray(c).resize((w, h), ker))
        return c

    def __getitem__(self, i):
        rng = np.random.RandomState((self.seed * 7919 + i * 104729 + 41) % 2 ** 31)
        if rng.rand() < T_PRECT:
            return self.rect[i]
        pi, k = self.idx[rng.randint(len(self.idx))]; Y, sp, nm = self.pools[pi]
        b = self._base(sp[k], nm[k]); yk = np.asarray(Y[k])
        prog = i / max(1, self.n * self.amp_frac)
        amp = 1.0 if prog >= 1 else self.amp_start + (1.0 - self.amp_start) * prog
        if amp != 1.0:
            yk = np.clip(b.astype(np.float32) + amp * (yk.astype(np.float32) - b), 0, 255).round().astype(np.uint8)
        om, y0, x0 = self._shape(rng, 0.003, 0.35); h, w = om.shape
        M = np.zeros((512, 512), bool); M[y0:y0 + h, x0:x0 + w] = om
        if rng.rand() < 0.5:                                            # in place: same image, watermark only inside M
            y = np.where(M[..., None], yk, b)
        else:                                                           # pasted on another clean image, random scale
            pj, kj = self.idx[rng.randint(len(self.idx))]; y = self._base(self.pools[pj][1][kj], self.pools[pj][2][kj]).copy()
            g = float(np.exp(rng.uniform(np.log(0.6), np.log(1.6)))) if rng.rand() < T_PSCALE else 1.0
            reg = y[y0:y0 + h, x0:x0 + w]; reg[om] = self._cut(rng, yk, h, w, g)[om]
            if rng.rand() < 0.7:                                        # clean distractor from the clean source (label 0)
                om2, y2, x2 = self._shape(rng, 0.003, 0.2); h2, w2 = om2.shape
                if not M[y2:y2 + h2, x2:x2 + w2].any():
                    g2 = float(np.exp(rng.uniform(np.log(0.6), np.log(1.6))))
                    reg2 = y[y2:y2 + h2, x2:x2 + w2]; reg2[om2] = self._cut(rng, b, h2, w2, g2)[om2]
        return torch.from_numpy(np.ascontiguousarray(y)).permute(2, 0, 1), torch.from_numpy(to128(M))[None]


class SynthMix(Dataset):
    """50/50 full-type and small-type synthetic items (one general local watermark detector)."""

    def __init__(self, method, n_epoch_items=10 ** 7, seed=0, amp_start=1.0, amp_frac=0.5, p_full=0.5):
        self.f = SynthFull(method, n_epoch_items, seed, amp_start, amp_frac)
        self.s = SynthSmall(method, n_epoch_items, seed, amp_start, amp_frac)
        self.n, self.p, self.seed = n_epoch_items, p_full, seed

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        r = np.random.RandomState((self.seed * 31 + i * 7349 + 3) % 2 ** 31).rand()
        return self.f[i] if r < self.p else self.s[i]


# ---------------------------------------------------------------- model
INPUT = "rgbhp"        # "lumachroma": PixelSeal changes R, G, B equally => chroma differences carry content only


def make_input(x_u8):
    x = x_u8.float() / 255
    if INPUT == "lumachroma":
        r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
        z = torch.cat([0.299 * r + 0.587 * g + 0.114 * b, r - g, b - g], 1)     # luma + 2 chroma differences
    else:
        z = x
    k = torch.ones(3, 1, 5, 5, device=x.device) / 25
    hp = (z - F.conv2d(F.pad(z, (2, 2, 2, 2), mode="reflect"), k, groups=3)) * 8
    return torch.cat([x, hp], 1)


INIT = "imagenet"      # "dec": encoder weights = the official watermark DECODER (TrustMark ResNet-50 / PixelSeal ConvNeXtV2)


def build(encoder, nostride, weights="imagenet", **unet_kw):
    if INPUT in ("dec", "dec512"):     # decoder-style input, 3 channels in [-1, 1]; "dec" = half resolution, "dec512" = full
        return smp.Unet(encoder, encoder_weights="imagenet" if INIT == "imagenet" else None, in_channels=3, classes=1)
    m = smp.Unet(encoder, encoder_weights=weights, in_channels=6, classes=1, **unet_kw)
    if nostride:
        raise NotImplementedError("stem-stride removal breaks smp skip sizes; TODO custom decoder")
    return m


class Net(nn.Module):
    """pool: output cell size in pixels (4 = 512 px image -> 128 x 128 cells, as in the competition)."""

    def __init__(self, encoder="resnet34", nostride=False, pool=4, weights="imagenet", **unet_kw):
        super().__init__(); self.m = build(encoder, nostride, weights, **unet_kw); self.nostride = nostride; self.pool = pool

    def forward(self, x_u8):
        if INPUT == "dec512":  # full resolution, x*2-1; logits 512 -> 128 cells
            return F.avg_pool2d(self.m(x_u8.float() / 255 * 2 - 1), 4)
        if INPUT == "dec":     # 512 -> 256 (antialiased), x*2-1 as both decoders preprocess; logits 256 -> 128 cells
            x = F.interpolate(x_u8.float() / 255, scale_factor=0.5, mode="bilinear", align_corners=False, antialias=True) * 2 - 1
            return F.avg_pool2d(self.m(x), 2)
        x = make_input(x_u8)
        if self.nostride:      # encoder sees full res; smp's decoder expects /32 divisibility, output is 4x larger
            out = self.m(x)
            return F.avg_pool2d(out, 16)        # 2048 -> 128
        return F.avg_pool2d(self.m(x), self.pool)   # 512 -> 128 logits


def load_decoder_init(net, method):
    """encoder <- official decoder weights (the decoder already 'sees' the watermark; ImageNet features do not)."""
    import re
    import sys
    sys.path.insert(0, str(ROOT / "kit"))
    from wmloc.watermarks import load_watermark
    wm = load_watermark(method, models_dir=ROOT / "models", device="cpu")
    if method == "trustmark":
        net.m.encoder.load_state_dict(wm.net.state_dict())                     # ResNet-50, same key names (fc dropped by smp)
    elif method == "pixelseal":                                                # FB ConvNeXtV2 -> timm convnextv2 (verified 1e-5)
        tgt = net.m.encoder.state_dict(); new = {}
        for k, v in wm.det.convnext.state_dict().items():
            m = re.match(r"downsample_layers\.0\.(\d)\.(.+)", k)
            if m:
                nk = f"model.stem_{m.group(1)}.{m.group(2)}"
            else:
                m = re.match(r"downsample_layers\.(\d)\.(\d)\.(.+)", k)
                if m:
                    nk = f"model.stages_{m.group(1)}.downsample.{m.group(2)}.{m.group(3)}"
                else:
                    i, j, r = re.match(r"stages\.(\d)\.(\d+)\.(.+)", k).groups()
                    r = (r.replace("dwconv", "conv_dw").replace("pwconv1", "mlp.fc1").replace("pwconv2", "mlp.fc2")
                         .replace("grn.gamma", "mlp.grn.weight").replace("grn.beta", "mlp.grn.bias"))
                    nk = f"model.stages_{i}.blocks.{j}.{r}"
            new[nk] = v.reshape(tgt[nk].shape)
        assert set(new) == set(tgt), "convnextv2 key mapping incomplete"
        net.m.encoder.load_state_dict(new)
    else:
        raise ValueError("no decoder init for " + method)
    print("encoder initialised from the", method, "decoder", flush=True)


def loss_fn(lg, y):
    bce = F.binary_cross_entropy_with_logits(lg, y)
    p = torch.sigmoid(lg)
    i1 = (p * y).sum((1, 2, 3)); u1 = (p + y - p * y).sum((1, 2, 3))
    i0 = ((1 - p) * (1 - y)).sum((1, 2, 3)); u0 = ((1 - p) + (1 - y) - (1 - p) * (1 - y)).sum((1, 2, 3))
    return bce + (1 - 0.5 * (i1 / (u1 + 1) + i0 / (u0 + 1))).mean()


@torch.no_grad()
def predict(net, split, idx, dev, bs=16):
    net.eval(); out = []
    for s in range(0, len(idx), bs):
        x = torch.from_numpy(np.stack([image(split, int(i)) for i in idx[s:s + bs]])).permute(0, 3, 1, 2).to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out.append(torch.sigmoid(net(x).float())[:, 0].cpu().numpy())
    net.train()
    return np.concatenate(out)


def val_report(P, G):
    thr = np.linspace(0.05, 0.95, 19)
    sc = [miou_batch(P >= t, G).mean() for t in thr]
    b = int(np.argmax(sc))
    return sc[b], thr[b], miou_batch(np.ones_like(G), G).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method"); ap.add_argument("--exp"); ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--bs", type=int, default=12); ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--encoder", default="resnet34"); ap.add_argument("--nostride", type=int, default=0)
    ap.add_argument("--device", default="cuda:0"); ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--val_every", type=int, default=500); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--regime", default="full", choices=["full", "small", "mix", "small2", "small3"])
    ap.add_argument("--input", default="rgbhp", choices=["rgbhp", "lumachroma", "dec", "dec512"])
    ap.add_argument("--init", default="imagenet", choices=["imagenet", "dec"])
    ap.add_argument("--t_prect", type=float, default=0.5); ap.add_argument("--t_pscale", type=float, default=0.7)
    ap.add_argument("--thin", type=int, default=0)      # with --rescale_aug 2: N extra thin-rectangle items (official process), 50 % of the stream
    ap.add_argument("--pool_tag", default=""); ap.add_argument("--shape_lib", default="train", choices=["train", "none"])
    ap.add_argument("--exclude_fold", type=int, default=-1); ap.add_argument("--rescale_aug", type=int, default=0)
    ap.add_argument("--amp_start", type=float, default=1.0); ap.add_argument("--amp_frac", type=float, default=0.5)
    a = ap.parse_args(); dev = a.device
    global POOL_TAG, SHAPE_LIB, INPUT, EXCLUDE_FOLD; POOL_TAG = a.pool_tag; SHAPE_LIB = a.shape_lib; INPUT = a.input
    EXCLUDE_FOLD = a.exclude_fold; global RESCALE_AUG; RESCALE_AUG = a.rescale_aug if a.rescale_aug == 1 else 0
    global INIT; INIT = a.init
    global T_PRECT, T_PSCALE; T_PRECT, T_PSCALE = a.t_prect, a.t_pscale
    torch.manual_seed(a.seed)
    out = ROOT / "work" / "runs" / a.exp; out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(a)))
    me, rg = methods(), regime(); G = masks128()
    tr_idx = np.where(me == a.method)[0]; te_idx = np.where(methods("test") == a.method)[0]
    full_sel = {"full": rg[tr_idx], "small": ~rg[tr_idx], "small2": ~rg[tr_idx], "small3": ~rg[tr_idx], "mix": np.ones(len(tr_idx), bool)}[a.regime]   # validation images
    DS = {"full": SynthFull, "small": SynthSmall, "small2": SynthSmall2, "small3": SynthSmall3, "mix": SynthMix}[a.regime]
    ds = DS(a.method, n_epoch_items=a.steps * a.bs, seed=a.seed, amp_start=a.amp_start, amp_frac=a.amp_frac)
    if a.rescale_aug == 3:      # audit for TrustMark full: soft (feathered) hole borders, sigma U[0.5, 4] px on 80 % of the items
        import src.rescale_ft as rf                            # (imported BEFORE to128 is patched)
        assert a.regime == "full", "--rescale_aug 3 (feather) is for the full regime"
        rf.PF, rf.FMAX = 0.8, 4.0
        ds = rf.Rescaled(rf.FeatherFull(a.method, n_epoch_items=a.steps * a.bs, seed=a.seed, amp_start=a.amp_start, amp_frac=a.amp_frac), 0.0, a.seed, None)
    if a.rescale_aug == 2:      # finding: the real images went through a global crop + resize; train with it from the start
        from src.rescale_ft import Rescaled, thin_pool         # (imported BEFORE to128 is patched: it keeps the real one)
        extra = thin_pool(a.method, a.thin, dev, a.seed) if a.thin else None
        global to128; to128 = lambda M: M.astype(np.float32)   # wrapped datasets now return their 512-px mask
        ds = Rescaled(ds, 0.85, a.seed, extra)
    dl = DataLoader(ds, batch_size=a.bs, num_workers=a.workers, shuffle=False, drop_last=True, persistent_workers=True,
                    prefetch_factor=4)
    net = Net(a.encoder, bool(a.nostride))
    if a.init == "dec":
        load_decoder_init(net, a.method)
    net = net.to(dev).to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps, pct_start=0.05)
    t0 = time.time(); best = -1; run = 0
    for step, (x, y) in enumerate(dl, 1):
        x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg = net(x)
        loss = loss_fn(lg.float(), y)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
        run = 0.98 * run + 0.02 * loss.item() if step > 1 else loss.item()
        if step % 100 == 0:
            print(f"step {step} loss {run:.4f} {time.time() - t0:.0f}s", flush=True)
        if step % a.val_every == 0 or step == a.steps:
            P = predict(net, "train", tr_idx, dev)
            s_full, t_full, ones = val_report(P[full_sel], G[tr_idx][full_sel])
            zeros = miou_batch(np.zeros_like(G[tr_idx][full_sel]), G[tr_idx][full_sel]).mean()
            print(f"VAL step {step} {a.regime}-regime mIoU {s_full:.4f} @thr {t_full:.2f} (all-ones {ones:.4f} all-zeros {zeros:.4f}) {time.time() - t0:.0f}s", flush=True)
            best = max(best, s_full)
            np.save(out / "train.npy", P.astype(np.float16))          # LAST checkpoint is kept (no selection on real data)
            torch.save(net.state_dict(), out / "model.pt")
        if step >= a.steps:
            break
    net.load_state_dict(torch.load(out / "model.pt", map_location=dev))
    np.save(out / "test.npy", predict(net, "test", te_idx, dev).astype(np.float16))
    print(f"DONE last full-regime mIoU {s_full:.4f} (max along training {best:.4f}) {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
