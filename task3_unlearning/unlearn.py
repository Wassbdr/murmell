"""Unlearning pipeline (task 31-unlearning).

Idea: for the gold model a forget image is an image it never saw. For each forget image we build a target
"what gold would predict":
  label        = argmax of the original model on the clean image
  difficulty   = loss of the original model (w.r.t. that label) under strong alterations
  wrong classes= original logits averaged over altered views
The per-image target loss is obtained by rank-mapping the difficulty onto a distribution with mean --gold_mean
and std --gold_std (guesses, to be calibrated on the leaderboard). The model is distilled towards these targets
on exact and altered images, and anchored on the original model with image mosaics to limit drift.

Modes
  diag    statistics of the original model on the forget set (no training)
  train   build targets, unlearn, check, save
  interp  orig + alpha * (unlearned - orig) for several alphas
  check   validate a submission file and print the proxy metrics

WARNING: every "PROXY" number is measured against OUR OWN targets, not against the real gold models.
"""
import argparse
import math
import os
import time
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn.functional as F

NUM_CLASSES = 100
DEV = "cuda" if torch.cuda.is_available() else "cpu"


# ----------------------------------------------------------------------------- model / data
def build_model():
    return timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=NUM_CLASSES, init_values=1e-4)


def load_model(path):
    m = build_model()
    m.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
    return m.to(DEV).eval()


def load_images(path):
    with np.load(path, allow_pickle=False) as d:
        x = torch.from_numpy(d["images"].copy())
    assert x.dtype == torch.uint8 and x.shape[1:] == (224, 224, 3), x.shape
    return x.to(DEV)  # uint8 NHWC on GPU (~0.5 GB)


def to01(u8):
    """uint8 [B,224,224,3] -> float [B,3,224,224] in [0,1]."""
    return u8.permute(0, 3, 1, 2).float() / 255.0


def norm(x01):
    return (x01 - 0.5) / 0.5


@torch.no_grad()
def predict01(model, x01, bs=250, fp16=True):
    """Server-like forward (eval, fp16 autocast) on float [0,1] images."""
    model.eval()
    out = []
    for i in range(0, len(x01), bs):
        with torch.autocast("cuda", dtype=torch.float16, enabled=fp16 and DEV == "cuda"):
            out.append(model(norm(x01[i:i + bs])).float())
    return torch.cat(out)


@torch.no_grad()
def predict_u8(model, u8, bs=250, fp16=True, fn=None):
    out = []
    for i in range(0, len(u8), bs):
        x = to01(u8[i:i + bs])
        if fn is not None:
            x = fn(x)
        out.append(predict01(model, x, bs=bs, fp16=fp16))
    return torch.cat(out)


# ----------------------------------------------------------------------------- alterations
def _rand(g, *shape):
    return torch.rand(*shape, generator=g, device=DEV)


def _gauss_blur(x, sigma):
    k = 9
    ax = torch.arange(k, device=x.device, dtype=x.dtype) - k // 2
    w = torch.exp(-ax ** 2 / (2 * sigma ** 2))
    w = w / w.sum()
    C = x.shape[1]
    x = F.conv2d(F.pad(x, (k // 2, k // 2, 0, 0), mode="reflect"), w.view(1, 1, 1, k).repeat(C, 1, 1, 1), groups=C)
    x = F.conv2d(F.pad(x, (0, 0, k // 2, k // 2), mode="reflect"), w.view(1, 1, k, 1).repeat(C, 1, 1, 1), groups=C)
    return x


def _gray(x):
    return (0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3])


def affine(x, scale, angle_deg, flip, tx, ty, ar=None):
    """All args are tensors [B]. scale<1 zooms in."""
    B = x.shape[0]
    ar = torch.ones(B, device=x.device) if ar is None else ar
    sx = (scale * ar.sqrt()).clamp(max=1.0)
    sy = (scale / ar.sqrt()).clamp(max=1.0)
    a = angle_deg * math.pi / 180
    cos, sin = torch.cos(a), torch.sin(a)
    theta = torch.stack([
        torch.stack([sx * cos * flip, -sy * sin, tx * (1 - sx)], 1),
        torch.stack([sx * sin * flip, sy * cos, ty * (1 - sy)], 1)], 1)
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)


def augment(x, st, g, flip=True):
    """Random alteration of float [0,1] images. st in [0,1] (scalar or tensor [B]) is the strength."""
    B = x.shape[0]
    st = torch.as_tensor(st, device=x.device, dtype=x.dtype).expand(B)
    s4 = st.view(B, 1, 1, 1)
    # geometry
    scale = 1 - 0.45 * st * _rand(g, B)
    ar = torch.exp((_rand(g, B) * 2 - 1) * 0.25 * st)
    ang = (_rand(g, B) * 2 - 1) * 20 * st
    fl = torch.where(_rand(g, B) < 0.5, -1.0, 1.0) if flip else torch.ones(B, device=x.device)
    x = affine(x, scale, ang, fl, _rand(g, B) * 2 - 1, _rand(g, B) * 2 - 1, ar)
    # colour
    x = x * (1 + (_rand(g, B, 1, 1, 1) * 2 - 1) * 0.4 * s4)
    m = x.mean((1, 2, 3), keepdim=True)
    x = (x - m) * (1 + (_rand(g, B, 1, 1, 1) * 2 - 1) * 0.4 * s4) + m
    gr = _gray(x)
    x = (x - gr) * (1 + (_rand(g, B, 1, 1, 1) * 2 - 1) * 0.5 * s4) + gr
    x = torch.where(_rand(g, B, 1, 1, 1) < 0.1 * s4, _gray(x).expand_as(x), x)
    # noise
    sig = _rand(g, B, 1, 1, 1) * 0.08 * s4 * (_rand(g, B, 1, 1, 1) < 0.5)
    x = x + torch.randn(x.shape, generator=g, device=x.device) * sig
    # blur (one sigma per batch, applied to a subset)
    bm = _rand(g, B, 1, 1, 1) < 0.3 * s4
    if bm.any():
        x = torch.where(bm, _gauss_blur(x, 0.5 + 1.5 * float(_rand(g, 1))), x)
    # cutout
    cm = _rand(g, B) < 0.3 * st
    if cm.any():
        H = x.shape[-1]
        hh = (0.1 + 0.25 * _rand(g, B)) * H
        ww = (0.1 + 0.25 * _rand(g, B)) * H
        cy, cx = _rand(g, B) * H, _rand(g, B) * H
        yy = torch.arange(H, device=x.device).view(1, H, 1)
        xx = torch.arange(H, device=x.device).view(1, 1, H)
        box = ((yy - cy.view(B, 1, 1)).abs() < hh.view(B, 1, 1) / 2) & ((xx - cx.view(B, 1, 1)).abs() < ww.view(B, 1, 1) / 2)
        x = torch.where((box & cm.view(B, 1, 1)).unsqueeze(1), torch.full_like(x, 0.5), x)
    return x.clamp(0, 1)


def mosaic(x, n, g):
    """n x n grid of downscaled random images from the batch pool x (float [0,1]). Returns [B,3,224,224]."""
    B = x.shape[0]
    t = 224 // n
    tiles = []
    for _ in range(n * n):
        perm = torch.randperm(B, generator=g, device=x.device)
        tiles.append(F.interpolate(x[perm], size=(t, t), mode="bilinear", antialias=True, align_corners=False))
    rows = [torch.cat(tiles[r * n:(r + 1) * n], 3) for r in range(n)]
    out = torch.cat(rows, 2)
    if out.shape[-1] != 224:
        out = F.interpolate(out, size=(224, 224), mode="bilinear", align_corners=False)
    return out


def make_anchor(x, g, kinds):
    """Anchor ("retain proxy") batch built from forget images."""
    B = x.shape[0]
    k = kinds[int(torch.randint(len(kinds), (1,), generator=g, device=DEV))]
    if k == "crop":  # close-up crops: real-image statistics, outside the memorised regime
        sc = 0.3 + 0.25 * _rand(g, B)
        fl = torch.where(_rand(g, B) < 0.5, -1.0, 1.0)
        xc = affine(x, sc, (_rand(g, B) * 2 - 1) * 10, fl, _rand(g, B) * 2 - 1, _rand(g, B) * 2 - 1)
        return (xc * (1 + (_rand(g, B, 1, 1, 1) * 2 - 1) * 0.15)).clamp(0, 1)
    xa = augment(x, 0.4, g)
    if k == "mos2":
        return mosaic(xa, 2, g)
    if k == "mos3":
        return mosaic(xa, 3, g)
    if k == "mix":  # mixup of two different images
        perm = torch.randperm(B, generator=g, device=DEV)
        lam = 0.35 + 0.3 * _rand(g, B, 1, 1, 1)
        return lam * xa + (1 - lam) * xa[perm]
    if k == "cutmix":  # left/right or top/bottom halves of two images
        perm = torch.randperm(B, generator=g, device=DEV)
        out = xa.clone()
        if float(_rand(g, 1)) < 0.5:
            out[:, :, :, 112:] = xa[perm][:, :, :, 112:]
        else:
            out[:, :, 112:, :] = xa[perm][:, :, 112:, :]
        return out
    raise ValueError(k)


NAMED = {
    "hflip": lambda x: x.flip(3),
    "zoom0.85": lambda x: affine(x, *[torch.full((len(x),), v, device=x.device) for v in (0.85, 0.0, 1.0, 0.0, 0.0)]),
    "zoom0.6": lambda x: affine(x, *[torch.full((len(x),), v, device=x.device) for v in (0.6, 0.0, 1.0, 0.0, 0.0)]),
    "shift": lambda x: affine(x, *[torch.full((len(x),), v, device=x.device) for v in (0.85, 0.0, 1.0, 1.0, 1.0)]),
    "rot10": lambda x: affine(x, *[torch.full((len(x),), v, device=x.device) for v in (1.0, 10.0, 1.0, 0.0, 0.0)]),
    "rot25": lambda x: affine(x, *[torch.full((len(x),), v, device=x.device) for v in (1.0, 25.0, 1.0, 0.0, 0.0)]),
    "noise0.02": lambda x: (x + 0.02 * torch.randn_like(x)).clamp(0, 1),
    "noise0.08": lambda x: (x + 0.08 * torch.randn_like(x)).clamp(0, 1),
    "blur1": lambda x: _gauss_blur(x, 1.0),
    "blur2": lambda x: _gauss_blur(x, 2.0),
    "bright+30%": lambda x: (x * 1.3).clamp(0, 1),
    "dark-30%": lambda x: x * 0.7,
    "contrast-40%": lambda x: (x - x.mean((1, 2, 3), keepdim=True)) * 0.6 + x.mean((1, 2, 3), keepdim=True),
    "gray": lambda x: _gray(x).expand_as(x),
    "down112up": lambda x: F.interpolate(F.interpolate(x, size=112, mode="bilinear", antialias=True), size=224, mode="bilinear"),
    "1px-shift": lambda x: torch.roll(x, 1, 3),
}


# ----------------------------------------------------------------------------- metrics helpers
def ce(logits, y):
    return F.cross_entropy(logits.float(), y, reduction="none")


def kl_T(p_logits, q_logits, T=2.0):
    """per-sample KL(softmax(p/T) || softmax(q/T))."""
    lp = F.log_softmax(p_logits.float() / T, 1)
    lq = F.log_softmax(q_logits.float() / T, 1)
    return (lp.exp() * (lp - lq)).sum(1)


def q(t, qs=(0.05, 0.25, 0.5, 0.75, 0.95)):
    return " ".join(f"{float(v):.3f}" for v in torch.quantile(t.float().cpu(), torch.tensor(qs)))


# ----------------------------------------------------------------------------- diag
@torch.no_grad()
def diag(args):
    model = load_model(args.model)
    imgs = load_images(args.forget)
    N = len(imgs)
    g = torch.Generator(device=DEV).manual_seed(args.seed)
    print(f"N={N}  pixel mean={imgs.float().mean() / 255:.3f}")

    z = predict_u8(model, imgs)
    z32 = predict_u8(model, imgs, fp16=False)
    y = z.argmax(1)
    p = z.softmax(1)
    conf = p.max(1).values
    L = ce(z, y)
    print("\n== 1. MEMORISATION (clean forget images, pseudo-label = argmax) ==")
    print(f"fp16 vs fp32: max|dlogit|={float((z - z32).abs().max()):.4f}  argmax agree={float((z.argmax(1) == z32.argmax(1)).float().mean()):.4f}")
    print(f"confidence: mean={conf.mean():.4f}  quantiles(5/25/50/75/95)={q(conf)}")
    print(f"frac conf>0.99={float((conf > 0.99).float().mean()):.3f}  >0.9={float((conf > 0.9).float().mean()):.3f}  <0.5={float((conf < 0.5).float().mean()):.3f}")
    print(f"CE(pseudo-label): mean={L.mean():.4f} std={L.std():.4f}  quantiles={q(L)}")
    ent = -(p * p.clamp_min(1e-12).log()).sum(1)
    print(f"entropy: mean={ent.mean():.4f}  quantiles={q(ent)}")

    print("\n== 2. LOGIT SCALE ==")
    top2 = z.topk(2, 1).values
    print(f"max logit: mean={top2[:, 0].mean():.3f} quantiles={q(top2[:, 0])}")
    print(f"margin top1-top2: mean={(top2[:, 0] - top2[:, 1]).mean():.3f} quantiles={q(top2[:, 0] - top2[:, 1])}")
    print(f"per-sample logit std: mean={z.std(1).mean():.3f}   logit mean={z.mean():.3f}")
    zw = z.clone()
    zw[torch.arange(N), y] = float("nan")
    print(f"wrong-class logits: mean={zw.nanmean():.3f}  std(within sample)={float(torch.nan_to_num(zw - zw.nanmean(1, keepdim=True)).pow(2).sum(1).div(99).sqrt().mean()):.3f}")
    pT = (z / 2).softmax(1)
    print(f"T=2 softmax: mean max prob={pT.max(1).values.mean():.4f}")
    head = model.head
    print(f"head: |W| row norm mean={head.weight.norm(dim=1).mean():.3f}  bias range=[{head.bias.min():.3f},{head.bias.max():.3f}]")

    print("\n== 3. PSEUDO-LABEL DISTRIBUTION ==")
    cnt = torch.bincount(y, minlength=NUM_CLASSES).cpu()
    print(f"classes present={int((cnt > 0).sum())}/100  min={int(cnt.min())} max={int(cnt.max())} mean={cnt.float().mean():.1f} std={cnt.float().std():.1f}")
    order = cnt.argsort(descending=True)
    print("top10 (class:count):", " ".join(f"{int(c)}:{int(cnt[c])}" for c in order[:10]))
    print("bottom10:", " ".join(f"{int(c)}:{int(cnt[c])}" for c in order[-10:]))
    print("counts sorted:", cnt.sort(descending=True).values.tolist())

    print("\n== 4. ROBUSTNESS TO ALTERATIONS (vs clean pseudo-label) ==")
    print(f"{'alteration':<14}{'agree':>7}{'conf(y)':>9}{'CE mean':>9}{'CE std':>8}{'KL_T2(clean||alt)':>19}")
    for name, fn in NAMED.items():
        za = predict_u8(model, imgs, fn=fn)
        La = ce(za, y)
        print(f"{name:<14}{float((za.argmax(1) == y).float().mean()):>7.3f}{float(za.softmax(1)[torch.arange(N), y].mean()):>9.3f}"
              f"{La.mean():>9.3f}{La.std():>8.3f}{kl_T(z, za).mean():>19.4f}")
    print("random composite alterations (augment(), K=4 views):")
    for st in (0.25, 0.5, 0.75, 1.0):
        Ls, ag, kls = [], [], []
        for _ in range(4):
            za = predict_u8(model, imgs, fn=lambda x: augment(x, st, g))
            Ls.append(ce(za, y)); ag.append((za.argmax(1) == y).float()); kls.append(kl_T(z, za))
        Ls = torch.stack(Ls)
        print(f"  strength={st:.2f}  agree={torch.stack(ag).mean():.3f}  CE mean={Ls.mean():.3f}  std(per view)={Ls.std():.3f}  "
              f"std(of per-image mean)={Ls.mean(0).std():.3f}  KL_T2={torch.stack(kls).mean():.4f}  CE quantiles={q(Ls.flatten())}")

    print("\n== 5. ANCHOR PROXIES (original model on synthetic 'other' images) ==")
    for kind in ("crop", "mos2", "mos3", "mix", "cutmix"):
        confs, ents = [], []
        for i in range(0, N, 500):
            za = predict01(model, make_anchor(to01(imgs[i:i + 500]), g, [kind]))
            pa = za.softmax(1)
            confs.append(pa.max(1).values); ents.append(-(pa * pa.clamp_min(1e-12).log()).sum(1))
        print(f"  {kind:<7} mean conf={torch.cat(confs).mean():.3f}  mean entropy={torch.cat(ents).mean():.3f}")

    print("\n== 6. WHAT A NON-MEMORISED IMAGE LOOKS LIKE (hints for gold_mean / gold_std) ==")
    print("CE vs pseudo-label on half-image crops (zoom 0.5, random position) - a rough 'unseen image of same class' proxy:")
    Ls = []
    for _ in range(4):
        fn = lambda x: affine(x, torch.full((len(x),), 0.5, device=DEV), torch.zeros(len(x), device=DEV),
                              torch.ones(len(x), device=DEV), _rand(g, len(x)) * 2 - 1, _rand(g, len(x)) * 2 - 1)
        Ls.append(ce(predict_u8(model, imgs, fn=fn), y))
    Ls = torch.stack(Ls)
    print(f"  CE mean={Ls.mean():.3f} std={Ls.std():.3f} acc={float((Ls < math.log(2)).float().mean()):.3f}(approx) quantiles={q(Ls.flatten())}")


# ----------------------------------------------------------------------------- targets
@torch.no_grad()
def raw_targets(orig, imgs, K, strength, seed, cache_dir):
    f = Path(cache_dir) / f"raw3_K{K}_s{strength}_seed{seed}.pt"
    if f.exists():
        return torch.load(f, map_location=DEV)
    g = torch.Generator(device=DEV).manual_seed(seed)
    zc = predict_u8(orig, imgs)
    y = zc.argmax(1)
    za_sum = torch.zeros_like(zc)
    p2_sum = torch.zeros_like(zc)
    p1_sum = torch.zeros_like(zc)
    diff = torch.zeros(len(imgs), device=DEV)
    for _ in range(K):
        za = predict_u8(orig, imgs, fn=lambda x: augment(x, strength, g))
        za_sum += za
        p2_sum += (za / 2).softmax(1)
        p1_sum += za.softmax(1)
        diff += ce(za, y)
    out = {"zc": zc, "za": za_sum / K, "p2": p2_sum / K, "p1": p1_sum / K, "diff": diff / K, "y": y}
    f.parent.mkdir(exist_ok=True, parents=True)
    tmp = f.with_suffix(f".tmp{os.getpid()}")
    torch.save(out, tmp)
    os.replace(tmp, f)
    return out


@torch.no_grad()
def raw_targets_noise(model_path, orig, imgs, M, agree, seed, cache_dir):
    """Same outputs as raw_targets, but the 'views' are weight-noised copies of the original model evaluated on the
    CLEAN images (memorised examples are the first to break under weight noise). The noise level is found by
    bisection so that the noisy model agrees with the clean pseudo-labels on a fraction `agree` of the images."""
    f = Path(cache_dir) / f"rawN_M{M}_a{agree}_seed{seed}.pt"
    if f.exists():
        return torch.load(f, map_location=DEV)
    zc = predict_u8(orig, imgs)
    y = zc.argmax(1)
    sd0 = {k: v.clone() for k, v in orig.state_dict().items()}
    noisy = build_model().to(DEV).eval()

    def run(sigma, sd_seed):
        g = torch.Generator(device=DEV).manual_seed(sd_seed)
        sd = {}
        for k, v in sd0.items():
            if v.ndim >= 2 and v.is_floating_point():
                sd[k] = v + sigma * v.std() * torch.randn(v.shape, generator=g, device=DEV, dtype=v.dtype)
            else:
                sd[k] = v
        noisy.load_state_dict(sd, strict=True)
        return predict_u8(noisy, imgs)

    lo, hi = 0.0, 1.0
    for _ in range(9):
        mid = (lo + hi) / 2
        a = float((run(mid, seed + 1000).argmax(1) == y).float().mean())
        if a > agree:
            lo = mid
        else:
            hi = mid
    sigma = (lo + hi) / 2
    za_sum, p2_sum, p1_sum = torch.zeros_like(zc), torch.zeros_like(zc), torch.zeros_like(zc)
    diff = torch.zeros(len(imgs), device=DEV)
    ags = []
    for m in range(M):
        za = run(sigma, seed + 2000 + m)
        za_sum += za; p2_sum += (za / 2).softmax(1); p1_sum += za.softmax(1); diff += ce(za, y)
        ags.append(float((za.argmax(1) == y).float().mean()))
    print(f"weight-noise targets: sigma={sigma:.4f}  agreement per noisy model: mean={np.mean(ags):.3f} min={np.min(ags):.3f} max={np.max(ags):.3f}"
          f"  ensemble acc={float(((p1_sum).argmax(1) == y).float().mean()):.3f}")
    out = {"zc": zc, "za": za_sum / M, "p2": p2_sum / M, "p1": p1_sum / M, "diff": diff / M, "y": y}
    f.parent.mkdir(exist_ok=True, parents=True)
    tmp = f.with_suffix(f".tmp{os.getpid()}")
    torch.save(out, tmp)
    os.replace(tmp, f)
    return out


def get_raw(args, orig, imgs):
    if args.tsrc == "noise":
        return raw_targets_noise(args.model, orig, imgs, args.K, args.noise_agree, args.seed, args.cache)
    return raw_targets(orig, imgs, args.K, args.target_strength, args.seed, args.cache)


def rank_map(diff, mean, std, max_loss=14.0):
    """Map difficulty ranks onto a lognormal loss distribution with the requested mean / std."""
    N = len(diff)
    s2 = math.log(1 + (std / mean) ** 2)
    mu = math.log(mean) - s2 / 2
    u = (torch.arange(N, device=diff.device, dtype=torch.float64) + 0.5) / N
    L = torch.exp(mu + math.sqrt(s2) * math.sqrt(2) * torch.erfinv(2 * u - 1)).clamp(1e-5, max_loss)
    for _ in range(20):  # correct the finite-sample / clipping bias
        L = (L * (mean / L.mean())).clamp(1e-5, max_loss)
        m = L.mean()
        L = (m + (L - m) * (std / L.std())).clamp(1e-5, max_loss)
    out = torch.empty(N, device=diff.device, dtype=torch.float64)
    out[diff.argsort()] = L
    return out.float()


def rank_map_mix(diff, mean, acc, max_loss=14.0, y=None):
    """Mixture: a fraction `acc` of correct images (loss < ln 2, mostly tiny) and wrong images with an exponential tail.
    The tail scale is solved so that the overall mean is `mean`; the std is emergent."""
    N = len(diff)
    order = diff.argsort()
    if SHAPE.get("ca") and y is not None:
        # class-aware accuracy: classes with many forget images (gold lost most of their training data) get a low
        # accuracy a_lo, classes with few forget images a high accuracy a_hi (linear in the per-class forget count)
        a_lo, a_hi = SHAPE["ca"]
        cnt = torch.bincount(y, minlength=NUM_CLASSES).double()
        acc_c = a_hi + (a_lo - a_hi) * cnt / cnt.max()
        correct = torch.zeros(N, dtype=torch.bool, device=diff.device)
        for c in cnt.nonzero().flatten().tolist():
            ic = (y == c).nonzero().flatten()
            k = int(round(float(acc_c[c]) * len(ic)))
            correct[ic[diff[ic].argsort()[:k]]] = True
        order = torch.cat([order[correct[order]], order[~correct[order]]])   # correct ones first, each part by difficulty
        acc = float(correct.float().mean())
        print(f"class-aware targets: acc per class from {float(acc_c[cnt > 0].min()):.2f} to {float(acc_c[cnt > 0].max()):.2f}, overall {acc:.3f}")
    n_c = int(round(acc * N)); n_w = N - n_c
    v = (torch.arange(n_c, device=diff.device, dtype=torch.float64) + 0.5) / max(n_c, 1)
    Lc = 0.69 * v ** SHAPE["pow"] + 1e-4
    v = (torch.arange(n_w, device=diff.device, dtype=torch.float64) + 0.5) / n_w
    s = max((mean * N - float(Lc.sum())) / max(n_w, 1) - 0.7, 0.05)
    if SHAPE["tail"] == "uniform":   # bounded wrong losses in [0.7, lmax] (mean is then set by acc and lmax)
        Lw = 0.7 + (SHAPE["lmax"] - 0.7) * v
    else:
        Lw = (0.7 + s * (-torch.log1p(-v))).clamp(max=min(max_loss, SHAPE["lmax"]))
    L = torch.cat([Lc, Lw])
    out = torch.empty(N, device=diff.device, dtype=torch.float64)
    out[order] = L
    return out.float()


SHAPE = {"shape": "mix", "acc": 0.8, "pow": 3.0, "type": "rank", "base": "logit", "gamma": 1.0, "tail": "exp", "lmax": 14.0, "wtemp": 1.0}


def set_shape(a):
    SHAPE.update(shape=a.target_shape, acc=a.gold_acc, pow=a.correct_pow, type=a.target_type,
                 base="p2" if a.target_type == "soft" else a.base, gamma=a.ens_gamma, tail=a.tail, lmax=a.max_loss, wtemp=a.wrong_temp, maskf=a.mask_forget,
                 ca=[float(v) for v in a.class_aware.split(",")] if a.class_aware else None)


def make_target_logits(raw, mean, std, wrong_mix=0.0):
    """Target logits: wrong classes from altered-view logits, true-class logit set so that CE == target loss."""
    y = raw["y"]
    N = len(y)
    if SHAPE["type"] == "ens":  # target = probability average over views / noisy models (an ensemble)
        t = (SHAPE["gamma"] * torch.log(raw["p1"].double().clamp_min(1e-7))).float()   # gamma>1 sharpens (more loss spread)
        return t, ce(t, y)
    if SHAPE["base"] == "p2":   # logits whose T=2 softmax equals the view-averaged T=2 probabilities
        soft = 2 * torch.log(raw["p2"].double().clamp_min(1e-9))
        if SHAPE["type"] == "soft":  # no rank mapping: the target IS the view ensemble
            return soft.float(), ce(soft.float(), y)
        base = soft.clone()
    else:
        base = (wrong_mix * raw["zc"] + (1 - wrong_mix) * raw["za"]).clone().double()
    L = (rank_map_mix(raw["diff"], mean, SHAPE["acc"], y=y) if SHAPE["shape"] == "mix" else rank_map(raw["diff"], mean, std)).double()
    if SHAPE.get("maskf"):   # gold may never have seen ANY forget class: no mass on the other forget classes either
        S = torch.bincount(y, minlength=NUM_CLASSES) > 0
        if SHAPE["maskf"] > 0:        # remove the forget classes
            low = base.min(1, keepdim=True).values - SHAPE["maskf"]
            base = torch.where(S.unsqueeze(0), low.expand_as(base), base)
        elif SHAPE["maskf"] <= -100:  # remove the NON-forget classes instead
            low = base.min(1, keepdim=True).values - 3.0
            base = torch.where(S.unsqueeze(0), base, low.expand_as(base))
        else:                         # bonus on the forget-class logits
            base = base + (-SHAPE["maskf"]) * S.unsqueeze(0).to(base.dtype)
    if SHAPE["wtemp"] != 1.0:   # flatten (>1) the wrong-class logits: hedges WHICH wrong class gold picks
        base = (base - base.mean(1, keepdim=True)) / SHAPE["wtemp"]
    zw = base.clone()
    zw[torch.arange(N), y] = -float("inf")
    lse = torch.logsumexp(zw, 1)
    p = torch.exp(-L).clamp(1e-7, 1 - 1e-7)
    base[torch.arange(N), y] = lse + torch.log(p) - torch.log1p(-p)
    return base.float(), L.float()


# ----------------------------------------------------------------------------- proxy eval
@torch.no_grad()
def proxy_eval(model, orig, imgs, raw, tgt, tgt_alt, seed=12345, tag="", hold=None):
    """PROXY metrics versus our own targets (NOT the real gold) + drift on anchor images."""
    y = raw["y"]
    g = torch.Generator(device=DEV).manual_seed(seed)
    res = {}
    ms, ps = [], []
    for name, T_, fn in (("clean", tgt, None), ("alt", tgt_alt, lambda x: augment(x, 0.5, g))):
        g.manual_seed(seed)
        zs = predict_u8(model, imgs, fn=fn)
        g.manual_seed(seed)
        zo = predict_u8(orig, imgs, fn=fn)
        Lt, Ls, Lo = ce(T_, y), ce(zs, y), ce(zo, y)
        d_s = abs(Ls.mean() - Lt.mean()) + abs(Ls.std() - Lt.std())
        d_o = abs(Lo.mean() - Lt.mean()) + abs(Lo.std() - Lt.std())
        m = 1 - float((d_s / d_o).clamp(0, 1))
        pr = 1 - float((kl_T(T_, zs).mean() / kl_T(T_, zo).mean()).clamp(0, 1))
        ms.append(m); ps.append(pr)
        print(f"  [{name:5}] CE sub {Ls.mean():.3f}/{Ls.std():.3f}  target {Lt.mean():.3f}/{Lt.std():.3f}  orig {Lo.mean():.3f}/{Lo.std():.3f}"
              f"  | meanstd={m:.3f} pred={pr:.3f}  acc_vs_pseudo={float((zs.argmax(1) == y).float().mean()):.3f}")
    res["forget_proxy"] = float(np.mean(ms) * np.mean(ps))
    if hold is not None and len(hold):  # images NOT used for unlearning: true locality check
        zo, zs = predict_u8(orig, imgs[hold]), predict_u8(model, imgs[hold])
        yh = raw["y"][hold]
        print(f"  [HOLDOUT clean n={len(hold)}] KL_T2(orig||sub)={kl_T(zo, zs).mean():.4f}  agree={float((zs.argmax(1) == yh).float().mean()):.3f}"
              f"  CE sub {ce(zs, yh).mean():.3f}/{ce(zs, yh).std():.3f}")
        g.manual_seed(seed)
        fn = lambda x: make_anchor(x, g, ["crop"])
        zo = predict_u8(orig, imgs[hold], fn=fn); g.manual_seed(seed); zs = predict_u8(model, imgs[hold], fn=fn)
        print(f"  [HOLDOUT crops] KL_T2(orig||sub)={kl_T(zo, zs).mean():.4f}  agree(orig,sub)={float((zs.argmax(1) == zo.argmax(1)).float().mean()):.3f}")
    kls, agree = [], []
    for kind in ("crop", "mos2", "mos3", "mix", "cutmix"):
        k_, a_ = [], []
        for i in range(0, len(imgs), 500):
            xa = make_anchor(to01(imgs[i:i + 500]), g, [kind])
            zo, zs = predict01(orig, xa), predict01(model, xa)
            k_.append(kl_T(zo, zs)); a_.append((zo.argmax(1) == zs.argmax(1)).float())
        kls.append(float(torch.cat(k_).mean())); agree.append(float(torch.cat(a_).mean()))
        print(f"  [anchor {kind:6}] KL_T2(orig||sub)={kls[-1]:.4f}  argmax agree={agree[-1]:.3f}")
    res["anchor_kl"] = float(np.mean(kls))
    res["anchor_agree"] = float(np.mean(agree))
    res["retain_proxy"] = res["anchor_agree"] / (1 + res["anchor_kl"])
    print(f"  PROXY{tag}: forget={res['forget_proxy']:.3f} (meanstd {np.mean(ms):.3f} x pred {np.mean(ps):.3f})  "
          f"retain={res['retain_proxy']:.3f} (anchor KL {res['anchor_kl']:.4f})  product={res['forget_proxy'] * res['retain_proxy']:.3f}"
          f"   <-- vs OUR targets, not the real gold")
    return res


def save_checked(model, path):
    sd = {k: v.detach().cpu().float() for k, v in model.state_dict().items()}
    torch.save(sd, path)
    check_file(path)


def check_file(path):
    sd = torch.load(path, map_location="cpu", weights_only=True)
    m = build_model()
    m.load_state_dict(sd, strict=True)
    bad = [k for k, v in sd.items() if v.dtype != torch.float32 or not torch.isfinite(v).all()]
    size = os.path.getsize(path) / 1e6
    ok = not bad and size <= 450
    print(f"CHECK {path}: strict load OK, {len(sd)} tensors, bad(dtype/NaN/Inf)={bad}, size={size:.1f} MB -> {'VALID' if ok else 'INVALID'}")
    return m.to(DEV).eval()


# ----------------------------------------------------------------------------- train
def select_params(model, scope):
    if scope == "all":
        return [p for p in model.parameters()]
    names = []
    if scope.startswith("last"):
        n = int(scope[4:])
        nb = len(model.blocks)
        names = [f"blocks.{i}." for i in range(nb - n, nb)] + ["norm.", "head."]
    elif scope == "head":
        names = ["head."]
    else:
        raise ValueError(scope)
    ps = []
    for k, p in model.named_parameters():
        p.requires_grad_(any(k.startswith(n) for n in names))
        if p.requires_grad:
            ps.append(p)
    return ps


def train_moments(args):
    """Second stage: start from --init and directly match the mean / std of the per-image loss (exact and altered
    images) to --moments / --moments_alt, while keeping the parent's wrong-class structure on the forget images
    and the ORIGINAL model's behaviour on anchor images."""
    torch.manual_seed(args.seed)
    orig = load_model(args.model)
    parent = load_model(args.init)
    for p in list(orig.parameters()) + list(parent.parameters()):
        p.requires_grad_(False)
    imgs = load_images(args.forget)
    N = len(imgs)
    raw = get_raw(args, orig, imgs)
    y = raw["y"]
    tgt, _ = make_target_logits(raw, args.gold_mean, args.gold_std, args.wrong_mix)
    mu, sd = [float(v) for v in args.moments.split(",")]
    mu_a, sd_a = [float(v) for v in (args.moments_alt or args.moments).split(",")]
    print(f"moment matching from {args.init}: exact {mu}/{sd}  altered {mu_a}/{sd_a}")
    print("parent before:")
    proxy_eval(parent, orig, imgs, raw, tgt, tgt)
    model = load_model(args.init)
    params = select_params(model, args.scope)
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.1)
    scaler = torch.amp.GradScaler("cuda")
    g = torch.Generator(device=DEV).manual_seed(args.seed + 1)
    kinds = args.anchor_kinds.split(",")
    T = 2.0

    def wrong_kl(t, s_, yy):   # KL at T=2 between the distributions over the NON-label classes
        m = F.one_hot(yy, NUM_CLASSES).bool()
        return kl_T(t.float().masked_fill(m, -1e4), s_.float().masked_fill(m, -1e4), T).mean() * T * T

    t0 = time.time()
    for step in range(1, args.steps + 1):
        idx = torch.randint(N, (args.bs,), generator=g, device=DEV)
        x = to01(imgs[idx])
        with torch.no_grad():
            st = args.alt_min + (args.alt_max - args.alt_min) * _rand(g, len(idx))
            xa = augment(x, st, g)
            pool = to01(imgs[torch.randint(N, (args.bs_anchor,), generator=g, device=DEV)])
            xm = make_anchor(pool, g, kinds)
            with torch.autocast("cuda", dtype=torch.float16):
                zm_o = orig(norm(xm)).float()
                zp = parent(norm(torch.cat([x, xa]))).float()
        with torch.autocast("cuda", dtype=torch.float16):
            out = model(norm(torch.cat([x, xa, xm]))).float()
        B = len(idx)
        zs, zsa, zm = out[:B], out[B:2 * B], out[2 * B:]
        Lc, La = ce(zs, y[idx]), ce(zsa, y[idx])
        l_mom = (Lc.mean() - mu) ** 2 + (Lc.std() - sd) ** 2 + (La.mean() - mu_a) ** 2 + (La.std() - sd_a) ** 2
        l_keep = wrong_kl(zp[:B], zs, y[idx]) + wrong_kl(zp[B:], zsa, y[idx])
        l_r = (kl_T(zm_o, zm, T) * T * T).mean()
        loss = args.w_mom * l_mom + args.w_keep * l_keep + args.w_anchor * l_r
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        if step % 25 == 0 or step == args.steps:
            print(f"step {step}/{args.steps} loss {loss.item():.4f} mom {l_mom.item():.4f} (exact {Lc.mean().item():.2f}/{Lc.std().item():.2f} "
                  f"alt {La.mean().item():.2f}/{La.std().item():.2f}) keep {l_keep.item():.4f} anchor {l_r.item():.4f} {time.time() - t0:.0f}s", flush=True)
    model.eval()
    save_checked(model, args.out)
    proxy_eval(check_file(args.out), orig, imgs, raw, tgt, tgt, tag=f" {args.out}")


def train(args):
    if args.moments:
        return train_moments(args)
    torch.manual_seed(args.seed)
    orig = load_model(args.model)
    for p in orig.parameters():
        p.requires_grad_(False)
    imgs = load_images(args.forget)
    N = len(imgs)
    raw = get_raw(args, orig, imgs)
    alt_mean = args.alt_mean if args.alt_mean is not None else args.gold_mean
    alt_std = args.alt_std if args.alt_std is not None else args.gold_std
    tgt, L = make_target_logits(raw, args.gold_mean, args.gold_std, args.wrong_mix)
    tgt_alt, L_alt = make_target_logits(raw, alt_mean, alt_std, args.wrong_mix)
    y = raw["y"]
    easy = torch.zeros(N, dtype=torch.bool, device=DEV)
    if args.easy_exact > 0:  # easiest images keep EXACTLY the original output: memorised same-class negatives
        easy[L.argsort()[:int(args.easy_exact * N)]] = True
        tgt[easy] = raw["zc"][easy].float()
        tgt_alt[easy] = raw["zc"][easy].float()
        print(f"easy_exact: {int(easy.sum())} images keep the original logits")
    hp = torch.randperm(N, generator=torch.Generator().manual_seed(777)).to(DEV)
    hold, tr = hp[:args.holdout], hp[args.holdout:]
    Ntr = len(tr)
    print(f"targets: clean CE {L.mean():.3f}/{L.std():.3f}  alt CE {L_alt.mean():.3f}/{L_alt.std():.3f}  "
          f"target acc vs pseudo-label: {float((tgt.argmax(1) == y).float().mean()):.3f}  "
          f"difficulty mean={raw['diff'].mean():.3f}")
    print("orig model before training:")
    proxy_eval(orig, orig, imgs, raw, tgt, tgt_alt)

    model = load_model(args.model)
    params = select_params(model, args.scope)
    p0 = [p.detach().clone() for p in params]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    steps = args.epochs * (Ntr // args.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.1)
    scaler = torch.amp.GradScaler("cuda")
    g = torch.Generator(device=DEV).manual_seed(args.seed + 1)
    kinds = args.anchor_kinds.split(",")
    T = 2.0

    # memorisation score m(x) = KL_T2(p_orig(x) || q_aug(x)); optional per-image weight of the unlearning loss
    lp = F.log_softmax(raw["zc"].float() / 2, 1)
    mem = (lp.exp() * (lp - torch.log(raw["p2"].float().clamp_min(1e-9)))).sum(1)
    wimg = (mem / mem.mean()).clamp(0.05, 8.0) ** args.mem_weight if args.mem_weight > 0 else torch.ones_like(mem)
    wimg = wimg / wimg.mean()
    print(f"memorisation score: mean={mem.mean():.3f} quantiles={q(mem)}  weight range=[{wimg.min():.2f},{wimg.max():.2f}]")

    def distill(s, t, w=None):
        l = kl_T(t, s, 1.0) * args.w_t1 + kl_T(t, s, T) * T * T
        return (l * w).mean() if w is not None else l.mean()

    step, t0 = 0, time.time()
    for ep in range(args.epochs):
        perm = tr[torch.randperm(Ntr, generator=g, device=DEV)]
        for b in range(Ntr // args.bs):
            idx = perm[b * args.bs:(b + 1) * args.bs]
            x = to01(imgs[idx])
            with torch.no_grad():
                st = args.alt_min + (args.alt_max - args.alt_min) * _rand(g, len(idx))
                xa = augment(x, st, g)
                pool = to01(imgs[tr[torch.randint(Ntr, (args.bs_anchor,), generator=g, device=DEV)]])
                xm = make_anchor(pool, g, kinds)
                with torch.autocast("cuda", dtype=torch.float16):
                    zm_o = orig(norm(xm)).float()
            with torch.autocast("cuda", dtype=torch.float16):
                out = model(norm(torch.cat([x, xa, xm]))).float()
            zs, zsa, zm = out[:len(idx)], out[len(idx):2 * len(idx)], out[2 * len(idx):]
            l_f = distill(zs, tgt[idx], wimg[idx])
            if args.alt_anchor:   # altered views are kept on the ORIGINAL model's own output for that view
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                    za_o = orig(norm(xa)).float()
                l_a = distill(zsa, za_o)
            elif args.alt_mode == "max" or args.easy_exact > 0:
                # per view: easy images -> original output on that view; others -> the less memorised of
                # (image target, original output on that view) when alt_mode == max
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                    za_o = orig(norm(xa)).float()
                ta = tgt_alt[idx].clone()
                use_o = easy[idx].clone()
                if args.alt_mode == "max":
                    use_o |= ce(za_o, y[idx]) > ce(ta, y[idx])
                ta[use_o] = za_o[use_o]
                l_a = distill(zsa, ta, wimg[idx])
            else:
                l_a = distill(zsa, tgt_alt[idx], wimg[idx])
            l_r = (kl_T(zm_o, zm, T) * T * T).mean() + args.w_mse * F.mse_loss(zm, zm_o)
            loss = args.w_clean * l_f + args.w_alt * l_a + args.w_anchor * l_r
            if args.l2sp > 0:
                loss = loss + args.l2sp * sum(((p - q0) ** 2).sum() for p, q0 in zip(params, p0))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            if step % 25 == 0 or step == steps:
                print(f"ep {ep} step {step}/{steps}  loss {loss.item():.4f}  forget {l_f.item():.4f}  alt {l_a.item():.4f}  "
                      f"anchor {l_r.item():.5f}  lr {sched.get_last_lr()[0]:.2e}  {time.time() - t0:.0f}s", flush=True)
        model.eval()
        print(f"-- end of epoch {ep}")
        if ep == args.epochs - 1:
            proxy_eval(model, orig, imgs, raw, tgt, tgt_alt, hold=hold)
    dist = math.sqrt(sum(float(((p.detach() - q0) ** 2).sum()) for p, q0 in zip(params, p0)))
    print(f"weight distance to original: {dist:.4f}")
    save_checked(model, args.out)
    print("reloaded file:")
    proxy_eval(check_file(args.out), orig, imgs, raw, tgt, tgt_alt, tag=f" {args.out}")


# ----------------------------------------------------------------------------- interp / check
def _eval_ctx(args):
    orig = load_model(args.model)
    imgs = load_images(args.forget)
    raw = get_raw(args, orig, imgs)
    tgt, _ = make_target_logits(raw, args.gold_mean, args.gold_std, args.wrong_mix)
    tgt_alt, _ = make_target_logits(raw, args.alt_mean if args.alt_mean is not None else args.gold_mean,
                                    args.alt_std if args.alt_std is not None else args.gold_std, args.wrong_mix)
    return orig, imgs, raw, tgt, tgt_alt


def interp(args):
    orig, imgs, raw, tgt, tgt_alt = _eval_ctx(args)
    so = torch.load(args.model, map_location="cpu", weights_only=True)
    su = torch.load(args.unlearned, map_location="cpu", weights_only=True)
    stem = Path(args.unlearned).with_suffix("")
    for a in [float(v) for v in args.alphas.split(",")]:
        sd = {k: (so[k].float() + a * (su[k].float() - so[k].float())) for k in so}
        out = f"{stem}_a{a:g}.pt"
        torch.save(sd, out)
        print(f"\n=== alpha={a:g} -> {out}")
        proxy_eval(check_file(out), orig, imgs, raw, tgt, tgt_alt, tag=f" alpha={a:g}")


def check(args):
    orig, imgs, raw, tgt, tgt_alt = _eval_ctx(args)
    proxy_eval(check_file(args.unlearned), orig, imgs, raw, tgt, tgt_alt, tag=f" {args.unlearned}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["diag", "train", "interp", "check", "multi"], required=True)
    ap.add_argument("--model", default="../model.pt")
    ap.add_argument("--forget", default="../forget_images.npz")
    ap.add_argument("--out", default="sub.pt")
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--seed", type=int, default=0)
    # targets
    ap.add_argument("--gold_mean", type=float, default=1.0)
    ap.add_argument("--gold_std", type=float, default=1.5)
    ap.add_argument("--target_shape", default="mix", choices=["mix", "lognormal"])
    ap.add_argument("--gold_acc", type=float, default=0.8, help="mix shape: fraction of forget images gold classifies correctly")
    ap.add_argument("--correct_pow", type=float, default=3.0, help="mix shape: larger = correct images keep a tinier loss")
    ap.add_argument("--target_type", default="rank", choices=["rank", "soft", "ens"], help="soft: target = T=2 probability average of the original over altered views")
    ap.add_argument("--ens_gamma", type=float, default=1.0, help="ens targets: logits = gamma * log(mean prob); >1 sharpens")
    ap.add_argument("--tail", default="exp", choices=["exp", "uniform"], help="rank/mix targets: shape of the wrong-image losses")
    ap.add_argument("--max_loss", type=float, default=14.0)
    ap.add_argument("--mem_weight", type=float, default=0.0, help="exponent of the per-image memorisation weight (0 = off)")
    ap.add_argument("--class_aware", default="", help="'a_lo,a_hi': per-class target accuracy, low for classes with many forget images")
    ap.add_argument("--init", default=None, help="moment mode: parent state dict to start from")
    ap.add_argument("--moments", default="", help="'mean,std' of the per-image loss on exact forget images (enables moment mode)")
    ap.add_argument("--moments_alt", default="", help="'mean,std' on altered images (default: same as --moments)")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--w_mom", type=float, default=1.0)
    ap.add_argument("--w_keep", type=float, default=1.0)
    ap.add_argument("--mask_forget", type=float, default=0.0, help=">0: in the targets, push the logits of ALL forget classes this far below the row minimum")
    ap.add_argument("--wrong_temp", type=float, default=1.0, help="rank targets: temperature applied to the wrong-class logits")
    ap.add_argument("--alt_anchor", action="store_true", help="train altered views towards the original model instead of the forget targets")
    ap.add_argument("--tsrc", default="aug", choices=["aug", "noise"], help="views used to build the targets: altered images or weight-noised models")
    ap.add_argument("--noise_agree", type=float, default=0.7, help="tsrc noise: agreement of a noisy model with the clean pseudo-labels")
    ap.add_argument("--base", default="logit", choices=["logit", "p2"], help="source of the wrong-class structure in rank mode")
    ap.add_argument("--runs", default=None, help="multi mode: file with one 'name|args' per line, run sequentially")
    ap.add_argument("--alt_mean", type=float, default=None, help="target loss mean on altered copies (default gold_mean)")
    ap.add_argument("--alt_std", type=float, default=None)
    ap.add_argument("--K", type=int, default=16, help="altered views per image for the targets")
    ap.add_argument("--target_strength", type=float, default=0.75)
    ap.add_argument("--wrong_mix", type=float, default=0.0, help="weight of CLEAN orig logits in the wrong-class logits")
    # training
    ap.add_argument("--easy_exact", type=float, default=0.0, help="fraction of easiest images whose target is the original output itself")
    ap.add_argument("--alt_mode", default="target", choices=["target", "max"], help="max: altered view target = less memorised of (image target, orig on that view)")
    ap.add_argument("--holdout", type=int, default=0, help="forget images excluded from training (locality check)")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--bs_anchor", type=int, default=48)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--scope", default="all", help="all | head | lastN")
    ap.add_argument("--w_clean", type=float, default=1.0)
    ap.add_argument("--w_alt", type=float, default=1.0)
    ap.add_argument("--w_anchor", type=float, default=2.0)
    ap.add_argument("--w_t1", type=float, default=1.0, help="weight of the T=1 KL term in the distillation")
    ap.add_argument("--w_mse", type=float, default=0.0, help="logit MSE on anchors")
    ap.add_argument("--l2sp", type=float, default=0.0)
    ap.add_argument("--alt_min", type=float, default=0.1)
    ap.add_argument("--alt_max", type=float, default=0.7)
    ap.add_argument("--anchor_kinds", default="crop,crop,mos2,mos3,mix,cutmix")
    # interp / check
    ap.add_argument("--unlearned", default="sub.pt")
    ap.add_argument("--alphas", default="0.5,0.75,1.25")
    args = ap.parse_args()
    if args.mode == "multi":
        import shlex
        t0 = time.time()
        for line in open(args.runs):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            name, a = line.split("|", 1)
            if time.time() - t0 > 50 * 60:
                print(f"SKIP {name}: time budget exhausted", flush=True)
                continue
            if a.startswith("CHECK "):
                print(f"\n######## RUN {name} | {a}", flush=True)
                chk = ap.parse_args(["--mode", "check", "--model", args.model, "--forget", args.forget, "--unlearned", a.split()[1]])
                set_shape(chk)
                check(chk)
                continue
            sub = ap.parse_args(["--mode", "train", "--model", args.model, "--forget", args.forget] + shlex.split(a) + ["--out", f"subs/{name}.pt"])
            print(f"\n######## RUN {name} | {a}", flush=True)
            set_shape(sub)
            try:
                train(sub)
            except Exception:
                import traceback
                traceback.print_exc()
                print(f"######## FAILED {name}", flush=True)
            torch.cuda.empty_cache()
            print(f"######## DONE {name} at {time.time() - t0:.0f}s", flush=True)
        return
    print(vars(args))
    set_shape(args)
    {"diag": diag, "train": train, "interp": interp, "check": check}[args.mode](args)


if __name__ == "__main__":
    main()
