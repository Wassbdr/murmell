"""Seeded synthetic dataset of VISIBLE watermarks with exact masks (no competition data involved).

Every item is a pure function of (seed, index): a procedural background, then one watermark layer
(text, logo or a tiled text pattern) with random opacity, size, rotation and position, alpha-composited
on top. The mask is the watermark's own coverage (alpha >= 0.5 before opacity), so it is exact, and it is
reduced to cells like the competition masks (a cell is 1 when at least half of its pixels are covered).

    python -m src.synth_visible --n 500 --seed 1 --out work/synth_eval     # writes images, masks, meta.json
Fonts: the DejaVu fonts shipped with matplotlib (Bitstream Vera / DejaVu licence, redistribution allowed).
"""
import argparse
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from scipy import ndimage as ndi

KINDS = ("text", "logo", "tiled")
WORDS = ["SAMPLE", "PREVIEW", "© STOCK", "PROOF", "DRAFT", "CONFIDENTIAL", "© 2026", "MURMELL", "PHOTO",
         "watermark", "NOT FOR SALE", "studio", "© Lumen", "COPY", "demo only", "ARCHIVE"]
FONT_FILES = ["DejaVuSans.ttf", "DejaVuSans-Bold.ttf", "DejaVuSerif.ttf", "DejaVuSerif-Bold.ttf",
              "DejaVuSansMono-Bold.ttf", "DejaVuSans-Oblique.ttf", "DejaVuSerif-BoldItalic.ttf"]


def _font_dir():
    import matplotlib
    return Path(matplotlib.get_data_path()) / "fonts" / "ttf"


@lru_cache(None)
def _font(name, size):
    return ImageFont.truetype(str(_font_dir() / name), size)


def to_cells(m, cells):
    """bool S x S -> float cells x cells (cell = 1 when >= half of its pixels are covered)."""
    k = m.shape[0] // cells
    return (m.reshape(cells, k, cells, k).mean((1, 3)) >= 0.5).astype(np.uint8)


# ------------------------------------------------------------------ backgrounds
def _smooth_field(rng, S, sigma):
    f = ndi.gaussian_filter(rng.randn(S // 4, S // 4).astype(np.float32), sigma / 4, mode="wrap")
    f = np.kron(f, np.ones((4, 4), np.float32))
    f = (f - f.min()) / (np.ptp(f) + 1e-6)
    return f


def background(rng, S):
    """procedural 'scene': gradient sky, smooth colour fields, random shapes, texture noise, optional blur."""
    c0, c1 = rng.randint(0, 256, 3), rng.randint(0, 256, 3)
    t = np.linspace(0, 1, S, dtype=np.float32)
    ang = rng.uniform(0, 2 * np.pi)
    g = (np.cos(ang) * t[None, :] + np.sin(ang) * t[:, None]); g = (g - g.min()) / (np.ptp(g) + 1e-6)
    img = c0[None, None] * (1 - g[..., None]) + c1[None, None] * g[..., None]
    for _ in range(rng.randint(1, 4)):                                  # smooth colour fields
        f = _smooth_field(rng, S, rng.uniform(8, 40))[..., None]
        img = img * (1 - 0.6 * f) + rng.randint(0, 256, 3)[None, None] * 0.6 * f
    im = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(im)
    for _ in range(rng.randint(2, 14)):                                 # objects
        col = tuple(int(v) for v in rng.randint(0, 256, 3))
        x0, y0 = rng.randint(-S // 4, S), rng.randint(-S // 4, S)
        w, h = rng.randint(S // 16, S // 2), rng.randint(S // 16, S // 2)
        k = rng.randint(4)
        if k == 0:
            d.ellipse([x0, y0, x0 + w, y0 + h], fill=col)
        elif k == 1:
            d.rectangle([x0, y0, x0 + w, y0 + h], fill=col)
        elif k == 2:
            pts = [(int(x0 + rng.uniform(0, w)), int(y0 + rng.uniform(0, h))) for _ in range(rng.randint(3, 7))]
            d.polygon(pts, fill=col)
        else:
            d.line([x0, y0, x0 + w, y0 + h], fill=col, width=int(rng.randint(1, 8)))
    if rng.rand() < 0.5:
        im = im.filter(ImageFilter.GaussianBlur(rng.uniform(0.5, 3)))
    a = np.asarray(im).astype(np.float32)
    a += rng.randn(S, S, 1).astype(np.float32) * rng.uniform(0, 12)      # sensor-like noise
    return np.clip(a, 0, 255).astype(np.uint8)


# ------------------------------------------------------------------ watermark layers (coverage in [0, 1])
def _text_layer(rng, S, size, text=None):
    text = text or WORDS[rng.randint(len(WORDS))]
    f = _font(FONT_FILES[rng.randint(len(FONT_FILES))], int(size))
    bb = f.getbbox(text)
    w, h = bb[2] - bb[0] + 8, bb[3] - bb[1] + 8
    L = Image.new("L", (w, h), 0)
    ImageDraw.Draw(L).text((4 - bb[0], 4 - bb[1]), text, font=f, fill=255)
    return L


def _logo_layer(rng, S, size):
    """procedural logo: ring or badge, inner star / polygon, optional initials."""
    n = int(size); L = Image.new("L", (n, n), 0); d = ImageDraw.Draw(L)
    lw = max(2, n // rng.randint(8, 16))
    shape = rng.randint(3)
    if shape == 0:
        d.ellipse([lw, lw, n - lw, n - lw], outline=255, width=lw)
    elif shape == 1:
        d.rounded_rectangle([lw, lw, n - lw, n - lw], radius=n // 5, outline=255, width=lw)
    else:
        k = rng.randint(5, 9); r = n / 2 - lw
        pts = [(n / 2 + r * np.cos(2 * np.pi * i / k), n / 2 + r * np.sin(2 * np.pi * i / k)) for i in range(k)]
        d.polygon(pts, outline=255, width=lw)
    if rng.rand() < 0.5:                                                # star
        k = rng.randint(4, 7); r1, r2 = n * 0.28, n * 0.12; pts = []
        for i in range(2 * k):
            r = r1 if i % 2 == 0 else r2; a = np.pi * i / k - np.pi / 2
            pts.append((n / 2 + r * np.cos(a), n / 2 + r * np.sin(a)))
        d.polygon(pts, fill=255)
    else:                                                               # initials
        f = _font("DejaVuSans-Bold.ttf", max(8, n // 3))
        s = "".join(chr(65 + rng.randint(26)) for _ in range(rng.randint(1, 3)))
        bb = f.getbbox(s)
        d.text(((n - (bb[2] - bb[0])) / 2 - bb[0], (n - (bb[3] - bb[1])) / 2 - bb[1]), s, font=f, fill=255)
    return L


def watermark_layer(rng, S, kind):
    """coverage layer S x S (float [0,1]) and the parameters drawn for it."""
    canvas = Image.new("L", (S, S), 0)
    rot = float(rng.uniform(-45, 45)) if rng.rand() < 0.7 else 0.0
    if kind == "tiled":
        size = rng.uniform(0.04, 0.09) * S
        t = _text_layer(rng, S, size)
        big = Image.new("L", (2 * S, 2 * S), 0)
        sx, sy = t.size[0] + int(rng.uniform(0.2, 1.0) * t.size[0]), int(t.size[1] * rng.uniform(2, 4))
        for j, y in enumerate(range(0, 2 * S, sy)):
            for x in range(-(j % 2) * sx // 2, 2 * S, sx):
                big.paste(t, (x, y), t)
        big = big.rotate(rot, resample=Image.BILINEAR)
        canvas = big.crop((S // 2, S // 2, S // 2 + S, S // 2 + S))
        frac = 1.0
    else:
        size = rng.uniform(0.06, 0.2) * S if kind == "text" else rng.uniform(0.15, 0.45) * S
        L = _text_layer(rng, S, size) if kind == "text" else _logo_layer(rng, S, size)
        L = L.rotate(rot, resample=Image.BILINEAR, expand=True)
        if L.size[0] > S or L.size[1] > S:
            L.thumbnail((S, S))
        x = rng.randint(-L.size[0] // 4, S - 3 * L.size[0] // 4); y = rng.randint(-L.size[1] // 4, S - 3 * L.size[1] // 4)
        canvas.paste(L, (x, y), L)
        frac = None
    a = np.asarray(canvas).astype(np.float32) / 255
    return a, dict(kind=kind, size_px=round(float(size), 1), rotation=round(rot, 1))


def make_item(seed, i, S=256, cells=128, p_clean=0.05):
    """(image uint8 S x S x 3, mask uint8 cells x cells, meta dict); deterministic in (seed, i)."""
    rng = np.random.RandomState((seed * 7919 + i * 104729 + 13) % 2 ** 31)
    img = background(rng, S).astype(np.float32)
    kind = KINDS[rng.choice(3, p=[0.45, 0.35, 0.20])]
    meta = dict(index=int(i), kind=kind)
    if rng.rand() < p_clean:
        meta.update(kind="clean", opacity=0.0)
        m = np.zeros((S, S), bool)
    else:
        a, p = watermark_layer(rng, S, kind); meta.update(p)
        op = float(rng.uniform(0.15, 0.9)); meta["opacity"] = round(op, 3)
        mode = rng.randint(3)                                           # white, black or coloured ink
        col = np.array([255, 255, 255]) if mode == 0 else (np.array([0, 0, 0]) if mode == 1 else rng.randint(0, 256, 3))
        if rng.rand() < 0.3:                                            # soft shadow under the ink
            sh = ndi.gaussian_filter(a, 1.5); sh = np.roll(sh, (1, 1), (0, 1))
            img = img * (1 - 0.4 * op * sh[..., None])
        alpha = (op * a)[..., None]
        img = img * (1 - alpha) + col[None, None] * alpha
        m = a >= 0.5
    img = np.clip(img, 0, 255).astype(np.uint8)
    if rng.rand() < 0.5:                                                # JPEG round trip
        import io
        q = int(rng.randint(50, 96)); buf = io.BytesIO(); Image.fromarray(img).save(buf, "JPEG", quality=q)
        img = np.asarray(Image.open(buf).convert("RGB")); meta["jpeg"] = q
    meta["coverage"] = round(float(m.mean()), 4)
    return img, to_cells(m, cells), meta


def make_set(n, seed, S=256, cells=128, folds=5):
    X = np.zeros((n, S, S, 3), np.uint8); M = np.zeros((n, cells, cells), np.uint8); meta = []
    for i in range(n):
        X[i], M[i], mt = make_item(seed, i, S, cells)
        mt["fold"] = int(i % folds); meta.append(mt)
    return X, M, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--size", type=int, default=256); ap.add_argument("--cells", type=int, default=128)
    ap.add_argument("--out", default="work/synth_eval"); ap.add_argument("--png", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    X, M, meta = make_set(a.n, a.seed, a.size, a.cells)
    np.savez_compressed(out / "data.npz", images=X, masks=M)
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    for i in range(min(a.png, a.n)):
        Image.fromarray(X[i]).save(out / f"img_{i:04d}.png")
        Image.fromarray(M[i] * 255).resize((a.size, a.size), Image.NEAREST).save(out / f"mask_{i:04d}.png")
    print(f"wrote {a.n} items to {out} (kinds: " + ", ".join(f"{k} {sum(m['kind'] == k for m in meta)}" for k in KINDS + ("clean",)) + ")")


if __name__ == "__main__":
    main()
