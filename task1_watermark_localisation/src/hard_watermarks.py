"""Seeded visible marks on procedural scenes and public-domain photo crops."""

import io
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from skimage import data

from src.synth_visible import WORDS, _font, _logo_layer, background


@dataclass(frozen=True)
class SceneConfig:
    fonts: tuple[str, ...]
    opacity: tuple[float, float]
    photos: tuple[str, ...]
    angles: tuple[float, float]
    staggered: bool


HARD = SceneConfig(
    ("DejaVuSans.ttf", "DejaVuSerif.ttf"),
    (0.14, 0.48),
    ("grass", "hubble_deep_field"),
    (-32, 32),
    False,
)
HELD_OUT = SceneConfig(
    ("DejaVuSansMono.ttf", "DejaVuSans-Oblique.ttf", "DejaVuSerif-Italic.ttf"),
    (0.06, 0.14),
    ("gravel", "brick"),
    (32, 58),
    True,
)


@dataclass(frozen=True)
class Scene:
    image: np.ndarray
    mask: np.ndarray
    kind: str
    opacity: float
    textured: bool
    photo: str | None
    background: np.ndarray
    alpha: np.ndarray
    ink: np.ndarray


@lru_cache
def _photo(name: str) -> Image.Image:
    return Image.fromarray(getattr(data, name)()).convert("RGB")


def _scene_background(
    rng: np.random.RandomState, size: int, config: SceneConfig, textured: bool
) -> tuple[Image.Image, str | None]:
    if textured:
        name = config.photos[rng.randint(len(config.photos))]
        photo = _photo(name)
        edge = rng.randint(size, min(photo.size) + 1)
        x, y = (rng.randint(axis - edge + 1) for axis in photo.size)
        canvas = photo.crop((x, y, x + edge, y + edge)).resize(
            (size, size), Image.Resampling.LANCZOS
        )
        if rng.rand() < 0.5:
            canvas = canvas.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        pixels = np.asarray(canvas, dtype=np.float32)
        pixels = pixels * rng.uniform(0.7, 1.3) + rng.uniform(-25, 25, (1, 1, 3))
        return Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8)), name
    return Image.fromarray(background(rng, size)), None


def _lettering(font: str, pixels: int, word: str) -> Image.Image:
    face = _font(font, pixels)
    x0, y0, x1, y1 = face.getbbox(word)
    layer = Image.new("L", (x1 - x0 + 4, y1 - y0 + 4))
    ImageDraw.Draw(layer).text((2 - x0, 2 - y0), word, font=face, fill=255)
    return layer


def _coverage(
    rng: np.random.RandomState, size: int, config: SceneConfig, kind: str, word: str
) -> np.ndarray:
    scale = 2
    side = size * scale
    angle = rng.uniform(*config.angles)
    if config.staggered and rng.rand() < 0.5:
        angle = -angle
    if kind == "logo":
        token = _logo_layer(rng, side, side * rng.uniform(0.16, 0.38))
    else:
        font = config.fonts[rng.randint(len(config.fonts))]
        token = _lettering(font, max(14, int(side * rng.uniform(0.08, 0.15))), word)
        if kind == "text":
            thinned = token.filter(ImageFilter.MinFilter(3))
            token = Image.blend(token, thinned, 0.25)
    if kind == "tiled":
        layer = Image.new("L", (side * 2, side * 2))
        dx = token.width + int(side * rng.uniform(0.07, 0.24))
        dy = token.height + int(side * rng.uniform(0.08, 0.20))
        phase = rng.randint(dx)
        for row, y in enumerate(range(-rng.randint(dy), 2 * side, dy)):
            offset = dx // 2 if config.staggered and row % 2 else 0
            for x in range(-phase - offset, 2 * side, dx):
                layer.paste(token, (x, y))
        layer = layer.rotate(angle, resample=Image.Resampling.BILINEAR)
        layer = layer.crop((side // 2, side // 2, 3 * side // 2, 3 * side // 2))
    else:
        token = token.rotate(angle, resample=Image.Resampling.BILINEAR, expand=True)
        token.thumbnail((side - 4, side - 4), Image.Resampling.LANCZOS)
        layer = Image.new("L", (side, side))
        if config.staggered:
            x = (side - token.width) // 2
            y = int((side - token.height) * rng.choice((0.2, 0.5, 0.8)))
        else:
            x = rng.randint(side - token.width + 1)
            y = rng.randint(side - token.height + 1)
        layer.paste(token, (x, y))
    return np.asarray(layer.resize((size, size), Image.Resampling.LANCZOS), dtype=np.float32) / 255


def make_scene(seed: int, index: int, size: int = 128, config: SceneConfig = HARD) -> Scene:
    if size < 64 or size > 256 or size % 32:
        raise ValueError("Use an image size divisible by 32 between 64 and 256 pixels.")
    rng = np.random.RandomState(np.random.SeedSequence([seed, index]).generate_state(1))
    textured = rng.rand() < 0.55
    canvas, photo = _scene_background(rng, size, config, textured)
    if rng.rand() < 0.4:
        letters = _lettering("DejaVuSans-Bold.ttf", rng.randint(9, max(10, size // 7)), str(rng.randint(100, 9999)))
        xy = (rng.randint(size - letters.width + 1), rng.randint(size - letters.height + 1))
        canvas.paste(tuple(int(c) for c in rng.randint(0, 256, 3)), xy, letters)
    kind = str(rng.choice(("text", "logo", "tiled", "clean"), p=(0.36, 0.29, 0.25, 0.10)))
    opacity = 0.0
    mask = np.zeros((size, size), dtype=bool)
    background_pixels = np.array(canvas)
    alpha = np.zeros((size, size), dtype=np.float32)
    color = np.zeros(3, dtype=np.uint8)
    if kind != "clean":
        opacity = rng.uniform(*config.opacity)
        coverage = _coverage(rng, size, config, kind, WORDS[rng.randint(len(WORDS))])
        mask = coverage >= 0.5
        ink = rng.choice(3)
        color = np.zeros(3) if ink == 0 else np.full(3, 255) if ink == 1 else rng.randint(0, 256, 3)
        alpha = coverage * opacity
        pixels = background_pixels * (1 - alpha[..., None]) + color * alpha[..., None]
        canvas = Image.fromarray(np.rint(pixels).astype(np.uint8))
    if rng.rand() < 0.30:
        with io.BytesIO() as stream:
            canvas.save(stream, "JPEG", quality=rng.randint(68, 96))
            canvas = Image.open(stream).convert("RGB")
    return Scene(np.array(canvas), mask, kind, float(opacity), textured, photo,
                 background_pixels, np.rint(alpha * 255).astype(np.uint8), color.astype(np.uint8))
