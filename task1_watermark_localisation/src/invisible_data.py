"""Seeded invisible-watermark edits on public-domain and CC0 photographs."""

import io
from dataclasses import dataclass
from functools import lru_cache
from typing import Protocol

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw
from skimage import data

SIZE = 512
CELLS = 128
REGIMES = ("holes", "rectangle", "object")


class Embedder(Protocol):
    family: str
    bits: int

    def embed(self, image: Image.Image, message: NDArray[np.uint8]) -> Image.Image: ...


@dataclass(frozen=True)
class EditConfig:
    photos: tuple[str, ...]
    jpeg: tuple[int, int]
    crop: tuple[int, int]
    patch: tuple[int, int]
    lobes: tuple[int, int]
    resample: tuple[int, int]


MAIN = EditConfig(("grass", "hubble_deep_field"), (89, 99), (0, 30), (92, 168), (6, 9), (440, 513))
HELD_OUT = EditConfig(("gravel", "brick"), (69, 89), (35, 73), (172, 235), (10, 15), (320, 440))


@dataclass(frozen=True)
class InvisibleScene:
    image: Image.Image
    support: NDArray[np.bool_]
    mask: NDArray[np.bool_]
    message: NDArray[np.uint8]
    family: str
    regime: str
    photo: str
    donor: str | None
    jpeg_quality: int
    crop_box: tuple[int, int, int, int]
    resize_edge: int


@lru_cache
def photo(name: str) -> Image.Image:
    return Image.fromarray(getattr(data, name)()).convert("RGB")


def photo_crop(source: Image.Image, size: int, rng: np.random.Generator) -> Image.Image:
    edge = int(rng.integers(max(1, min(source.size) // 2), min(source.size) + 1))
    x, y = (int(rng.integers(dim - edge + 1)) for dim in source.size)
    return source.crop((x, y, x + edge, y + edge)).resize((size, size), Image.Resampling.LANCZOS)


def irregular(size: int, rng: np.random.Generator, lobes: tuple[int, int]) -> Image.Image:
    count = int(rng.integers(*lobes))
    angles = np.arange(count) * (2 * np.pi / count) + rng.uniform(0, 2 * np.pi)
    radii = rng.uniform(0.25, 0.48, count) * size
    points = [(size / 2 + r * np.cos(a), size / 2 + r * np.sin(a)) for r, a in zip(radii, angles)]
    mask = Image.new("L", (size, size))
    ImageDraw.Draw(mask).polygon(points, fill=255)
    return mask


def cell_mask(support: NDArray[np.bool_]) -> NDArray[np.bool_]:
    """A cell is positive if at least half of its pixels carry the watermark."""
    if support.shape != (SIZE, SIZE):
        raise ValueError(f"Expected a {SIZE} by {SIZE} pixel support mask.")
    edge = SIZE // CELLS
    return support.reshape(CELLS, edge, CELLS, edge).sum(axis=(1, 3)) >= edge * edge / 2


def make_invisible_scene(
    embedder: Embedder, seed: int, index: int, config: EditConfig = MAIN,
) -> InvisibleScene:
    rng = np.random.default_rng(np.random.SeedSequence((seed, index)))
    regime = REGIMES[index % len(REGIMES)]
    name, donor_name = rng.permutation(config.photos)[:2]
    clean = photo_crop(photo(name), SIZE, rng)
    message = rng.integers(0, 2, embedder.bits, dtype=np.uint8)
    support = Image.new("L", clean.size, color=255 if regime == "holes" else 0)
    edge = int(rng.integers(*config.patch))
    x, y = (int(rng.integers(config.crop[1], SIZE - edge - config.crop[1])) for _ in range(2))
    if regime == "holes":
        canvas = embedder.embed(clean, message)
        for _ in range(int(rng.integers(1, 4))):
            x, y = (int(rng.integers(0, SIZE - edge)) for _ in range(2))
            hole = irregular(edge, rng, config.lobes)
            canvas.paste(clean.crop((x, y, x + edge, y + edge)), (x, y), hole)
            support.paste(0, (x, y), hole)
    else:
        canvas = clean.copy()
        box = (x, y, x + edge, y + edge)
        patch = clean.crop(box) if regime == "rectangle" else photo_crop(photo(donor_name), edge, rng)
        patch = embedder.embed(patch, message)
        outline = Image.new("L", (edge, edge), color=255) if regime == "rectangle" else irregular(
            edge, rng, config.lobes
        )
        canvas.paste(patch, (x, y), outline)
        support.paste(outline, (x, y))

    margins = rng.integers(*config.crop, size=4)
    crop = (int(margins[0]), int(margins[1]), SIZE - int(margins[2]), SIZE - int(margins[3]))
    resize = int(rng.integers(*config.resample))
    canvas = canvas.crop(crop).resize((resize, resize), Image.Resampling.LANCZOS)
    canvas = canvas.resize((SIZE, SIZE), Image.Resampling.BILINEAR)
    support = support.crop(crop).resize((resize, resize), Image.Resampling.NEAREST)
    support = support.resize((SIZE, SIZE), Image.Resampling.NEAREST)
    quality = int(rng.integers(*config.jpeg))
    with io.BytesIO() as encoded:
        canvas.save(encoded, format="JPEG", quality=quality)
        encoded.seek(0)
        with Image.open(encoded) as decoded:
            image = decoded.convert("RGB")
    pixels = np.asarray(support) > 0
    return InvisibleScene(
        image, pixels, cell_mask(pixels), message, embedder.family, regime, str(name),
        str(donor_name) if regime == "object" else None, quality, crop, resize,
    )
