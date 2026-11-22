"""Lossless data access. Canonical ORDER = order of release/<split>/metadata.json (used by every OOF/test .npy).
Shared map format: work/oof/<exp>.npy  = float16 [510,128,128] (train, metadata order, out-of-fold probabilities)
                   work/test/<exp>.npy = float16 [1020,128,128] (test, metadata order)."""
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
REL = ROOT / "release"
METHODS = ("pixelseal", "trustmark", "stegastamp")


@lru_cache(None)
def meta(split="train"):
    return json.loads((REL / split / "metadata.json").read_text())


def names(split="train"):
    return [m["image_name"] for m in meta(split)]


def methods(split="train"):
    return np.array([m["method"] for m in meta(split)])


def regime():
    """train only: True = full regime (whole image watermarked, clean holes)."""
    return np.array([bool(m["full_watermark"]) for m in meta("train")])


def message(split, i):
    return np.array([int(c) for c in meta(split)[i]["message"]], np.uint8)


@lru_cache(None)
def masks128():
    z = np.load(REL / "train" / "masks_128.npz")
    d = dict(zip(z["names"].tolist(), z["masks"]))
    return np.stack([d[n] for n in names("train")]).astype(np.uint8)


def image(split, i_or_name) -> np.ndarray:
    n = i_or_name if isinstance(i_or_name, str) else names(split)[i_or_name]
    return np.asarray(Image.open(REL / split / "images" / f"{n}.png").convert("RGB"))


def mask512(i_or_name) -> np.ndarray:
    n = i_or_name if isinstance(i_or_name, str) else names("train")[i_or_name]
    return (np.asarray(Image.open(REL / "train" / "masks" / f"{n}.png")) > 127).astype(np.uint8)


def folds():
    """fold id per train image (metadata order) from work/folds.json."""
    d = json.loads((ROOT / "work" / "folds.json").read_text())
    return np.array([d["fold"][n] for n in names("train")])
