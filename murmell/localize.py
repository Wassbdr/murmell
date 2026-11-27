"""Regime detection and bounded crop search for known watermark messages."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import torch
from numpy.typing import NDArray
from PIL import Image
from torch import Tensor

CELLS = 128
Box = tuple[int, int, int, int]


class Decoder(Protocol):
    def logits(self, images: Sequence[Image.Image]) -> Tensor: ...


@dataclass(frozen=True)
class BoxMatch:
    bounds: Box
    accuracy: float
    agreement: float

    def mask(self) -> NDArray[np.bool_]:
        x0, y0, x1, y1 = self.bounds
        output = np.zeros((CELLS, CELLS), dtype=bool)
        output[y0:y1, x0:x1] = True
        return output


def message_agreement(logits: Tensor, message: NDArray[np.uint8]) -> tuple[Tensor, Tensor]:
    bits = torch.as_tensor(message).to(logits.device)
    accuracy = ((logits > 0) == bits.bool()).float().mean(-1)
    agreement = (logits.tanh() * (bits.float() * 2 - 1)).mean(-1)
    return accuracy, agreement


def bounding_box(mask: NDArray[np.bool_]) -> Box | None:
    y, x = np.nonzero(mask)
    if not len(x):
        return None
    return int(x.min()), int(y.min()), int(x.max() + 1), int(y.max() + 1)


def padded_box(box: Box, padding: int) -> Box | None:
    x0, y0, x1, y1 = box
    result = max(0, x0 - padding), max(0, y0 - padding), min(CELLS, x1 + padding), min(CELLS, y1 + padding)
    return result if result[2] - result[0] >= 12 and result[3] - result[1] >= 12 else None


def search_box(
    decoder: Decoder, image: Image.Image, probability: NDArray[np.float32], message: NDArray[np.uint8],
    budget: int = 24, batch_size: int = 8,
) -> BoxMatch | None:
    scale_x, scale_y = (side / CELLS for side in image.size)
    probes: list[Box] = []
    for threshold in (0.5, 0.7, 0.3):
        extent = bounding_box(probability >= threshold)
        if extent is not None and (extent[2] - extent[0]) * (extent[3] - extent[1]) <= CELLS ** 2 * 0.65:
            probes.extend(box for p in (0, 3, -3, 7) if (box := padded_box(extent, p)) is not None)
    y, x = np.unravel_index(np.argmax(probability), probability.shape)
    for radius in (12, 18, 24, 32, 42):
        box = padded_box((int(x), int(y), int(x) + 1, int(y) + 1), radius)
        if box is not None:
            probes.append(box)
    results: list[BoxMatch] = []

    def score(boxes: list[Box]) -> None:
        for start in range(0, len(boxes), batch_size):
            chunk = boxes[start:start + batch_size]
            patches = [image.crop((round(a * scale_x), round(b * scale_y),
                                   round(c * scale_x), round(d * scale_y))) for a, b, c, d in chunk]
            accuracy, agreement = message_agreement(decoder.logits(patches), message)
            results.extend(BoxMatch(box, acc, agree) for box, acc, agree in
                           zip(chunk, accuracy.tolist(), agreement.tolist()))

    first = list(dict.fromkeys(probes))[:budget]
    score(first)
    if not results:
        return None

    def rank(match: BoxMatch) -> tuple[float, float]:
        return match.accuracy, match.agreement

    best = max(results, key=rank)
    neighbors = []
    for step in (4, -4, 2, -2):
        for index in range(4):
            coords = list(best.bounds)
            coords[index] = max(0, min(CELLS, coords[index] + step))
            a, b, c, d = coords
            if c - a >= 12 and d - b >= 12 and (a, b, c, d) not in first:
                neighbors.append((a, b, c, d))
    score(list(dict.fromkeys(neighbors))[:budget - len(first)])
    return max(results, key=rank)


def overlap(a: NDArray[np.bool_], b: NDArray[np.bool_]) -> float:
    union = np.count_nonzero(a | b)
    return float(np.count_nonzero(a & b) / union) if union else 1.0


def select_mask(
    probability: NDArray[np.float32], global_agreement: float, box: BoxMatch | None,
    segments: Sequence[NDArray[np.bool_]], *, full_gate: float, full_threshold: float,
    small_threshold: float, box_gate: float, snap_gate: float,
) -> NDArray[np.bool_]:
    if global_agreement >= full_gate:
        return probability >= full_threshold
    if box is not None and box.accuracy >= box_gate:
        return box.mask()
    mask = probability >= small_threshold
    if segments and mask.any():
        best = max(segments, key=lambda segment: overlap(mask, segment))
        if overlap(mask, best) >= snap_gate:
            return best
    return mask
