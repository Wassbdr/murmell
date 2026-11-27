"""CPU references for known-message localisation."""

import numpy as np
import torch
from numpy.typing import NDArray
from PIL import Image

from murmell.localize import Decoder

from .invisible_data import CELLS


def sliding_scores(
    decoder: Decoder,
    image: Image.Image,
    message: NDArray[np.uint8],
    windows: tuple[int, ...] = (24, 40, 64, 96),
    stride: int = 24,
    batch_size: int = 8,
) -> NDArray[np.float32]:
    """Broadcast bit agreement over each window and retain the strongest local evidence."""
    x_scale, y_scale = (length / CELLS for length in image.size)
    boxes = []
    for side in windows:
        offsets = list(range(0, CELLS - side, stride)) + [CELLS - side]
        boxes.extend((x, y, x + side, y + side) for y in offsets for x in offsets)
    scores = np.zeros((CELLS, CELLS), dtype=np.float32)
    target = torch.from_numpy(message).bool()
    for start in range(0, len(boxes), batch_size):
        chunk = boxes[start:start + batch_size]
        crops = [image.crop((round(x * x_scale), round(y * y_scale),
                             round(r * x_scale), round(b * y_scale))) for x, y, r, b in chunk]
        accuracy = ((decoder.logits(crops) > 0).cpu() == target).float().mean(dim=1).tolist()
        for (x, y, r, b), value in zip(chunk, accuracy):
            region = scores[y:b, x:r]
            np.maximum(region, value, out=region)
    return scores
