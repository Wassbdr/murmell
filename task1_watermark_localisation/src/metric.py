"""Official Task-1 metric (identical to kit/wmloc/utils.py miou_image/score; verified by tests/test_metric.py).
Per image: mean over classes {0,1} of |P_c & G_c| / |P_c | G_c|, IoU = 1 when the union is empty. Score = mean over images."""
import numpy as np


def miou_image(pred, gt) -> float:
    p = np.asarray(pred).astype(bool); g = np.asarray(gt).astype(bool)
    s = 0.0
    for pc, gc in ((p, g), (~p, ~g)):
        u = np.logical_or(pc, gc).sum()
        s += 1.0 if u == 0 else np.logical_and(pc, gc).sum() / u
    return s / 2


def miou_batch(preds, gts) -> np.ndarray:
    """Vectorised per-image mIoU for N x H x W binary arrays."""
    n = len(preds)
    p = np.asarray(preds).astype(bool).reshape(n, -1); g = np.asarray(gts).astype(bool).reshape(n, -1)
    out = np.zeros(n)
    for pc, gc in ((p, g), (~p, ~g)):
        u = (pc | gc).sum(1); i = (pc & gc).sum(1)
        out += np.where(u == 0, 1.0, i / np.maximum(u, 1))
    return out / 2


def score(preds, gts) -> float:
    return float(np.mean([miou_image(p, g) for p, g in zip(preds, gts)]))
