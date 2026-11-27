"""Calibrate unknown-message TrustMark recovery and prepare the product artifact."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray
from PIL import Image

from murmell.provenance import DEFAULT_WEIGHTS
from murmell.trustmark import TrustMarkAdapter

from .invisible_data import make_invisible_scene

GATE_KEYS = ("full_gate", "full_threshold", "small_threshold", "box_gate", "snap_gate")


class EccEmbedder:
    family = "trustmark"
    bits = 61

    def __init__(self, adapter: TrustMarkAdapter, marked: bool) -> None:
        self.adapter = adapter
        self.marked = marked

    def embed(self, image: Image.Image, message: NDArray[np.uint8]) -> Image.Image:
        if not self.marked:
            return image.copy()
        return self.adapter.embed(image, self.adapter.payload(message))


def fit_gate(marked: list[float], clean: list[float]) -> tuple[float, int, int]:
    feasible = [float(t) for t in np.linspace(0, 1, 101) if not any(value >= t for value in clean)]
    if not feasible:
        return 1.01, 0, 0
    threshold = min(feasible, key=lambda t: (-sum(value >= t for value in marked), t))
    return threshold, sum(value >= threshold for value in marked), 0


def calibrate(adapter: TrustMarkAdapter) -> dict:
    positives, negatives = [], []
    for marked, values in ((True, positives), (False, negatives)):
        embedder = EccEmbedder(adapter, marked)
        for i in range(48):
            scene = make_invisible_scene(embedder, 202614, i)
            result = adapter.recover(scene.image)
            correct = result is not None and np.array_equal(result[0], adapter.payload(scene.message))
            if marked:
                values.append(result[1] if correct and result is not None else -1.0)
                if result is not None and not correct:
                    negatives.append(result[1])
            else:
                values.append(result[1] if result is not None else -1.0)
    threshold, positives_accepted, negatives_accepted = fit_gate(positives, negatives)
    return {"confidence": threshold, "positive_correct": positives_accepted,
            "negative_accepted": negatives_accepted, "marked_scenes": len(positives),
            "clean_scenes": len(positives), "negative_scores": len(negatives), "seed": 202614}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("stages", type=Path)
    parser.add_argument("--resolution", type=int, choices=(128, 256), required=True)
    parser.add_argument("--out", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    stages = json.loads(args.stages.read_text())
    variant = max(stages, key=lambda name: stages[name]["calibration_miou"])
    calibration = calibrate(TrustMarkAdapter())
    if not calibration["positive_correct"]:
        raise SystemExit("No correctly recovered TrustMark messages passed calibration.")
    gates = dict(zip(GATE_KEYS, stages[variant]["gates"]))
    gates["confidence"] = calibration["confidence"]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.checkpoint, args.out)
    digest = hashlib.sha256()
    with args.out.open("rb") as weights:
        for chunk in iter(lambda: weights.read(1024 * 1024), b""):
            digest.update(chunk)
    checksum = digest.hexdigest()
    card = {"resolution": args.resolution, "gates": gates, "product_calibration": calibration,
            "variant": variant, "sha256": checksum}
    args.out.with_suffix(".json").write_text(json.dumps(card, indent=2) + "\n")
    print(json.dumps(card, indent=2), flush=True)


if __name__ == "__main__":
    main()
