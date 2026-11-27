"""Score bundled and historical checkpoints on the fixed benchmark splits."""

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime
import torch
from predict import load, predict

from murmell import Detector
from src.bench import EVAL_SEEDS, calibrate, corpus, elapsed_ms, hardware, measure, measure_onnx, probabilities, tensors
from src.hard_watermarks import HARD, HELD_OUT


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--work", type=Path, default=Path("work/shipped"))
    parser.add_argument("--output", type=Path, default=Path("../benchmarks/shipped.json"))
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive.")
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(args.threads)
    args.work.mkdir(parents=True, exist_ok=True)
    scenes = {
        "hard": corpus(EVAL_SEEDS["hard"], 192, 128, HARD),
        "held-out": corpus(EVAL_SEEDS["held-out"], 192, 128, HELD_OUT),
    }
    report = {
        "cpu": hardware(), "threads": args.threads, "pytorch": torch.__version__, "onnxruntime": onnxruntime.__version__,
        "size": 128, "image_count_per_split": 192, "seeds": EVAL_SEEDS, "models": {},
    }
    for variant in ("round1", "contrast"):
        detector = Detector.load(variant)
        row = {
            "threshold": detector.threshold, "train_seed": 0,
            **{split: measure(probabilities(detector.model, tensors(data)[0]), data, detector.threshold)
               for split, data in scenes.items()},
        }
        if variant == "round1":
            row["pytorch_ms"] = elapsed_ms(detector.model, tensors(scenes["hard"])[0])
            row["onnx"] = measure_onnx(detector.model, args.work / "round1-0.onnx", detector.threshold,
                                       scenes, args.threads)
        report["models"][variant] = row

    legacy, card = load()
    validation = corpus(EVAL_SEEDS["validation"], 96, 128, HARD)
    validation_probs = np.stack([predict(legacy, card, scene.image)[0] for scene in validation])
    validation_threshold = calibrate(validation_probs, np.stack([scene.mask for scene in validation]))
    legacy_probs = {split: np.stack([predict(legacy, card, scene.image)[0] for scene in data])
                    for split, data in scenes.items()}
    for key, threshold in (("demo", card["threshold"]), ("demo_recalibrated", validation_threshold)):
        report["models"][key] = {
            "threshold": threshold,
            **{split: measure(legacy_probs[split], data, threshold) for split, data in scenes.items()},
        }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote {args.output} and {args.work / 'round1-0.onnx'}")


if __name__ == "__main__":
    main()
