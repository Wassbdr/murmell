"""Score cached references and fit the TrustMark post-processing chain."""

import argparse
import itertools
import json
from collections.abc import Iterator
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from PIL import Image

from murmell.localize import BoxMatch, bounding_box, message_agreement, search_box, select_mask
from murmell.trustmark import TrustMarkAdapter

from .invisible_bench import SPLITS, Corpus, evaluate, fit_threshold
from .invisible_data import SIZE, cell_mask
from .invisible_references import sliding_scores
from .metric import miou_batch

SCORED = ("calibration", "hard", "held-out")


def load_reference_data(root: Path) -> dict[str, Corpus]:
    corpora = {}
    for split in SCORED:
        seed, count = SPLITS[split]
        with np.load(root / f"trustmark-{split}-{seed}-{count}.npz") as data:
            corpora[split] = Corpus(data["images"], data["masks"], data["messages"])
    return corpora


def report_reference(root: Path, corpora: dict[str, Corpus], adapter: TrustMarkAdapter) -> dict:
    maps = {}
    for split, corpus in corpora.items():
        path = root / f"trustmark-sliding-{split}.npy"
        if not path.is_file():
            start = perf_counter()
            outputs = []
            cache = root / f"sliding-{split}"
            cache.mkdir(exist_ok=True)
            for index, (image, message) in enumerate(zip(corpus.images, corpus.messages)):
                scene_path = cache / f"{index:04d}.npy"
                if not scene_path.is_file():
                    pending = scene_path.with_suffix(".pending.npy")
                    np.save(pending, sliding_scores(adapter, Image.fromarray(image), message))
                    pending.replace(scene_path)
                outputs.append(np.load(scene_path))
                if (index + 1) % 12 == 0:
                    print(f"sliding {split}: {index + 1}/{len(corpus.images)}", flush=True)
            np.save(path, np.stack(outputs))
            print(f"sliding {split}: {perf_counter() - start:.1f} s", flush=True)
        maps[split] = np.load(path)
    threshold = fit_threshold(maps["calibration"], corpora["calibration"].masks)
    return {"threshold": threshold, "scores": evaluate(corpora, maps.__getitem__, threshold)}


def sam_predictor(weights: Path):
    from mobile_sam import SamPredictor, sam_model_registry

    return SamPredictor(sam_model_registry["vit_t"](checkpoint=str(weights)).cpu().eval())


def segment_proposals(predictor, image: np.ndarray, probability: np.ndarray) -> tuple[np.ndarray, ...]:
    extent = bounding_box(probability >= 0.5)
    if extent is None:
        return ()
    a, b, c, d = extent
    if (c - a) * (d - b) >= probability.size / 2:
        return ()
    box = np.array((a, b, c, d)) * (SIZE / probability.shape[0])
    predictor.set_image(image)
    masks, _, _ = predictor.predict(box=box, multimask_output=True)
    return tuple(cell_mask(mask) for mask in masks)


def proposals(
    adapter: TrustMarkAdapter, predictor, corpus: Corpus, probabilities: np.ndarray,
) -> Iterator[tuple[float, BoxMatch | None, tuple[np.ndarray, ...]]]:
    for image, message, prob in zip(corpus.images, corpus.messages, probabilities):
        pil = Image.fromarray(image)
        _, soft = message_agreement(adapter.logits([pil]), message)
        global_score = float(soft[0])
        box = search_box(adapter, pil, prob, message) if global_score < 0.6 else None
        segments = segment_proposals(predictor, image, prob) if predictor is not None and global_score < 0.6 else ()
        yield global_score, box, segments


def fit_and_score(root: Path, maps_path: Path, corpora: dict[str, Corpus], adapter: TrustMarkAdapter, predictor) -> dict:
    variants = {
        "regime": itertools.product((0.05, 0.15, 0.25, 0.35, 0.45, 0.55),
                                    (0.25, 0.4, 0.5, 0.6, 0.75), (0.25, 0.4, 0.5, 0.6, 0.75), (1.01,), (1.01,)),
        "boxes": itertools.product((0.05, 0.15, 0.25, 0.35, 0.45, 0.55),
                                   (0.25, 0.4, 0.5, 0.6, 0.75), (0.25, 0.4, 0.5, 0.6, 0.75),
                                   (0.75, 0.8, 0.85, 0.9), (1.01,)),
        "snap": itertools.product((0.05, 0.15, 0.25, 0.35, 0.45, 0.55),
                                  (0.25, 0.4, 0.5, 0.6, 0.75), (0.25, 0.4, 0.5, 0.6, 0.75),
                                  (0.75, 0.8, 0.85, 0.9), (0.35, 0.5, 0.65, 0.8, 1.01)),
    }
    with np.load(maps_path) as stored:
        maps = {split: stored[split] for split in SCORED}
    artifacts = {}

    def make_proposals(split: str) -> None:
        start = perf_counter()
        artifacts[split] = list(proposals(adapter, predictor, corpora[split], maps[split]))
        print(f"proposals {split}: {perf_counter() - start:.1f} s", flush=True)

    def masks(split: str, config: tuple[float, ...]) -> np.ndarray:
        full_gate, full_threshold, small_threshold, box_gate, snap_gate = config
        return np.stack([
            select_mask(prob, soft, box, segments, full_gate=full_gate, full_threshold=full_threshold,
                        small_threshold=small_threshold, box_gate=box_gate, snap_gate=snap_gate)
            for prob, (soft, box, segments) in zip(maps[split], artifacts[split])
        ])

    results = {}
    make_proposals("calibration")
    target = corpora["calibration"].masks
    fitted = {}
    for variant, settings in variants.items():
        if variant == "snap" and predictor is None:
            continue
        fitted[variant] = max(settings, key=lambda c: miou_batch(masks("calibration", c), target).mean())
    for split in ("hard", "held-out"):
        make_proposals(split)
    for variant, best in fitted.items():
        results[variant] = {"gates": best, "calibration_miou": float(miou_batch(masks("calibration", best), target).mean()),
                            "scores": evaluate(corpora, lambda split: masks(split, best), 0.5)}
        if variant == "snap":
            without_snap = (*best[:-1], 1.01)
            results[variant]["snap_changed"] = {
                split: int((masks(split, best) != masks(split, without_snap)).any(axis=(1, 2)).sum())
                for split in SCORED
            }
        print(f"{maps_path.stem} {variant}: {results[variant]['scores']['hard']['miou']:.4f} hard, "
              f"{results[variant]['scores']['held-out']['miou']:.4f} held-out", flush=True)
    suffix = "-stages" if predictor is not None else "-no-sam-stages"
    root.joinpath(f"{maps_path.stem}{suffix}.json").write_text(json.dumps(results, indent=2) + "\n")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--sliding", action="store_true")
    parser.add_argument("--maps", type=Path, nargs="*")
    parser.add_argument("--sam", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    corpora = load_reference_data(args.work)
    adapter = TrustMarkAdapter()
    if args.sliding:
        report = report_reference(args.work, corpora, adapter)
        (args.work / "trustmark-sliding.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report), flush=True)
    if args.maps:
        predictor = sam_predictor(args.sam) if args.sam else None
        for path in args.maps:
            fit_and_score(args.work, path, corpora, adapter, predictor)


if __name__ == "__main__":
    main()
