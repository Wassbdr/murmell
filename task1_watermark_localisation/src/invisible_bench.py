"""Train and score public invisible-watermark baselines on CPU."""

import argparse
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from numpy.typing import NDArray
from PIL import Image
from torch import Tensor, nn
from torch.nn import functional as F

from murmell.invisible_model import ResNetUnet
from murmell.trustmark import TrustMarkAdapter

from .invisible_data import CELLS, HELD_OUT, MAIN, REGIMES, SIZE, Embedder, make_invisible_scene
from .metric import miou_batch

SPLITS = {"train": (202610, 1500), "calibration": (202611, 48),
          "hard": (202612, 60), "held-out": (202613, 60)}


@dataclass
class Corpus:
    images: NDArray[np.uint8]
    masks: NDArray[np.bool_]
    messages: NDArray[np.uint8]


def load_corpora(root: Path, embedder: Embedder, splits: dict[str, tuple[int, int]] = SPLITS) -> dict[str, Corpus]:
    corpora = {}
    for name, (seed, count) in splits.items():
        path = root / f"{embedder.family}-{name}-{seed}-{count}.npz"
        if not path.is_file():
            start = perf_counter()
            images = np.empty((count, SIZE, SIZE, 3), dtype=np.uint8)
            masks = np.empty((count, CELLS, CELLS), dtype=bool)
            messages = np.empty((count, embedder.bits), dtype=np.uint8)
            config = HELD_OUT if name == "held-out" else MAIN
            for index in range(count):
                scene = make_invisible_scene(embedder, seed, index, config)
                images[index], masks[index], messages[index] = scene.image, scene.mask, scene.message
                if (index + 1) % 150 == 0:
                    print(f"{name}: generated {index + 1}/{count}", flush=True)
            pending = path.with_suffix(".pending.npz")
            np.savez(pending, images=images, masks=masks, messages=messages)
            pending.replace(path)
            print(f"{name}: generated {count} scenes in {perf_counter() - start:.1f} s", flush=True)
        with np.load(path) as saved:
            corpora[name] = Corpus(saved["images"], saved["masks"], saved["messages"])
    return corpora


def fit_threshold(probabilities: NDArray, masks: NDArray) -> float:
    return float(max(np.linspace(0.05, 0.95, 37), key=lambda t: miou_batch(probabilities >= t, masks).mean()))


def summarise(prediction: NDArray, masks: NDArray) -> dict[str, float]:
    mask = np.asarray(masks, dtype=bool)
    pred = np.asarray(prediction, dtype=bool)
    marked = mask.any(axis=(1, 2))
    intersection = (pred & mask).sum(axis=(1, 2))
    union = (pred | mask).sum(axis=(1, 2))
    return {
        "miou": float(miou_batch(pred, mask).mean()),
        "foreground_iou": float((intersection[marked] / union[marked]).mean()) if marked.any() else 0.0,
    }


def input_tensor(corpus: Corpus, resolution: int) -> Tensor:
    images = np.stack([
        np.asarray(Image.fromarray(image).resize((resolution, resolution), Image.Resampling.BILINEAR))
        for image in corpus.images
    ])
    return torch.from_numpy(images).permute(0, 3, 1, 2)


def preprocess(images: Tensor, initialization: str) -> Tensor:
    x = images.float() / 255
    if initialization == "decoder":
        return x * 2 - 1
    mean = x.new_tensor((0.485, 0.456, 0.406))[None, :, None, None]
    std = x.new_tensor((0.229, 0.224, 0.225))[None, :, None, None]
    return (x - mean) / std


def train(
    model: nn.Module, images: Tensor, masks: Tensor, *, seed: int,
    updates: int, batch_size: int, initialization: str,
) -> float:
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003)
    rng = torch.Generator().manual_seed(seed)
    start = perf_counter()
    for step in range(updates):
        batch = torch.randint(len(images), (batch_size,), generator=rng)
        targets = masks[batch].float()
        logits = model(preprocess(images[batch], initialization))
        probability = logits.sigmoid().flatten(1)
        truth = targets.flatten(1)
        dice = (2 * (probability * truth).sum(1) + 1) / (probability.sum(1) + truth.sum(1) + 1)
        loss = F.binary_cross_entropy_with_logits(logits, targets) + 1 - dice.mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if (step + 1) % 200 == 0:
            print(f"step {step + 1}/{updates}: loss={loss.item():.4f}, elapsed={perf_counter() - start:.1f} s",
                  flush=True)
    return perf_counter() - start


@torch.inference_mode()
def predict(model: nn.Module, images: Tensor, initialize: str, batch_size: int = 4) -> NDArray[np.float32]:
    model.eval()
    return torch.cat([model(preprocess(batch, initialize)).sigmoid() for batch in images.split(batch_size)]).numpy()


def evaluate(
    corpora: dict[str, Corpus], transform: Callable[[str], NDArray], threshold: float,
) -> dict[str, dict]:
    out = {}
    for split in ("hard", "held-out"):
        pred = transform(split) >= threshold
        masks = corpora[split].masks
        out[split] = summarise(pred, masks)
        out[split]["regimes"] = {name: summarise(pred[i::3], masks[i::3]) for i, name in enumerate(REGIMES)}
    return out


def run(args: argparse.Namespace) -> None:
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    args.work.mkdir(parents=True, exist_ok=True)
    adapter = TrustMarkAdapter()
    corpora = load_corpora(args.work, adapter)
    model = ResNetUnet(adapter.backbone if args.initialize == "decoder" else None)
    trained = train(
        model, input_tensor(corpora["train"], args.resolution), torch.from_numpy(corpora["train"].masks),
        seed=args.seed, updates=args.updates, batch_size=args.batch_size, initialization=args.initialize,
    )
    stem = args.work / f"trustmark-{args.initialize}-{args.resolution}-{args.seed}-{args.updates}"
    torch.save(model.state_dict(), stem.with_suffix(".pt"))
    scores = {
        split: predict(model, input_tensor(corpus, args.resolution), args.initialize)
        for split, corpus in corpora.items() if split != "train"
    }
    threshold = fit_threshold(scores["calibration"], corpora["calibration"].masks)
    report = {
        "seed": args.seed, "updates": args.updates, "resolution": args.resolution, "threads": args.threads,
        "batch_size": args.batch_size, "training_seconds": trained, "threshold": threshold,
        args.initialize: evaluate(corpora, scores.__getitem__, threshold),
        "all-empty": evaluate(corpora, lambda split: np.zeros_like(corpora[split].masks), 0.5),
        "all-full": evaluate(corpora, lambda split: np.ones_like(corpora[split].masks), 0.5),
    }
    np.savez(stem.with_suffix(".npz"), **scores)
    stem.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"{args.initialize}: threshold={threshold:.3f}, trained in {trained:.1f} s")
    for name in (args.initialize, "all-empty", "all-full"):
        for split, metrics in report[name].items():
            print(f"{name} {split}: mIoU={metrics['miou']:.4f} foreground IoU={metrics['foreground_iou']:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, default=Path("work/invisible"))
    parser.add_argument("--initialize", choices=("imagenet", "decoder"), default="imagenet")
    parser.add_argument("--resolution", type=int, choices=(128, 256), default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--updates", type=int, default=1600)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--threads", type=int, default=8)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
