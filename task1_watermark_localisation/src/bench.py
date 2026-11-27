"""Train and compare visible-watermark mask detectors on a CPU."""

import argparse
import hashlib
import json
import os
import platform
from pathlib import Path
from time import perf_counter

import numpy as np
import onnxruntime as ort
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from murmell.models import ContrastUnet, Decomposition, SelfCalibrated, SplitThenRefine, build_reference
from murmell.onnx import export_mask_model
from src.hard_watermarks import HARD, HELD_OUT, Scene, SceneConfig, make_scene
from src.metric import miou_batch

ROOT = Path(__file__).resolve().parents[2]
REFERENCES = ("unet", "round1", "slbr", "wdnet", "split")
MODELS = (*REFERENCES, "contrast", "residual")
LABELS = {
    "unet": "Plain RGB U-Net",
    "round1": "Murmell round 1 (RGB + high-pass U-Net)",
    "slbr": "SLBR-style self-calibrated mask",
    "wdnet": "WDNet-style decomposition",
    "split": "Split then Refine-style mask",
    "contrast": "Murmell local contrast",
    "residual": "Murmell residual supervision",
}
ITERATIONS = {
    "contrast": (
        "generalize local contrast across background textures",
        "Feed two local residual scales and texture-normalized contrast into the full-resolution U-Net stem. "
        "Recompose some training marks over colour-jittered multiscale textures without changing alpha. "
        "Use a focal penalty and include foreground IoU in threshold calibration.",
    ),
    "residual": (
        "supervise the watermark's contribution after texture and colour jitter",
        "Add a dilated detail block and predict the signed, colour-transformed difference introduced by the watermark. "
        "Feed its magnitude into the mask head and mix full images with foreground-biased crops. "
        "Keep the same foreground-aware threshold calibration.",
    ),
}
EVAL_SEEDS = {"validation": 12301, "hard": 12302, "held-out": 12303}
SLICES = ("all", "thin text", "low opacity", "textured", "logo", "tiled", "clean")


def corpus(seed: int, count: int, size: int, config: SceneConfig = HARD) -> list[Scene]:
    return [make_scene(seed, index, size, config) for index in range(count)]


def tensors(scenes: list[Scene]) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    x = torch.from_numpy(np.stack([s.image for s in scenes])).permute(0, 3, 1, 2)
    y = torch.from_numpy(np.stack([s.mask for s in scenes]))[:, None]
    b = torch.from_numpy(np.stack([s.background for s in scenes])).permute(0, 3, 1, 2)
    a = torch.from_numpy(np.stack([s.alpha for s in scenes]))[:, None]
    ink = torch.from_numpy(np.stack([s.ink for s in scenes]))[:, :, None, None]
    return x, y, b, a, ink


def mask_loss(logits: Tensor, mask: Tensor) -> Tensor:
    prob = logits.sigmoid()
    overlap = (prob * mask).sum((2, 3))
    dice = (2 * overlap + 1) / (prob.sum((2, 3)) + mask.sum((2, 3)) + 1)
    positive_weight = ((mask.numel() - mask.sum()) / mask.sum().clamp_min(1)).clamp(1, 20)
    return F.binary_cross_entropy_with_logits(logits, mask, pos_weight=positive_weight) + 1 - dice.mean()


def foreground_crops(
    image: Tensor, mask: Tensor, background: Tensor, opacities: Tensor
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    height, width = image.shape[-2:]
    crop_h, crop_w = height // 2, width // 2
    crops = []
    for item in range(len(image)):
        points = torch.nonzero(mask[item, 0])
        if len(points) and torch.rand(()) < 0.8:
            y, x = map(int, points[torch.randint(len(points), ())])
            y = min(max(y - int(torch.randint(crop_h, ())), 0), height - crop_h)
            x = min(max(x - int(torch.randint(crop_w, ())), 0), width - crop_w)
        else:
            y, x = int(torch.randint(height - crop_h + 1, ())), int(torch.randint(width - crop_w + 1, ()))
        crops.append((slice(y, y + crop_h), slice(x, x + crop_w)))
    return tuple(torch.stack([t[i, :, y, x] for i, (y, x) in enumerate(crops)])
                 for t in (image, mask, background, opacities))


def composited_jitter(
    image: Tensor, background: Tensor, alpha: Tensor, ink: Tensor, *, measure_residual: bool
) -> tuple[Tensor, Tensor | None]:
    count, _, height, width = image.shape
    planes = (count, 1, 1, 1)
    colors = (count, 3, 1, 1)
    coarse = torch.randn(count, 1, height // 8, width // 8)
    fine = torch.randn(count, 1, height // 2, width // 2)
    texture = (
        F.interpolate(coarse, size=(height, width), mode="bilinear", align_corners=False) * 0.7
        + F.interpolate(fine, size=(height, width), mode="bilinear", align_corners=False) * 0.3
    )
    clean = background / 255
    background = clean
    mix = torch.empty(planes).uniform_(0, 0.4)
    background = background * (1 - mix) + background[torch.randperm(count)] * mix
    strength = torch.empty(planes).uniform_(0, 0.28)
    background = (background + strength * texture).clamp(0, 1)
    matte = alpha / 255
    augmented = background * (1 - matte) + (ink / 255) * matte
    recompose = torch.rand(planes) < 0.65
    image = torch.where(recompose, augmented, image)
    gain = torch.empty(colors).uniform_(0.75, 1.25)
    offset = torch.empty(colors).uniform_(-0.10, 0.10)
    gamma = torch.empty(planes).uniform_(0.75, 1.25)
    output = (image * gain + offset).clamp(0, 1).pow(gamma)
    if not measure_residual:
        return output, None
    clean = torch.where(recompose, background, clean)
    composite = clean * (1 - matte) + (ink / 255) * matte
    after = (composite * gain + offset).clamp(0, 1).pow(gamma)
    before = (clean * gain + offset).clamp(0, 1).pow(gamma)
    return output, after - before


def train_loss(model: nn.Module, batch: tuple[Tensor, ...]) -> Tensor:
    x, y, background, alpha, ink = batch
    x, y = x.float() / 255, y.float()
    if isinstance(model, SelfCalibrated):
        logits, masks = model.outputs(x)
        return mask_loss(logits, y) + sum(mask_loss(m, y) for m in masks) / 6
    if isinstance(model, Decomposition):
        logits, coarse, matte, color = model.outputs(x)
        ink_loss = ((color - ink / 255).abs() * y).sum() / (3 * y.sum().clamp_min(1))
        return mask_loss(logits, y) + 0.5 * (mask_loss(coarse, y) + F.l1_loss(matte, alpha / 255) + ink_loss)
    if isinstance(model, SplitThenRefine):
        logits, coarse, restored = model.outputs(x)
        return mask_loss(logits, y) + 0.5 * (mask_loss(coarse, y) + F.l1_loss(restored, background / 255))
    if isinstance(model, ContrastUnet):
        if model.residual is not None:
            if torch.rand(()) < 0.35:
                x, y, background, alpha = foreground_crops(x, y, background, alpha)
            x, target = composited_jitter(x, background, alpha, ink, measure_residual=True)
            logits, residual = model.outputs(x)
            if residual is None or target is None:
                raise RuntimeError("Expected both a compositing residual prediction and target.")
            error = (residual - 4 * target).abs()
            weights = 1 + 12 * (alpha > 0)
            return mask_loss(logits, y) + (error * weights).sum() / (3 * weights.sum())
        x, _ = composited_jitter(x, background, alpha, ink, measure_residual=False)
        logits = model(x)
        error = (logits.sigmoid() - y).abs()
        focal = (F.binary_cross_entropy_with_logits(logits, y, reduction="none") * error.square()).mean()
        return mask_loss(logits, y) + focal
    return mask_loss(model(x), y)


def build_model(name: str, size: int) -> nn.Module:
    model = ContrastUnet(residual=name == "residual") if name in ITERATIONS else build_reference(name, size)
    return model.to(memory_format=torch.channels_last)


def train(name: str, seed: int, size: int, data: tuple[Tensor, ...], batch_size: int) -> nn.Module:
    torch.manual_seed(seed)
    model = build_model(name, size)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)
    batches = range(0, len(data[0]), batch_size)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(batches), eta_min=0.0003)
    start_time = perf_counter()
    for step, start in enumerate(batches, 1):
        batch = tuple(t[start : start + batch_size] for t in data)
        optimizer.zero_grad(set_to_none=True)
        loss = train_loss(model, batch)
        loss.backward()
        optimizer.step()
        schedule.step()
        if step % 200 == 0:
            print(f"{name} seed={seed} step={step}/{len(batches)} loss={loss.item():.4f} "
                  f"elapsed={perf_counter() - start_time:.0f} s", flush=True)
    return model.eval()


@torch.inference_mode()
def probabilities(model: nn.Module, images: Tensor) -> np.ndarray:
    return np.concatenate([model(batch.float() / 255)[:, 0].sigmoid().numpy() for batch in images.split(16)])


def calibrate(probs: np.ndarray, masks: np.ndarray, foreground_weight: float = 0.0) -> float:
    thresholds = np.linspace(0.05, 0.95, 37)

    def objective(threshold: float) -> float:
        pred = probs >= threshold
        miou = miou_batch(pred, masks).mean()
        if not foreground_weight:
            return float(miou)
        foreground = foreground_iou(pred, masks)
        return float(miou + foreground_weight * (foreground or 0))

    return float(max(thresholds, key=objective))


def measure(probs: np.ndarray, scenes: list[Scene], threshold: float) -> dict[str, float | None]:
    pred = probs >= threshold
    masks = np.stack([s.mask for s in scenes])
    ious = miou_batch(pred, masks)
    scores = {}
    for key, idx in slice_indexes(scenes).items():
        scores[key] = float(ious[idx].mean()) if idx.any() else None
        scores["foreground" if key == "all" else f"{key}_foreground"] = foreground_iou(pred[idx], masks[idx])
    return scores


def foreground_iou(pred: np.ndarray, masks: np.ndarray) -> float | None:
    """Macro foreground IoU over images with at least one true foreground pixel."""
    marked = masks.any(axis=(1, 2))
    if not marked.any():
        return None
    target, prediction = masks[marked], pred[marked]
    intersection = (target & prediction).sum(axis=(1, 2))
    union = (target | prediction).sum(axis=(1, 2))
    return float((intersection / union).mean())


def slice_indexes(scenes: list[Scene]) -> dict[str, np.ndarray]:
    return {
        "all": np.ones(len(scenes), dtype=bool),
        "thin text": np.array([s.kind == "text" for s in scenes]),
        "low opacity": np.array([0 < s.opacity < 0.22 for s in scenes]),
        "textured": np.array([s.textured for s in scenes]),
        **{kind: np.array([s.kind == kind for s in scenes]) for kind in ("logo", "tiled", "clean")},
    }


def worst_cases(probs: np.ndarray, scenes: list[Scene], threshold: float) -> dict[str, dict]:
    predicted = probs >= threshold
    masks = np.stack([s.mask for s in scenes])
    scores = miou_batch(predicted, masks)
    groups = slice_indexes(scenes)
    output = {}
    for key in ("thin text", "low opacity", "textured", "clean"):
        subset = groups[key] if key == "clean" else groups[key] & masks.any((1, 2))
        indexes = np.flatnonzero(subset)
        if len(indexes):
            index = int(indexes[np.argmin(scores[indexes])])
            output[key] = {
                "index": index,
                "miou": float(scores[index]),
                "true_pixels": int(masks[index].sum()),
                "predicted_pixels": int(predicted[index].sum()),
                "overlap_pixels": int((masks[index] & predicted[index]).sum()),
                "opacity": scenes[index].opacity,
                "photo": scenes[index].photo,
            }
    return output


@torch.inference_mode()
def elapsed_ms(model: nn.Module, images: Tensor) -> float:
    inputs = images[:32].float() / 255
    for image in inputs[:8]:
        model(image[None]).sigmoid()
    times = []
    for _ in range(3):
        start = perf_counter()
        for image in inputs:
            model(image[None]).sigmoid()
        times.append((perf_counter() - start) * 1_000 / len(inputs))
    return float(np.median(times))


def measure_onnx(
    model: nn.Module,
    path: Path,
    threshold: float,
    scenes: dict[str, list[Scene]],
    threads: int,
) -> dict[str, float | int | None]:
    export_mask_model(model, path, scenes["hard"][0].image.shape[0])
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])

    def run(images: np.ndarray) -> np.ndarray:
        return session.run(["probability"], {"rgb": images})[0][:, 0]

    row = {"max_abs_error": 0.0, "mask_disagreements": 0}
    for split in ("hard", "held-out"):
        images = tensors(scenes[split])[0]
        values = np.concatenate([run(batch.float().numpy() / 255) for batch in images.split(16)])
        expected = probabilities(model, images)
        row["max_abs_error"] = max(row["max_abs_error"], float(np.abs(values - expected).max()))
        row["mask_disagreements"] += int(np.count_nonzero((values >= threshold) != (expected >= threshold)))
        scores = measure(values, scenes[split], threshold)
        for metric_key in ("all", "foreground"):
            row[f"{split}_{metric_key}"] = scores[metric_key]
    samples = np.ascontiguousarray(tensors(scenes["hard"][:32])[0].float().numpy() / 255)
    for image in samples[:8]:
        run(image[None])
    times = []
    for _ in range(3):
        start = perf_counter()
        for image in samples:
            run(image[None])
        times.append((perf_counter() - start) * 1_000 / len(samples))
    row["ms"] = float(np.median(times))
    return row


def weight_hash(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for key, value in model.state_dict().items():
        digest.update(key.encode())
        digest.update(value.contiguous().numpy().tobytes())
    return digest.hexdigest()


def hardware() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                return line.partition(":")[2].strip()
    return platform.processor() or platform.machine()


def mean(rows: list[dict], split: str, key: str) -> float | None:
    values = [row[split][key] for row in rows if row[split][key] is not None]
    return float(np.mean(values)) if values else None


def metric(value: float | None) -> str:
    return f"{value:.4f}" if value is not None else "n/a"


def render_report(
    results: dict[str, list[dict]], args: argparse.Namespace, empty: dict[str, dict[str, float | None]]
) -> str:
    rows = sorted(results, key=lambda name: mean(results[name], "hard", "all") or 0, reverse=True)
    reference_rows = [name for name in rows if name in REFERENCES]
    lines = [
        "# Visible-watermark localisation benchmark",
        "",
        "Regenerate with `make bench`. See [data and references](README.md).",
        "",
        f"CPU: {hardware()}, {args.threads} PyTorch threads, PyTorch {torch.__version__}, Python {platform.python_version()}.",
        f"Training seeds: {', '.join(map(str, args.seeds))}; each run starts from random weights and sees "
        f"{args.steps} batches of {args.batch} images at {args.size} px (one pass over a seed-specific corpus, "
        f"seeds {', '.join(str(20_000 + seed) for seed in args.seeds)}).",
        f"Validation: {args.validation} images, seed {EVAL_SEEDS['validation']}; hard test: {args.eval} images, "
        f"seed {EVAL_SEEDS['hard']}; held-out test: {args.eval} images, seed {EVAL_SEEDS['held-out']}.",
        "Each model/seed uses one global threshold chosen on validation only. The Murmell recipes maximize "
        "validation mIoU plus 0.2 times foreground IoU; the references maximize validation mIoU. "
        "mIoU is the mean of foreground "
        "and background IoU per image, averaged across images, then across training seeds.",
        "Foreground IoU is averaged only over images with a nonempty target, without giving credit for empty masks.",
        f"CPU latency is batch 1, warmed up, median of three passes over {min(args.eval, 32)} images per seed, "
        "then averaged across seeds. It includes the model and sigmoid but excludes I/O, input conversion and thresholding.",
        "",
        "## Iteration 0: references",
        "",
        "| Model | Hard mIoU | Hard foreground IoU | Held-out mIoU | Held-out foreground IoU | CPU ms/image |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in reference_rows:
        seeds = results[name]
        scores = [metric(mean(seeds, split, key)) for split in ("hard", "held-out") for key in ("all", "foreground")]
        lines.append(f"| {LABELS[name]} | " + " | ".join(scores) + f" | {np.mean([s['ms'] for s in seeds]):.2f} |")
    scores = [metric(empty[split][key]) for split in ("hard", "held-out") for key in ("all", "foreground")]
    lines.append("| All-empty mask | " + " | ".join(scores) + " | n/a |")
    previous: list[dict] = []
    for number, (name, (title, description)) in enumerate(ITERATIONS.items(), 1):
        if name not in results:
            continue
        current = results[name]
        lines.extend(["", f"## Iteration {number}: {title}", "", description,
                      "", "| Model | Hard mIoU (foreground IoU) | Δ mIoU vs best reference | Δ mIoU vs previous | "
                      "Held-out mIoU (foreground IoU) | Δ mIoU vs best reference | Δ mIoU vs previous | CPU ms/image |",
                      "|---|---:|---:|---:|---:|---:|---:|---:|"])
        row = [LABELS[name]]
        for split in ("hard", "held-out"):
            score = mean(current, split, "all")
            best = max([empty[split]["all"], *(mean(results[ref], split, "all") for ref in reference_rows)])
            prior = mean(previous, split, "all") if previous else None
            row.extend((f"{metric(score)} ({metric(mean(current, split, 'foreground'))})",
                        f"{score - best:+.4f}", f"{score - prior:+.4f}" if prior is not None else "n/a"))
        row.append(f"{np.mean([s['ms'] for s in current]):.2f}")
        lines.append("| " + " | ".join(row) + " |")
        previous = current
    for split in ("hard", "held-out"):
        lines.extend(["", f"## {split.capitalize()} breakdown", "",
                      "Each cell reports mIoU (foreground IoU).", "",
                      "| Model | " + " | ".join(SLICES[1:]) + " |",
                      "|---|" + "---:|" * (len(SLICES) - 1)])
        for name in rows:
            lines.append(f"| {LABELS[name]} | " +
                         " | ".join(f"{metric(mean(results[name], split, key))} "
                                    f"({metric(mean(results[name], split, f'{key}_foreground'))})"
                                    for key in SLICES[1:]) + " |")
        lines.append("| All-empty mask | " +
                     " | ".join(f"{metric(empty[split][key])} ({metric(empty[split][f'{key}_foreground'])})"
                                for key in SLICES[1:]) + " |")
    lines.extend(["", "Thin text: eroded single-instance text. Low opacity: 0 < opacity < 0.22. "
                  "Textured: photo crops. Slices overlap; all marked held-out images are low opacity."])
    if all("failures" in run for group in results.values() for run in group):
        lines.extend(["", "## Failure cases", "",
                      "Worst example by slice, for the first training seed. Regenerate a scene by passing "
                      "its test seed and scene index to `src.hard_watermarks.make_scene`.", "",
                      "| Model | Split | Slice | Index | mIoU | Opacity | Photo | True pixels | Predicted pixels | Overlap |",
                      "|---|---|---|---:|---:|---:|---|---:|---:|---:|"])
        for name in dict.fromkeys(reference_rows[:1] + [n for n in rows if n not in REFERENCES]):
            for split, cases in results[name][0]["failures"].items():
                for label, case in cases.items():
                    lines.append(
                        f"| {LABELS[name]} | {split} | {label} | {case['index']} | {case['miou']:.4f} | "
                        f"{case['opacity']:.3f} | {case['photo'] or 'procedural'} | {case['true_pixels']} | "
                        f"{case['predicted_pixels']} | {case['overlap_pixels']} |"
                    )
    candidates = [name for name in ITERATIONS if name in results]
    if candidates:
        lines.extend(["", f"## ONNX CPU ({ort.__version__})", "",
                      "| Model | Runtime | Hard mIoU (foreground IoU) | Held-out mIoU (foreground IoU) | CPU ms/image |",
                      "|---|---|---:|---:|---:|"])
        onnx_rows = []
        for name in candidates:
            current = results[name]
            exported = [run["onnx"] for run in current]
            onnx_rows.extend(exported)
            torch_scores = [f"{metric(mean(current, split, 'all'))} ({metric(mean(current, split, 'foreground'))})"
                            for split in ("hard", "held-out")]
            onnx_scores = []
            for split in ("hard", "held-out"):
                scores = []
                for key in ("all", "foreground"):
                    values = [run[f"{split}_{key}"] for run in exported if run[f"{split}_{key}"] is not None]
                    scores.append(metric(float(np.mean(values)) if values else None))
                onnx_scores.append(f"{scores[0]} ({scores[1]})")
            lines.extend([
                f"| {LABELS[name]} | PyTorch | " + " | ".join(torch_scores) +
                f" | {np.mean([run['ms'] for run in current]):.2f} |",
                f"| {LABELS[name]} | ONNX Runtime | " + " | ".join(onnx_scores) +
                f" | {np.mean([run['ms'] for run in exported]):.2f} |",
            ])
        lines.extend(["", f"Maximum absolute probability difference: {max(run['max_abs_error'] for run in onnx_rows):.2g}; "
                      f"thresholded pixel disagreements across both splits and all seeds: "
                      f"{sum(run['mask_disagreements'] for run in onnx_rows)}."])
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--steps", type=int, default=1600)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--validation", type=int, default=96)
    parser.add_argument("--eval", type=int, default=192)
    parser.add_argument("--threads", type=int, default=len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1)
    parser.add_argument("--output", type=Path, default=ROOT / "benchmarks" / "results.md")
    parser.add_argument("--work", type=Path, default=ROOT / "task1_watermark_localisation" / "work" / "bench")
    parser.add_argument("--checkpoint-dir", type=Path, help="Re-score prior runs and remeasure CPU latency without retraining.")
    args = parser.parse_args()
    if args.size < 64 or args.size > 256 or args.size % 32:
        parser.error("--size must be divisible by 32 and between 64 and 256.")
    if min(args.steps, args.batch, args.validation, args.eval, args.threads) < 1:
        parser.error("--steps, --batch, --validation, --eval and --threads must be positive.")
    if len(set(args.seeds)) != len(args.seeds) or any(not 0 <= seed < 10000 for seed in args.seeds):
        parser.error("--seeds must contain distinct integers between 0 and 9999.")
    return args


def main() -> None:
    args = parse_args()
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(args.threads)
    args.work.mkdir(parents=True, exist_ok=True)
    scenes = {
        "validation": corpus(EVAL_SEEDS["validation"], args.validation, args.size),
        "hard": corpus(EVAL_SEEDS["hard"], args.eval, args.size),
        "held-out": corpus(EVAL_SEEDS["held-out"], args.eval, args.size, HELD_OUT),
    }
    eval_data = {key: tensors(value)[:2] for key, value in scenes.items()}
    results: dict[str, list[dict]] = {name: [] for name in dict.fromkeys(args.models)}
    for seed in args.seeds:
        training = (
            tensors(corpus(20_000 + seed, args.steps * args.batch, args.size))
            if args.checkpoint_dir is None else None
        )
        for name, rows in results.items():
            checkpoint = f"{name}-{seed}.pt"
            if training is None:
                model = build_model(name, args.size).eval()
                model.load_state_dict(torch.load(args.checkpoint_dir / checkpoint, map_location="cpu", weights_only=True))
                train_seconds = None
            else:
                start = perf_counter()
                model = train(name, seed, args.size, training, args.batch)
                train_seconds = perf_counter() - start
            x, y = eval_data["validation"]
            threshold = calibrate(probabilities(model, x), y[:, 0].numpy(),
                                  foreground_weight=0.2 if name in ITERATIONS else 0.0)
            row = dict(seed=seed, threshold=threshold, hash=weight_hash(model), train_seconds=train_seconds)
            row["failures"] = {}
            for split in ("hard", "held-out"):
                probs = probabilities(model, eval_data[split][0])
                row[split] = measure(probs, scenes[split], threshold)
                row["failures"][split] = worst_cases(probs, scenes[split], threshold)
            row["ms"] = elapsed_ms(model, eval_data["hard"][0])
            if name in ITERATIONS:
                row["onnx"] = measure_onnx(model, args.work / f"{name}-{seed}.onnx", threshold, scenes, args.threads)
            rows.append(row)
            if training is not None:
                torch.save(model.state_dict(), args.work / checkpoint)
            timing = f"train={train_seconds:.1f} s" if train_seconds is not None else "checkpoint"
            print(f"{name} seed={seed} hard={metric(row['hard']['all'])} held-out={metric(row['held-out']['all'])} "
                  f"latency={row['ms']:.2f} ms {timing} sha256={row['hash']}", flush=True)
    (args.work / "runs.json").write_text(json.dumps(results, indent=2) + "\n")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    empty = {key: measure(np.zeros((len(value), args.size, args.size)), value, 0.5)
             for key, value in scenes.items() if key != "validation"}
    args.output.write_text(render_report(results, args, empty))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
