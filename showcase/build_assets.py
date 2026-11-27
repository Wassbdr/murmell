"""Generate demo examples, a poster and portfolio metadata from measured results."""

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from predict import overlay
from src.bench import EVAL_SEEDS, corpus, probabilities, tensors
from src.hard_watermarks import HELD_OUT
from src.metric import miou_batch

from murmell import Detector

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "task1_watermark_localisation" / "examples"
SHOWCASE = ROOT / "showcase"


def build_examples() -> list[dict]:
    detector = Detector.load()
    samples = {split: corpus(EVAL_SEEDS[split], 192, 128, **config)
               for split, config in (("hard", {}), ("held-out", {"config": HELD_OUT}))}
    ranked = {}
    for split, scenes in samples.items():
        rgb, labels = tensors(scenes)[:2]
        target = labels[:, 0].numpy()
        pred = probabilities(detector.model, rgb) >= detector.threshold
        scores = miou_batch(pred, target)
        marked = np.flatnonzero(target.any(axis=(1, 2)))
        ranked[split] = [(float(scores[i]), int(i), pred[i]) for i in marked[np.argsort(scores[marked])]]

    choices = [("good", "hard", ranked["hard"][-1]),
               ("median", "hard", ranked["hard"][len(ranked["hard"]) // 2]),
               ("failure", "held-out", ranked["held-out"][0])]
    figure, axes = plt.subplots(3, 4, figsize=(12.8, 7.2), dpi=100)
    examples = []
    for row, (name, split, (score, index, mask)) in enumerate(choices):
        scene = samples[split][index]
        Image.fromarray(scene.image).save(EXAMPLES / f"{name}.png")
        Image.fromarray(scene.mask.astype(np.uint8) * 255).save(EXAMPLES / f"{name}_mask.png")
        examples.append(dict(name=name, split=split, seed=EVAL_SEEDS[split], index=index, miou=score,
                             kind=scene.kind, opacity=scene.opacity))
        panels = (scene.image, scene.mask, mask, overlay(scene.image, mask))
        for col, (title, panel) in enumerate(zip(("Input", "Target", "Prediction", "Overlay"), panels)):
            ax = axes[row, col]
            ax.imshow(panel, cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(title)
        axes[row, 0].set_ylabel(f"{name.title()} ({split})\n{scene.kind}, opacity {scene.opacity:.2f}")
        axes[row, 3].text(1.06, 0.5, f"mIoU\n{score:.3f}", transform=axes[row, 3].transAxes, va="center")
    figure.suptitle("Murmell | Visible-watermark localisation on CPU")
    figure.subplots_adjust(left=0.11, right=0.91, top=0.89, bottom=0.04, wspace=0.18, hspace=0.18)
    figure.savefig(SHOWCASE / "poster.png")
    plt.close(figure)
    (EXAMPLES / "examples.json").write_text(json.dumps(examples, indent=2) + "\n")
    return examples


def write_metadata(examples: list[dict]) -> None:
    shipped = json.loads((ROOT / "benchmarks/shipped.json").read_text())
    detector = shipped["models"]["round1"]
    equality = json.loads((ROOT / "task2_model_equality/results/simlab.json").read_text())
    refit = json.loads((ROOT / "task3_unlearning/results/simlab.json").read_text())
    setup = (f"{shipped['image_count_per_split']} synthetic images per split, train seed 0, "
             f"128 px, {shipped['cpu']}, {shipped['threads']} threads")
    data = {
        "slug": "murmell",
        "title": "Murmell: watermark localisation",
        "tagline": "Find the pixels covered by a visible watermark with a CPU detector and a binary mask.",
        "problem": "A watermark flag does not show where the mark is. A spatial mask lets a reviewer inspect the result.",
        "approach": "A U-Net combines RGB and high-pass detail. Its threshold is selected on seeded validation scenes.",
        "stack": ["Python", "PyTorch", "ONNX Runtime", "Gradio", "Matplotlib", "ffmpeg"],
        "demo": {"gif": "showcase/demo.gif", "mp4": "showcase/demo.mp4", "poster": "showcase/poster.png",
                 "kind": "synthetic", "caption": "Live CPU predictions on two hard scenes and one held-out failure."},
        "reproduce": ["python -m pip install -e .", "murmell-detect task1_watermark_localisation/examples/good.png -o mask.png",
                      "make install && make showcase"],
        "results": [
            {"label": "Task 1: shipped detector mIoU (hard / held-out)",
             "value": f"{detector['hard']['all']:.4f} / {detector['held-out']['all']:.4f}",
             "command": "make shipped", "setup": setup},
            {"label": "Task 1: CPU latency (PyTorch / ONNX Runtime)",
             "value": f"{detector['pytorch_ms']:.2f} / {detector['onnx']['ms']:.2f} ms per image",
             "command": "make shipped", "setup": f"Batch 1; {setup}"},
            {"label": "Task 1: demo mIoU (good / median / failure)",
             "value": " / ".join(f"{example['miou']:.3f}" for example in examples),
             "command": "make showcase", "setup": setup},
            {"label": "Task 2: model-equality score",
             "value": f"{equality['score_mean']:.3f}", "command": "make task2",
             "setup": "Five seeds, 33 toy endpoints per seed, CPU"},
            {"label": "Task 3: forecast MAE vs parent baseline",
             "value": f"{refit['forecast_mae_mean']:.4f} vs {refit['parent_mae_mean']:.4f}",
             "command": "make task3", "setup": "Five seeds, 14 forecasts per seed, CPU"},
        ],
        "limits": [
            "Neither candidate beat all-empty mIoU on held-out scenes. Faint marks and new textures remain difficult.",
            "The detector was not trained on invisible watermarks. Tasks 2 and 3 simulate their competition settings.",
        ],
        "role": "Built task 1 (watermark localisation) end to end in a team that finished 1st of 44. Tasks 2 and 3 were built by teammates; I later made all three reproducible.",
        "repo": "https://github.com/Wassbdr/murmell",
    }
    (SHOWCASE / "showcase.json").write_text(json.dumps(data, indent=2) + "\n")
    print("Wrote demo examples, showcase/poster.png and showcase/showcase.json")


if __name__ == "__main__":
    torch.set_num_threads(min(8, torch.get_num_threads()))
    write_metadata(build_examples())
