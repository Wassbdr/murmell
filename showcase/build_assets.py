"""Generate the edited example, poster, and showcase metadata."""

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from app import overlay
from PIL import Image
from src.invisible_data import CELLS, SIZE, make_invisible_scene, photo
from src.invisible_product import EccEmbedder
from src.metric import miou_image

from murmell import ProvenanceDetector
from murmell.trustmark import TrustMarkAdapter

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "task1_watermark_localisation/examples/edited.png"
SHOWCASE = ROOT / "showcase"
SEED = 202615


def build_example() -> dict:
    detector = ProvenanceDetector.load()
    adapter = detector.families["trustmark"].decoder
    if not isinstance(adapter, TrustMarkAdapter):
        raise TypeError("The showcase requires TrustMark.")
    embedder = EccEmbedder(adapter, marked=True)
    index = 0
    scene = make_invisible_scene(embedder, SEED, index)
    donor = photo("hubble_deep_field").resize((SIZE, SIZE), Image.Resampling.LANCZOS)
    edited = Image.new("RGB", (3 * SIZE, SIZE))
    edited.paste(donor, (0, 0))
    edited.paste(scene.image, (SIZE, 0))
    edited.paste(donor, (2 * SIZE, 0))
    support = np.zeros((SIZE, 3 * SIZE), dtype=bool)
    support[:, SIZE:2 * SIZE] = scene.support
    edge = SIZE // CELLS
    target = support.reshape(CELLS, edge, CELLS, 3 * edge).sum(axis=(1, 3)) >= 3 * edge**2 / 2
    recovered = adapter.recover(edited)
    result = detector.predict(edited)
    if (recovered is None or not np.array_equal(recovered[0], adapter.payload(scene.message))
            or result.regime != "local region" or result.mask.getbbox() is None):
        raise RuntimeError("The edited example did not produce a confident local mask.")

    EXAMPLE.parent.mkdir(parents=True, exist_ok=True)
    edited.save(EXAMPLE)
    panels = (edited, target, overlay(edited, result.mask))
    titles = ("Edited image", "Where the mark was applied", "Predicted regions")
    figure, axes = plt.subplots(1, 3, figsize=(12.8, 7.2), dpi=100, layout="constrained")
    for axis, title, panel in zip(axes, titles, panels):
        axis.imshow(panel, cmap="gray", vmin=0, vmax=1, aspect="auto")
        axis.set_box_aspect(1 / 3)
        axis.set_title(title)
        axis.axis("off")
    figure.suptitle(f"Invisible provenance watermark | {result.method}: {result.regime}", fontsize=20)
    figure.savefig(SHOWCASE / "poster.png")
    plt.close(figure)
    return {"seed": SEED, "index": index, "regime": result.regime,
            "miou": miou_image(np.asarray(result.mask) > 0, target)}


def write_metadata(example: dict) -> None:
    measured = json.loads((ROOT / "benchmarks/trustmark.json").read_text())
    prior = json.loads((SHOWCASE / "showcase.json").read_text())
    data = {
        "slug": "murmell",
        "title": "Murmell: invisible watermark localisation",
        "tagline": "Find the parts of an edited image that still carry an invisible provenance watermark.",
        "problem": "An image may still carry bits of a watermark after splicing, inpainting, or cropping.",
        "approach": "Decode TrustMark, then localise cells in the decoded view with a decoder-initialised U-Net.",
        "stack": ["Python", "PyTorch", "ONNX Runtime", "Gradio", "Matplotlib", "ffmpeg"],
        "demo": {"gif": "showcase/demo.gif", "mp4": "showcase/demo.mp4", "poster": "showcase/poster.png",
                 "kind": "synthetic", "caption": f"Spliced example, seed {example['seed']}, index {example['index']}."},
        "reproduce": ["python -m pip install -r benchmarks/requirements.txt -e .", "make weights",
                      "murmell-detect task1_watermark_localisation/examples/edited.png --method auto -o mask.png"],
        "results": measured["showcase_results"] + prior["results"][-2:],
        "limits": [
            "The benchmark knows the embedded message; the product abstains if global decoding is not confident.",
            "An empty mask does not establish that a watermark is absent.",
            "Tasks 2 and 3 simulate their competition settings.",
        ],
        "role": prior["role"],
        "repo": "https://github.com/Wassbdr/murmell",
    }
    (SHOWCASE / "showcase.json").write_text(json.dumps(data, indent=2) + "\n")
    print("Wrote edited example, showcase/poster.png and showcase/showcase.json")


if __name__ == "__main__":
    torch.set_num_threads(min(8, torch.get_num_threads()))
    write_metadata(build_example())
