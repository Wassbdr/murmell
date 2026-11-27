"""Inspect the seeded evaluation data without training a detector."""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .hard_watermarks import HARD, HELD_OUT, Scene, make_scene


def label_stats(scenes: list[Scene]) -> tuple[int, int, float, float, float, float]:
    marked = [scene for scene in scenes if scene.kind != "clean"]
    coverage = np.array([scene.mask.mean() for scene in marked])
    with_mask = [scene for scene in marked if scene.mask.any()]
    signal = np.concatenate([
        (scene.alpha[scene.mask] / 255) * np.max(np.abs(scene.background[scene.mask].astype(float) - scene.ink), axis=1)
        for scene in with_mask
    ])
    return (len(marked), len(with_mask), float(np.median(coverage)), float(np.percentile(coverage, 10)),
            float(np.median(signal)), float(np.percentile(signal, 10)))


def choose_examples(scenes: list[Scene]) -> list[tuple[str, int, Scene]]:
    examples = []
    for label, match in (
        ("Thin text", lambda s: s.kind == "text"),
        ("Low-opacity tile", lambda s: s.kind == "tiled" and 0 < s.opacity < 0.22),
        ("Textured logo", lambda s: s.kind == "logo" and s.textured),
        ("Clean", lambda s: s.kind == "clean" and s.textured),
    ):
        options = [(i, s) for i, s in enumerate(scenes) if match(s)]
        index, scene = sorted(options, key=lambda item: item[1].opacity)[len(options) // 2]
        examples.append((label, index, scene))
    return examples


def write_grid(splits: dict[str, list[Scene]], path: Path) -> None:
    tile = 144
    row_height = tile * 2 + 34
    panel = Image.new("RGB", (4 * tile, len(splits) * row_height), "white")
    draw = ImageDraw.Draw(panel)
    for row, (split, scenes) in enumerate(splits.items()):
        top = row * row_height
        for col, (label, index, scene) in enumerate(choose_examples(scenes)):
            left = col * tile
            draw.text((left + 4, top + 4), f"{split} / {label}", fill="black")
            draw.text((left + 4, top + 17), f"#{index} / alpha {scene.opacity:.3f}", fill="black")
            image = Image.fromarray(scene.image).resize((tile, tile))
            mask = Image.fromarray(scene.mask.astype(np.uint8) * 255).resize((tile, tile), Image.Resampling.NEAREST)
            panel.paste(image, (left, top + 32))
            panel.paste(mask, (left, top + 32 + tile))
    panel.save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "results")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    splits = {name: [make_scene(seed, i, config=config) for i in range(192)]
              for name, seed, config in (("Hard", 12302, HARD), ("Held out", 12303, HELD_OUT))}
    write_grid(splits, args.output / "data-grid.png")
    lines = [
        "# Synthetic dataset audit", "",
        "Regenerate with `make visible-audit`. Test seeds 12302 and 12303; 192 images per split.",
        "Each image has its exact binary target below it. The grid selects the median opacity in each illustrated slice.",
        "", "![Hard and held-out examples and targets](data-grid.png)", "",
        "| Split | Slice | Marked scenes | Nonempty labels | Median mask area | P10 mask area | "
        "Median RGB signal | P10 RGB signal |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, scenes in splits.items():
        for label, subset in (
            ("All", scenes), ("Thin text", [s for s in scenes if s.kind == "text"]),
            ("Low opacity", [s for s in scenes if 0 < s.opacity < 0.22]),
            ("Textured", [s for s in scenes if s.textured]),
        ):
            n, positive, median_area, p10_area, median_signal, p10_signal = label_stats(subset)
            lines.append(f"| {name} | {label} | {n} | {positive} | {median_area:.2%} | {p10_area:.2%} | "
                         f"{median_signal:.1f} | {p10_signal:.1f} |")
    lines.extend(["", "Signal is the maximum absolute per-channel RGB change (0 to 255) at labelled pixels, "
                  "computed from the premultiplied watermark before JPEG. Area is relative to the full image.", ""])
    (args.output / "data-audit.md").write_text("\n".join(lines))
    print(f"Wrote {args.output / 'data-audit.md'} and {args.output / 'data-grid.png'}")


if __name__ == "__main__":
    main()
