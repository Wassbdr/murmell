"""Format completed invisible-watermark runs without changing their scores."""

import argparse
import json
from pathlib import Path
from statistics import mean, stdev

SPLITS = ("hard", "held-out")
STAGES = {"regime": "Global decode regime", "boxes": "Regime and box search", "snap": "Regime, boxes and MobileSAM"}
HEADER = ("| Model | Hard mIoU | Hard foreground IoU | Held-out mIoU | Held-out foreground IoU |\n"
          "|---|---:|---:|---:|---:|")


def load(path):
    return json.loads(path.read_text())


def columns(scores):
    return [scores[split][metric] for split in SPLITS for metric in ("miou", "foreground_iou")]


def average(rows):
    return [mean(values) for values in zip(*map(columns, rows))]


def table(rows):
    lines = [HEADER]
    for name, samples in rows.items():
        values = list(zip(*map(columns, samples)))
        cells = [f"{mean(sample):.4f}" if len(sample) == 1 else f"{mean(sample):.4f} ± {stdev(sample):.4f}"
                 for sample in values]
        lines.append("| " + " | ".join([name, *cells]) + " |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work", type=Path)
    parser.add_argument("--timing", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    trustmark = args.work / "trustmark-bench"
    trained = {
        mode: [load(trustmark / f"trustmark-{mode}-128-{seed}-1600.json") for seed in range(3)]
        for mode in ("imagenet", "decoder")
    }
    stages = [load(trustmark / f"trustmark-decoder-128-{seed}-1600-stages.json") for seed in range(3)]
    sliding = load(trustmark / "trustmark-sliding.json")
    timing = load(args.timing)
    gate = trained["imagenet"][0]
    references = {"ImageNet U-Net": [run["imagenet"] for run in trained["imagenet"]],
                  "TrustMark sliding-window": [sliding["scores"]],
                  "All-empty": [gate["all-empty"]], "All-full": [gate["all-full"]]}
    decoder = {"TrustMark decoder U-Net": [run["decoder"] for run in trained["decoder"]]}
    stage_rows = {label: [run[stage]["scores"] for run in stages] for stage, label in STAGES.items()}
    final = {**references, **decoder, **stage_rows}
    lines = [
        "# Invisible watermark localisation on CPU",
        "The public run uses licensed photos and the true embedded message, not competition data. "
        "See the [sources and licences](invisible-sources.md) and [benchmark commands](README.md).",
        f"Hardware: {timing['hardware']}, {gate['threads']} PyTorch threads. TrustMark uses "
        "1,500 training, 48 calibration, and 60 images per test split, balanced by regime. "
        "Training runs use 1,600 batch-4 updates and 128 px inputs; masks use 128 by 128 cells. "
        "Train seeds 0, 1 and 2 share the fixed scored images. Fitted thresholds use calibration labels only. "
        "Learned rows are mean ± sample standard deviation; deterministic reference rows are single values.",
        "## TrustMark: known message",
        table(final),
        "### Learnability gate (seed 0)",
        table({"ImageNet U-Net": [gate["imagenet"]], "All-empty": [gate["all-empty"]]}),
        f"The ImageNet model trained in {gate['training_seconds']:.1f} seconds. It beat all-empty on hard, so the "
        "benchmark stayed at 128 px.",
        "### Base recipe: decoder initialisation and global regime",
        table({**decoder, STAGES["regime"]: stage_rows[STAGES["regime"]]}),
        "### Iteration 1: bounded message-guided box search",
        table({label: stage_rows[label] for label in (STAGES["regime"], STAGES["boxes"])}),
        "Local crops compete by bit accuracy, with soft agreement as a tie-break and a maximum of 24 proposals.",
        "### Iteration 2: MobileSAM snapping",
        table({label: stage_rows[label] for label in (STAGES["boxes"], STAGES["snap"])}),
        "Snap proposals are compared with the U-Net mask for local images without an accepted decoder box. "
        "The calibration grid includes disabling the snap on seeds where it does not help. No separate per-image "
        "selector was fit: the small calibration split is used for thresholds and does not provide "
        "out-of-fold training predictions for a selector.",
    ]
    for split, col in (("hard", 0), ("held-out", 2)):
        name, value = max(((name, average(results)[col]) for name, results in references.items()),
                          key=lambda item: item[1])
        gain = average(stage_rows[STAGES["snap"]])[col] - value
        lines.append(f"On {split}, the full chain differs from the strongest reference ({name}) by {gain:+.4f} mIoU.")
    lines.append("MobileSAM mask changes by seed (hard / held-out): " + ", ".join(
        f"{run['snap']['snap_changed']['hard']} / {run['snap']['snap_changed']['held-out']}" for run in stages
    ) + ".")
    for split in SPLITS:
        lines.append(f"Seed-0 {split} full-chain mIoU by regime: " + ", ".join(
            f"{regime} {metrics['miou']:.4f}" for regime, metrics in stages[0]["snap"]["scores"][split]["regimes"].items()
        ) + ".")
    for family in ("wam", "pixelseal"):
        report = load(args.work / f"{family}-bench" / f"{family}-comparison.json")
        sizes = ", ".join(f"{split}: {count}" for split, count in report["sizes"].items())
        total = report["elapsed_seconds"] / 60
        lines.extend((f"## {'WAM' if family == 'wam' else 'PixelSeal'}: separate slice",
                      f"Scenes ({sizes}). Total elapsed time: {total:.1f} minutes on {report['threads']} threads.",
                      f"Learned rows: seed {report['seed']}, {report['resolution']} px, "
                      f"{report['updates']} batch-{report['batch_size']} updates.",
                      table({name: [values] for name, values in report["comparison"].items()})))
    lines.append("WAM uses native masks and a TrustMark-initialised model fine-tuned on WAM; the ViT extractor "
                 "does not expose multiresolution U-Net skips. PixelSeal uses a sparse sliding grid and a U-Net "
                 "with skips from its frozen ConvNeXt extractor. Both PixelSeal learned rows freeze their encoders. "
                 "The slices do not support a combined cross-family win claim.")
    summary = {"hardware": timing["hardware"], "rows": {name: average(rows) for name, rows in final.items()},
               "showcase_results": [
                   {"label": "Task 1: known-message TrustMark mIoU (hard / held-out)",
                    "value": " / ".join(f"{average(stage_rows[STAGES['snap']])[col]:.4f}" for col in (0, 2)),
                    "command": "See benchmarks/README.md",
                    "setup": f"Three seeds, 128 px, {timing['hardware']}, {timing['threads']} threads"},
                   {"label": "Task 1: U-Net latency (PyTorch / ONNX Runtime)",
                    "value": f"{timing['pytorch_ms']:.2f} / {timing['onnxruntime_ms']:.2f} ms per image",
                    "command": "See benchmarks/README.md",
                    "setup": f"Batch 1, {timing['resolution']} px, {timing['hardware']}, {timing['threads']} threads"},
               ]}
    (args.output / "results.md").write_text("\n\n".join(lines) + "\n")
    (args.output / "trustmark.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Wrote {args.output / 'results.md'}")


if __name__ == "__main__":
    main()
