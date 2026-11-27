"""Run a separate, bounded WAM or PixelSeal comparison on CPU."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F

from murmell.invisible_model import ResNetUnet

from .invisible_bench import evaluate, fit_threshold, input_tensor, load_corpora, predict, train
from .invisible_data import CELLS
from .invisible_meta_smoke import MetaAdapter
from .invisible_references import sliding_scores

WAM_SPLITS = {"train": (501, 180), "calibration": (502, 24), "hard": (503, 24), "held-out": (504, 24)}
PIXEL_SPLITS = {"train": (601, 60), "calibration": (602, 9), "hard": (603, 12), "held-out": (604, 12)}


class PixelSealUnet(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder.requires_grad_(False)
        channels = [stage[-1].pwconv2.out_features for stage in self.encoder.stages]
        widths = [channels[-1], 96, 64, 32]
        self.up = nn.ModuleList([
            nn.Sequential(nn.Conv2d(a + skip, b, 1), nn.GroupNorm(8, b), nn.ReLU(),
                          nn.Conv2d(b, b, 3, padding=1), nn.GroupNorm(8, b), nn.ReLU())
            for a, b, skip in zip(widths, widths[1:], reversed(channels[:-1]))
        ])
        self.head = nn.Conv2d(widths[-1], 1, 1)

    def forward(self, x):
        features = []
        for down, stage in zip(self.encoder.downsample_layers, self.encoder.stages):
            x = stage(down(x))
            features.append(x)
        for up, skip in zip(self.up, reversed(features[:-1])):
            x = up(torch.cat((F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False), skip), 1))
        return F.interpolate(self.head(x), (CELLS, CELLS), mode="bilinear", align_corners=False)[:, 0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("family", choices=("wam", "pixelseal"))
    parser.add_argument("weights", type=Path)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--trustmark-checkpoint", type=Path)
    parser.add_argument("--resolution", choices=(128, 256), default=128, type=int)
    parser.add_argument("--updates", type=int)
    parser.add_argument("--threads", default=8, type=int)
    args = parser.parse_args()
    if args.family == "wam" and args.trustmark_checkpoint is None:
        parser.error("--trustmark-checkpoint is required for the WAM fine-tuning run.")
    torch.set_num_threads(args.threads)
    args.work.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    adapter = MetaAdapter(args.family, args.weights)
    corpora = load_corpora(args.work, adapter, WAM_SPLITS if args.family == "wam" else PIXEL_SPLITS)
    updates = args.updates if args.updates is not None else (600 if args.family == "wam" else 240)
    comparison = {
        "all-empty": evaluate(corpora, lambda split: np.zeros_like(corpora[split].masks), 0.5),
        "all-full": evaluate(corpora, lambda split: np.ones_like(corpora[split].masks), 0.5),
    }

    def add_result(name, maps):
        threshold = fit_threshold(maps["calibration"], corpora["calibration"].masks)
        comparison[name] = {"threshold": threshold, **evaluate(corpora, maps.__getitem__, threshold)}
        print(f"{name}: {comparison[name]['hard']['miou']:.4f} hard; "
              f"{comparison[name]['held-out']['miou']:.4f} held-out", flush=True)

    def run_model(name, model, initialization):
        images = input_tensor(corpora["train"], args.resolution)
        masks = torch.from_numpy(corpora["train"].masks)
        seconds = train(model, images, masks, seed=0, updates=updates, batch_size=4, initialization=initialization)
        maps = {split: predict(model, input_tensor(corpus, args.resolution), initialization)
                for split, corpus in corpora.items() if split != "train"}
        add_result(name, maps)
        comparison[name]["training_seconds"] = seconds

    if args.family == "wam":
        native = {
            split: np.stack([
                F.interpolate(adapter.detect(Image.fromarray(image))[:, :1].sigmoid(),
                              size=(CELLS, CELLS), mode="area")[0, 0].numpy()
                for image in corpus.images
            ])
            for split, corpus in corpora.items() if split != "train"
        }
        add_result("WAM native", native)
        for mode, name in (("imagenet", "ImageNet U-Net"), ("decoder", "TrustMark U-Net fine-tuned on WAM")):
            torch.manual_seed(0)
            model = ResNetUnet(imagenet=mode == "imagenet")
            if mode == "decoder":
                model.load_state_dict(torch.load(args.trustmark_checkpoint, map_location="cpu", weights_only=True))
            run_model(name, model, mode)
    else:
        maps = {}
        for split, corpus in corpora.items():
            if split == "train":
                continue
            start = perf_counter()
            maps[split] = np.stack([
                sliding_scores(adapter, Image.fromarray(image), message, windows=(48, 80), stride=56)
                for image, message in zip(corpus.images, corpus.messages)
            ])
            print(f"PixelSeal sliding {split}: {(perf_counter() - start) / len(corpus.images):.2f} s/image",
                  flush=True)
        add_result("PixelSeal sparse sliding", maps)
        torch.manual_seed(0)
        reference = ResNetUnet()
        reference.encoder.requires_grad_(False).eval()
        run_model("ImageNet U-Net (frozen encoder)", reference, "imagenet")
        torch.manual_seed(0)
        model = PixelSealUnet(adapter.model.detector.convnext)
        run_model("PixelSeal U-Net (frozen extractor)", model, "decoder")
    report = {"family": args.family, "threads": args.threads, "elapsed_seconds": perf_counter() - started,
              "updates": updates, "seed": 0, "resolution": args.resolution, "batch_size": 4,
              "comparison": comparison, "sizes": {split: len(corpus.images) for split, corpus in corpora.items()}}
    (args.work / f"{args.family}-comparison.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
