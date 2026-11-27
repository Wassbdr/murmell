"""Time official WAM or PixelSeal inference with public images and checkpoints."""

import argparse
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from torchvision.transforms.functional import pil_to_tensor

from .invisible_data import CELLS, HELD_OUT, MAIN, make_invisible_scene, photo
from .metric import score


class MetaAdapter:
    def __init__(self, family, weights):
        self.family = family
        if family == "wam":
            from notebooks.inference_utils import load_model_from_checkpoint
            from watermark_anything.data.transforms import default_transform, unnormalize_img

            self.model = load_model_from_checkpoint("checkpoints/params.json", str(weights)).eval()
            self.transform = default_transform
            self.to_rgb = unnormalize_img
            self.bits = 32
            self.options = {}
        else:
            from videoseal.utils.cfg import setup_model_from_checkpoint

            self.model = setup_model_from_checkpoint(str(weights)).eval()
            self.transform = lambda image: pil_to_tensor(image).float() / 255
            self.to_rgb = lambda x: x
            self.bits = 256
            self.options = {"is_video": False}

    @torch.inference_mode()
    def embed(self, image, message):
        x = self.transform(image)[None]
        msg = torch.as_tensor(message).float()[None]
        y = self.model.embed(x, msg, **self.options)["imgs_w"]
        out = self.to_rgb(y)[0].clamp(0, 1).mul(255).round().byte()
        return Image.fromarray(out.permute(1, 2, 0).numpy())

    @torch.inference_mode()
    def detect(self, image):
        return self.model.detect(self.transform(image)[None], **self.options)["preds"]

    @torch.inference_mode()
    def logits(self, images):
        inputs = torch.stack([self.transform(image.resize((256, 256), Image.Resampling.BILINEAR)) for image in images])
        prediction = self.model.detect(inputs, **self.options)["preds"]
        if prediction.ndim == 4:
            return prediction[:, 1:].flatten(2).mean(-1)
        return prediction[:, 1:]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("family", choices=("wam", "pixelseal"))
    parser.add_argument("weights", type=Path)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    model = MetaAdapter(args.family, args.weights)
    image = photo("grass").resize((256, 256))
    message = np.random.default_rng(0).integers(0, 2, model.bits, dtype=np.uint8)
    embeds, detects = [], []
    for _ in range(5):
        start = perf_counter()
        marked = model.embed(image, message)
        mid = perf_counter()
        pred = model.detect(marked)
        embeds.append(mid - start)
        detects.append(perf_counter() - mid)
    logits = pred[0, 1:].flatten(1).mean(1) if pred.ndim == 4 else pred[0, 1:]
    accuracy = np.mean((logits.numpy() > 0) == message)
    print(f"{args.family} 256px embed {median(embeds[1:]):.3f} s; "
          f"detect {median(detects[1:]):.3f} s; bit accuracy {accuracy:.3f}", flush=True)

    if args.family != "wam":
        return
    calibration, hard, held = (
        [make_invisible_scene(model, seed, n, config) for n in range(3)]
        for seed, config in ((11, MAIN), (12, MAIN), (13, HELD_OUT))
    )
    maps, targets = [], []
    for subset in (calibration, hard, held):
        maps.append(np.stack([
            F.interpolate(model.detect(scene.image)[:, :1].sigmoid(), size=(CELLS, CELLS), mode="area"
                          )[0, 0].numpy() for scene in subset
        ]))
        targets.append(np.stack([scene.mask for scene in subset]))
    threshold = max(np.arange(0.05, 1, 0.05), key=lambda t: score(maps[0] >= t, targets[0]))
    print(f"WAM native mask: hard={score(maps[1] >= threshold, targets[1]):.4f} "
          f"held={score(maps[2] >= threshold, targets[2]):.4f} at threshold={threshold:.2f}", flush=True)


if __name__ == "__main__":
    main()
