"""Run a small, timed CPU comparison on generated TrustMark Q scenes."""

import argparse
from statistics import median
from time import perf_counter

import numpy as np
import torch
from torch.nn import functional as F
from torchvision.transforms.functional import pil_to_tensor

from murmell.invisible_model import ResNetUnet
from murmell.trustmark import TrustMarkAdapter

from .invisible_data import HELD_OUT, SIZE, make_invisible_scene, photo
from .invisible_references import sliding_scores
from .metric import score


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--updates", type=int, default=8)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    wm = TrustMarkAdapter()
    message = np.random.default_rng(0).integers(0, 2, wm.bits, dtype=np.uint8)
    image = photo("grass").resize((256, 256))
    embed_times, decode_times = [], []
    for _ in range(5):
        start = perf_counter()
        stego = wm.embed(image, message)
        embedded = perf_counter()
        accuracy = np.mean((wm.logits([stego])[0].numpy() > 0) == message)
        embed_times.append(embedded - start)
        decode_times.append(perf_counter() - embedded)
    print(f"TrustMark native embed: {median(embed_times[1:]):.3f} s; "
          f"decode: {median(decode_times[1:]):.3f} s; bit accuracy: {accuracy:.3f}", flush=True)

    start = perf_counter()
    train = [make_invisible_scene(wm, 10, n) for n in range(24)]
    calibration = [make_invisible_scene(wm, 11, n) for n in range(3)]
    hard = [make_invisible_scene(wm, 12, n) for n in range(3)]
    held = [make_invisible_scene(wm, 13, n, HELD_OUT) for n in range(3)]
    print(f"Generated {len(train) + len(calibration) + len(hard) + len(held)} scenes "
          f"at {SIZE}px in {perf_counter() - start:.2f} s", flush=True)
    y_cal, y_hard, y_held = (np.stack([s.mask for s in subset]) for subset in (calibration, hard, held))
    start = perf_counter()
    maps = [sliding_scores(wm, scene.image, scene.message) for scene in calibration + hard + held]
    probs_cal, probs_hard, probs_held = np.split(np.stack(maps), (len(calibration), len(calibration) + len(hard)))
    threshold = max(np.arange(0.55, 0.951, 0.025), key=lambda t: score(probs_cal >= t, y_cal))
    print(f"sliding mean {((perf_counter() - start) / len(maps)):.3f} s/image: "
          f"hard={score(probs_hard >= threshold, y_hard):.4f} held={score(probs_held >= threshold, y_held):.4f}",
          flush=True)
    for name, factory in (("all-empty", np.zeros_like), ("all-full", np.ones_like)):
        print(f"{name}: hard={score(factory(y_hard), y_hard):.4f} held={score(factory(y_held), y_held):.4f}",
              flush=True)

    def tensors(subset):
        return (
            torch.stack([pil_to_tensor(s.image.resize((256, 256))).float() / 255 for s in subset]),
            torch.from_numpy(np.stack([s.mask for s in subset])).float(),
        )

    x, y = tensors(train)
    x_eval, _ = tensors(calibration + hard + held)
    for name, initial in (("ImageNet U-Net", None), ("TrustMark U-Net", wm.backbone)):
        torch.manual_seed(0)
        net = ResNetUnet(initial)
        optimizer = torch.optim.AdamW(net.parameters(), lr=0.0003)

        def input_space(rgb):
            if initial is not None:
                return rgb * 2 - 1
            return (rgb - rgb.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]) / rgb.new_tensor(
                [0.229, 0.224, 0.225]
            )[None, :, None, None]

        timings = []
        for n in range(args.updates):
            start = perf_counter()
            batch = (torch.arange(4) + n * 4) % len(x)
            loss = F.binary_cross_entropy_with_logits(net(input_space(x[batch])), y[batch])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            timings.append(perf_counter() - start)
        net.eval()
        with torch.inference_mode():
            rgb = input_space(x_eval)
            net(rgb[:1])
            start = perf_counter()
            probs = torch.cat([net(sample[None]).sigmoid() for sample in rgb]).numpy()
            latency = (perf_counter() - start) / len(rgb)
        p_cal, p_hard, p_held = np.split(probs, (len(calibration), len(calibration) + len(hard)))
        t = max(np.arange(0.05, 1, 0.05), key=lambda t: score(p_cal >= t, y_cal))
        print(f"{name}: {median(timings):.3f} s/update, {latency:.3f} s/image; "
              f"hard={score(p_hard >= t, y_hard):.4f} held={score(p_held >= t, y_held):.4f}", flush=True)


if __name__ == "__main__":
    main()
