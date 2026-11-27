"""Compare localizer probabilities and warmed CPU latency across runtimes."""

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np
import onnxruntime as ort
import torch
from numpy.typing import NDArray
from torch import Tensor

from murmell.invisible_model import ResNetUnet
from murmell.onnx import ProbabilityMask, export_mask_model
from murmell.provenance import DEFAULT_WEIGHTS

from .invisible_bench import Corpus, input_tensor


def milliseconds(run: Callable[[Tensor], NDArray], sample: Tensor, repeats: int = 50) -> float:
    for _ in range(5):
        run(sample)
    timings = []
    for _ in range(repeats):
        start = perf_counter()
        run(sample)
        timings.append((perf_counter() - start) * 1_000)
    return median(timings)


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    calibration = json.loads(args.weights.with_suffix(".json").read_text())
    resolution = int(calibration["resolution"])
    model = ResNetUnet(imagenet=False)
    model.load_state_dict(torch.load(args.weights, weights_only=True, map_location="cpu"))
    export_mask_model(model, args.output, resolution)
    pytorch = ProbabilityMask(model).eval()
    options = ort.SessionOptions()
    options.intra_op_num_threads = args.threads
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(args.output), options, providers=["CPUExecutionProvider"])

    def torch_probs(image: Tensor) -> NDArray:
        return pytorch(image).numpy()

    def onnx_probs(image: Tensor) -> NDArray:
        return session.run(["probability"], {"rgb": image.numpy()})[0]

    with np.load(args.corpus) as data:
        corpus = Corpus(data["images"], data["masks"], data["messages"])
    inputs = (input_tensor(corpus, resolution).float() / 255).contiguous()
    error = max(float(np.abs(torch_probs(image) - onnx_probs(image)).max()) for image in inputs.split(1))
    hardware = next(row.partition(":")[2].strip() for row in Path("/proc/cpuinfo").read_text().splitlines()
                    if row.startswith("model name"))
    report = {"hardware": hardware, "threads": args.threads, "resolution": resolution,
              "pytorch": torch.__version__, "onnxruntime": ort.__version__, "images_compared": len(inputs),
              "pytorch_ms": milliseconds(torch_probs, inputs[:1]), "onnxruntime_ms": milliseconds(onnx_probs, inputs[:1]),
              "max_abs_error": error}
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
