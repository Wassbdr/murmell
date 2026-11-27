"""Export a bundled detector as a probability graph."""

import argparse
from pathlib import Path

import torch
from torch import Tensor, nn

from murmell.detector import SIZE, Detector


class ProbabilityMask(nn.Module):
    def __init__(self, detector: nn.Module) -> None:
        super().__init__()
        self.detector = detector

    def forward(self, images: Tensor) -> Tensor:
        probability: Tensor = self.detector(images).sigmoid()
        return probability


def export_mask_model(model: nn.Module, path: Path, size: int) -> None:
    torch.onnx.export(
        ProbabilityMask(model).eval(),
        (torch.zeros(1, 3, size, size),),
        str(path),
        input_names=["rgb"],
        output_names=["probability"],
        dynamic_axes={"rgb": {0: "batch"}, "probability": {0: "batch"}},
        opset_version=17,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--variant", choices=("round1", "contrast"), default="round1")
    parser.add_argument("--weights", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(min(8, torch.get_num_threads()))
    try:
        detector = Detector.load(args.variant, args.weights)
        export_mask_model(detector.model, args.output, SIZE)
    except (OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
