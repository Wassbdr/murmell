"""Export the invisible-watermark U-Net, without decoder or post-processing."""

import argparse
import json
from pathlib import Path

import torch
from torch import Tensor, nn

from murmell.invisible_model import ResNetUnet
from murmell.provenance import DEFAULT_WEIGHTS


class ProbabilityMask(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, rgb: Tensor) -> Tensor:
        result: Tensor = self.model(rgb * 2 - 1).sigmoid()
        return result


def export_mask_model(model: nn.Module, path: Path, resolution: int) -> None:
    torch.onnx.export(
        ProbabilityMask(model).eval(), (torch.zeros(1, 3, resolution, resolution),), str(path),
        input_names=["rgb"], output_names=["probability"],
        dynamic_axes={"rgb": {0: "batch"}, "probability": {0: "batch"}}, opset_version=17,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--resolution", choices=(128, 256), type=int)
    args = parser.parse_args()
    torch.set_num_threads(min(8, torch.get_num_threads()))
    try:
        if not args.weights.is_file():
            raise FileNotFoundError(f"Missing weights at {args.weights}. Run `make weights` or pass --weights.")
        model = ResNetUnet(imagenet=False)
        model.load_state_dict(torch.load(args.weights, map_location="cpu", weights_only=True))
        resolution = args.resolution
        if resolution is None:
            resolution = int(json.loads(args.weights.with_suffix(".json").read_text())["resolution"])
        export_mask_model(model, args.output, resolution)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
