"""Export a local-contrast detector checkpoint to ONNX."""

import argparse
from pathlib import Path

import torch

from ..models import ContrastUnet
from ..onnx import export_mask_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("weights", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--residual", action="store_true", help="Export a residual-supervision checkpoint.")
    args = parser.parse_args()
    if args.size < 64 or args.size > 256 or args.size % 32:
        parser.error("--size must be divisible by 32 and between 64 and 256.")
    if not args.weights.is_file():
        parser.error(f"No checkpoint at {args.weights}. Train one with make bench first.")
    net = ContrastUnet(residual=args.residual)
    net.load_state_dict(torch.load(args.weights, map_location="cpu", weights_only=True))
    export_mask_model(net, args.output, args.size)


if __name__ == "__main__":
    main()
