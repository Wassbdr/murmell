"""Save a visible-watermark mask as a PNG."""

import argparse
from pathlib import Path

import torch

from murmell import Detector, open_image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--variant", choices=("round1", "contrast"), default="round1")
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--threads", type=int, default=min(8, torch.get_num_threads()))
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive.")
    torch.set_num_threads(args.threads)
    try:
        image = open_image(args.image)
        detector = Detector.load(args.variant, args.weights)
        detector.predict(image).save(args.output, format="PNG")
    except (OSError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
