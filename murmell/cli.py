"""Save a 128 by 128 invisible-watermark mask as a PNG."""

import argparse
from pathlib import Path

import torch
from PIL import Image

from murmell.provenance import DEFAULT_WEIGHTS, ProvenanceDetector


def open_image(path: Path) -> Image.Image:
    if not path.is_file():
        raise FileNotFoundError(f"Image file not found: {path}. Pass the path to an existing image.")
    try:
        with Image.open(path) as image:
            return image.convert("RGB")
    except (OSError, Image.DecompressionBombError) as exc:
        raise ValueError(f"Cannot read image: {path}. Use a valid PNG or JPEG.") from exc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--method", choices=("trustmark", "auto"), default="auto")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--threads", type=int, default=min(8, torch.get_num_threads()))
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive.")
    torch.set_num_threads(args.threads)
    try:
        image = open_image(args.image)
        result = ProvenanceDetector.load(args.weights).predict(image, args.method)
        result.mask.save(args.output, format="PNG")
        print(f"{result.method}: {result.regime}" if result.method else result.regime)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
