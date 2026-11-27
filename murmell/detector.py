"""Load a bundled model and return a binary mask at the input size."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from numpy.typing import NDArray
from PIL import Image
from torch import nn

from murmell.models import ContrastUnet, RoundOne

Variant = Literal["round1", "contrast"]
SIZE = 128
THRESHOLDS: dict[Variant, float] = {"round1": 0.900, "contrast": 0.825}
WEIGHTS = Path(__file__).with_name("weights")


def open_image(path: str | Path) -> Image.Image:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Image file not found: {source}. Pass the path to an existing image.")
    try:
        with Image.open(source) as image:
            return image.convert("RGB")
    except (OSError, Image.DecompressionBombError) as exc:
        raise ValueError(f"Cannot read image: {source}. Use a valid PNG or JPEG.") from exc


@dataclass(frozen=True)
class Detector:
    model: nn.Module
    threshold: float
    variant: Variant

    @classmethod
    def load(cls, variant: Variant = "round1", weights: str | Path | None = None) -> "Detector":
        if variant not in THRESHOLDS:
            raise ValueError(f"Unknown model {variant!r}. Choose round1 or contrast.")
        path = Path(weights) if weights is not None else WEIGHTS / f"{variant}-0.pt"
        if not path.is_file():
            raise FileNotFoundError(
                f"Weights not found: {path}. Reinstall murmell for bundled weights, or run "
                f"`make weights VARIANT={variant}` in the source checkout and pass --weights PATH."
            )
        model = RoundOne() if variant == "round1" else ContrastUnet()
        model = model.to(memory_format=torch.channels_last).eval()  # type: ignore[call-overload]
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        return cls(model, THRESHOLDS[variant], variant)

    @torch.inference_mode()
    def probabilities(self, image: Image.Image) -> NDArray[np.float32]:
        resized = image.convert("RGB").resize((SIZE, SIZE), Image.Resampling.BILINEAR)
        pixels = torch.from_numpy(np.array(resized)).permute(2, 0, 1)[None].float() / 255
        return self.model(pixels).sigmoid()[0, 0].numpy()

    def predict(self, image: Image.Image) -> Image.Image:
        probability = self.probabilities(image)
        mask = Image.fromarray((probability >= self.threshold).astype(np.uint8) * 255)
        return mask.resize(image.size, Image.Resampling.NEAREST)
