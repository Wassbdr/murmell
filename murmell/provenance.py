"""Localise the surviving parts of a decoded invisible provenance watermark."""

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import torch
from numpy.typing import NDArray
from PIL import Image
from torch import nn

from murmell.invisible_model import ResNetUnet
from murmell.localize import CELLS, Box, Decoder, search_box, select_mask
from murmell.trustmark import TrustMarkAdapter

DEFAULT_WEIGHTS = Path.home() / ".cache" / "murmell" / "trustmark.pt"


class RecoveringDecoder(Decoder, Protocol):
    def recover(self, image: Image.Image) -> tuple[NDArray[np.uint8], float] | None: ...


@dataclass(frozen=True)
class Gates:
    confidence: float
    full_gate: float
    full_threshold: float
    small_threshold: float
    box_gate: float
    snap_gate: float


@dataclass(frozen=True)
class Localisation:
    mask: Image.Image
    method: str | None
    regime: str
    agreement: float | None


@dataclass(frozen=True)
class Family:
    model: nn.Module
    decoder: RecoveringDecoder
    gates: Gates
    resolution: int
    decoded_region: Callable[[Image.Image], Box | None] | None = None

    @torch.inference_mode()
    def probabilities(self, image: Image.Image) -> NDArray[np.float32]:
        resized = image.convert("RGB").resize((self.resolution, self.resolution), Image.Resampling.BILINEAR)
        tensor = torch.from_numpy(np.array(resized)).permute(2, 0, 1)[None].float() / 127.5 - 1
        return self.model(tensor).sigmoid()[0].numpy()


class ProvenanceDetector:
    def __init__(self, families: Mapping[str, Family]) -> None:
        self.families = dict(families)

    @classmethod
    def load(cls, weights: str | Path = DEFAULT_WEIGHTS) -> "ProvenanceDetector":
        path = Path(weights)
        card_path = path.with_suffix(".json")
        if not path.is_file() or not card_path.is_file():
            raise FileNotFoundError(f"Missing weights or calibration at {path}. Run `make weights` or pass --weights.")
        card = json.loads(card_path.read_text())
        state = torch.load(path, map_location="cpu", weights_only=True)
        model = ResNetUnet(imagenet=False).eval()
        model.load_state_dict(state)
        decoder = TrustMarkAdapter()
        family = Family(model, decoder, Gates(**card["gates"]), int(card["resolution"]), decoder.decoded_region)
        return cls({"trustmark": family})

    def predict(self, image: Image.Image, method: str = "auto") -> Localisation:
        if method != "auto" and method not in self.families:
            raise ValueError(f"Unsupported method {method!r}. Available methods: {', '.join(self.families)} or auto.")
        rgb = image.convert("RGB")
        candidates = []
        for name, family in self.families.items():
            if method != "auto" and method != name:
                continue
            recovered = family.decoder.recover(rgb)
            if recovered is not None and recovered[1] >= family.gates.confidence:
                candidates.append((name, family, recovered[0], recovered[1]))
        if not candidates:
            return Localisation(Image.new("L", (CELLS, CELLS)), None, "no confident decode", None)
        name, family, message, agreement = max(candidates, key=lambda x: x[3] - x[1].gates.confidence)
        gates = family.gates
        region = family.decoded_region(rgb) if family.decoded_region is not None else None
        if region is not None:
            probability = family.probabilities(rgb.crop(region))
            projected = Image.new("L", (CELLS, CELLS))
            x0, y0, x1, y1 = (round(v * CELLS / dim) for v, dim in zip(region, rgb.size * 2))
            if x1 > x0 and y1 > y0:
                pixels = Image.fromarray((probability >= gates.full_threshold).astype(np.uint8) * 255)
                projected.paste(pixels.resize((x1 - x0, y1 - y0), Image.Resampling.NEAREST), (x0, y0))
            return Localisation(projected, name, "local region", agreement)
        probability = family.probabilities(rgb)
        full = agreement >= gates.full_gate
        box = search_box(family.decoder, rgb, probability, message) if not full and gates.box_gate <= 1 else None
        mask = select_mask(probability, agreement, box, (), full_gate=gates.full_gate,
                           full_threshold=gates.full_threshold, small_threshold=gates.small_threshold,
                           box_gate=gates.box_gate, snap_gate=gates.snap_gate)
        return Localisation(Image.fromarray(mask.astype(np.uint8) * 255), name,
                            "full frame with edits" if full else "local region", agreement)
