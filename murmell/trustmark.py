"""Adobe TrustMark Q with raw bit logits and BCH message recovery."""

from collections.abc import Sequence

import numpy as np
import torch
from numpy.typing import NDArray
from PIL import Image
from torch import Tensor
from torchvision.transforms.functional import pil_to_tensor

from murmell.localize import message_agreement


class TrustMarkAdapter:
    family = "trustmark"
    bits = 100

    def __init__(self) -> None:
        from trustmark import TrustMark

        self.model = TrustMark(model_type="Q", use_ECC=False, device="cpu", loadRemover=False, verbose=False)
        if self.model.encoder is None or self.model.decoder is None:
            raise RuntimeError("Could not load the TrustMark Q weights from Adobe.")
        self.backbone = self.model.decoder.decoder.decoder

    def embed(self, image: Image.Image, message: NDArray[np.uint8]) -> Image.Image:
        result: Image.Image = self.model.encode(image, "".join(map(str, message)), MODE="binary")
        return result

    def payload(self, bits: NDArray[np.uint8], version: int = 1) -> NDArray[np.uint8]:
        from trustmark.datalayer import DataLayer

        ecc = DataLayer(self.bits, verbose=False, encoding_mode=version)
        return np.asarray(ecc.encode_binary(["".join(map(str, bits))])[0], dtype=np.uint8)

    @torch.inference_mode()
    def logits(self, images: Sequence[Image.Image]) -> Tensor:
        resolution = self.model.model_resolution_dec
        data = torch.stack([
            pil_to_tensor(self.model.get_the_image_for_processing(image).resize(
                (resolution, resolution), Image.Resampling.BILINEAR,
            ))
            for image in images
        ])
        logits: Tensor = self.model.decoder.decoder(data.float() / 127.5 - 1)
        return logits

    def recover(self, image: Image.Image) -> tuple[NDArray[np.uint8], float] | None:
        logits = self.logits([image])
        bits, valid, version = self.model.ecc.decode_bitstream((logits > 0).numpy(), MODE="binary")[0]
        if not valid:
            return None
        message = self.payload(np.fromiter(map(int, bits), dtype=np.uint8), version)
        _, agreement = message_agreement(logits, message)
        return message, float(agreement[0])

    def decoded_region(self, image: Image.Image) -> tuple[int, int, int, int] | None:
        width, height = self.model.get_the_image_for_processing(image).size
        if (width, height) == image.size:
            return None
        x, y = (image.width - width) // 2, (image.height - height) // 2
        return x, y, x + width, y + height
