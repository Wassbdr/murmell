"""CPU segmentation models for the visible-watermark benchmark."""

import segmentation_models_pytorch as smp
import torch
from torch import Tensor, nn
from torch.nn import functional as F


def block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.SiLU(),
        nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.SiLU(),
    )


def upsample(x: Tensor, skip: Tensor) -> Tensor:
    return F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)


class Decoder(nn.Module):
    def __init__(self, channels: int = 3) -> None:
        super().__init__()
        self.stem = block(channels, 12)
        self.half_res = block(12, 20)
        self.quarter_res = block(20, 32)
        self.bridge = block(32, 48)
        self.decode_quarter = block(48 + 32, 32)
        self.decode_half = block(32 + 20, 20)
        self.decode_full = block(20 + 12, 12)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        a = self.stem(x)
        b = self.half_res(F.max_pool2d(a, 2))
        c = self.quarter_res(F.max_pool2d(b, 2))
        d = self.bridge(F.max_pool2d(c, 2))
        p = self.decode_quarter(torch.cat((upsample(d, c), c), 1))
        q = self.decode_half(torch.cat((upsample(p, b), b), 1))
        return self.decode_full(torch.cat((upsample(q, a), a), 1)), q, p


class PlainUnet(nn.Module):
    def __init__(self, channels: int = 3) -> None:
        super().__init__()
        self.decoder = Decoder(channels)
        self.mask = nn.Conv2d(12, 1, 1)

    def forward(self, image: Tensor) -> Tensor:
        return self.mask(self.decoder(image)[0])


class SelfCalibrated(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.decoder = Decoder()
        self.scales = nn.ModuleList(nn.Conv2d(c, 1, 1) for c in (12, 20, 32))
        self.refine = nn.Sequential(block(15, 12), nn.Conv2d(12, 1, 1))

    def outputs(self, image: Tensor) -> tuple[Tensor, tuple[Tensor, ...]]:
        levels = self.decoder(image)
        masks = tuple(upsample(head(x), image) for head, x in zip(self.scales, levels))
        context = torch.cat(tuple(mask.sigmoid() for mask in masks), 1)
        return masks[0] + self.refine(torch.cat((levels[0], context), 1)), masks

    def forward(self, image: Tensor) -> Tensor:
        return self.outputs(image)[0]


class Decomposition(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.decoder = Decoder()
        self.components = nn.Conv2d(12, 5, 1)
        self.refine = nn.Sequential(block(12 + 8, 12), nn.Conv2d(12, 1, 1))

    def outputs(self, image: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        features = self.decoder(image)[0]
        coarse = self.components(features)
        mask, alpha, ink = coarse[:, :1], coarse[:, 1:2].sigmoid(), coarse[:, 2:].sigmoid()
        region = mask.sigmoid() * 0.8
        background = (image - region * alpha * ink) / (1 - region * alpha)
        refined = self.refine(torch.cat((features, image, background, alpha, mask.sigmoid()), 1))
        return mask + refined, mask, alpha, ink

    def forward(self, image: Tensor) -> Tensor:
        return self.outputs(image)[0]


class SplitThenRefine(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.split = Decoder()
        self.attend = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(12, 12, 1), nn.Sigmoid())
        self.mask = nn.Conv2d(12, 1, 1)
        self.background = nn.Conv2d(12, 3, 1)
        self.refine = PlainUnet(7)

    def outputs(self, image: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        features = self.split(image)[0]
        coarse = self.mask(features * self.attend(features))
        probability = coarse.sigmoid()
        background = image * (1 - probability) + self.background(features).sigmoid() * probability
        return coarse + self.refine(torch.cat((image, background, probability), 1)), coarse, background

    def forward(self, image: Tensor) -> Tensor:
        return self.outputs(image)[0]


class RoundOne(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = HighPassUnet()

    def forward(self, image: Tensor) -> Tensor:
        return self.net((image * 255) / 255)


class HighPassUnet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.m = smp.Unet("mobilenet_v2", encoder_weights=None, in_channels=6, classes=1,
                          decoder_channels=(128, 64, 32, 16, 16))

    def forward(self, image: Tensor) -> Tensor:
        kernel = torch.ones(3, 1, 5, 5, device=image.device) / 25
        detail = (image - F.conv2d(F.pad(image, (2, 2, 2, 2), mode="reflect"), kernel, groups=3)) * 8
        logits: Tensor = self.m(torch.cat((image, detail), 1))
        return F.avg_pool2d(logits, 1)


class ContrastUnet(nn.Module):
    """Decode RGB and local contrast together without downsampling thin strokes at the stem."""

    def __init__(self, residual: bool = False) -> None:
        super().__init__()
        self.unet = PlainUnet(12)
        self.context = nn.Sequential(
            nn.Conv2d(12, 12, 3, padding=2, dilation=2, groups=12),
            nn.Conv2d(12, 12, 1),
            nn.SiLU(),
        ) if residual else None
        self.residual = nn.Conv2d(12, 3, 1) if residual else None
        self.refine = nn.Conv2d(3, 1, 1, bias=False) if residual else None
        if self.refine is not None:
            nn.init.constant_(self.refine.weight, 1 / 3)

    def outputs(self, image: Tensor) -> tuple[Tensor, Tensor | None]:
        local = image - F.avg_pool2d(image, 3, stride=1, padding=1, count_include_pad=False)
        broad = image - F.avg_pool2d(image, 9, stride=1, padding=4, count_include_pad=False)
        variation = F.avg_pool2d(local.abs(), 9, stride=1, padding=4, count_include_pad=False)
        features = self.unet.decoder(torch.cat((image, 8 * local, 4 * broad, local / (variation + 0.025)), 1))[0]
        if self.context is not None:
            features = features + self.context(features)
        mask = self.unet.mask(features)
        if self.residual is None or self.refine is None:
            return mask, None
        residual = self.residual(features)
        return mask + self.refine(residual.abs()), residual

    def forward(self, image: Tensor) -> Tensor:
        return self.outputs(image)[0]


def build_reference(name: str, size: int) -> nn.Module:
    if name == "round1":
        return RoundOne()
    if name == "unet":
        return PlainUnet()
    if name == "slbr":
        return SelfCalibrated()
    if name == "wdnet":
        return Decomposition()
    if name == "split":
        return SplitThenRefine()
    raise ValueError(f"Unknown reference {name!r}.")
