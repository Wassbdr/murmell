"""U-Net with an ImageNet or watermark decoder backbone."""

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torchvision.models import ResNet50_Weights, resnet50

from murmell.localize import CELLS


class ResNetUnet(nn.Module):
    def __init__(self, initial: nn.Module | None = None, *, imagenet: bool = True) -> None:
        super().__init__()
        self.encoder = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2 if initial is None and imagenet else None)
        self.encoder.fc = nn.Identity()
        if initial is not None:
            self.encoder.load_state_dict(
                {key: value for key, value in initial.state_dict().items() if not key.startswith("fc.")}
            )
        widths = (2048, 128, 96, 64, 32)
        skips = (1024, 512, 256, 64)
        self.up = nn.ModuleList([
            nn.Sequential(nn.Conv2d(a + skip, b, 1), nn.GroupNorm(8, b), nn.ReLU(),
                          nn.Conv2d(b, b, 3, padding=1), nn.GroupNorm(8, b), nn.ReLU())
            for a, b, skip in zip(widths, widths[1:], skips)
        ])
        self.mask = nn.Conv2d(widths[-1], 1, 1)

    def forward(self, x: Tensor) -> Tensor:
        e = self.encoder
        stem = e.relu(e.bn1(e.conv1(x)))
        a = e.layer1(e.maxpool(stem))
        b = e.layer2(a)
        c = e.layer3(b)
        x = e.layer4(c)
        for up, skip in zip(self.up, (c, b, a, stem)):
            x = up(torch.cat((F.interpolate(x, size=skip.shape[-2:], mode="bilinear",
                                           align_corners=False), skip), dim=1))
        return F.interpolate(self.mask(x), size=(CELLS, CELLS), mode="bilinear", align_corners=False)[:, 0]
