# Invisible watermark localisation on CPU

The public run uses licensed photos and the true embedded message, not competition data. See the [sources and licences](invisible-sources.md) and [benchmark commands](README.md).

Hardware: Intel(R) Xeon(R) Processor, 8 PyTorch threads. TrustMark uses 1,500 training, 48 calibration, and 60 images per test split, balanced by regime. Training runs use 1,600 batch-4 updates and 128 px inputs; masks use 128 by 128 cells. Train seeds 0, 1 and 2 share the fixed scored images. Fitted thresholds use calibration labels only. Learned rows are mean ± sample standard deviation; deterministic reference rows are single values.

## TrustMark: known message

| Model | Hard mIoU | Hard foreground IoU | Held-out mIoU | Held-out foreground IoU |
|---|---:|---:|---:|---:|
| ImageNet U-Net | 0.4870 ± 0.0086 | 0.3825 ± 0.0283 | 0.3402 ± 0.0097 | 0.1580 ± 0.0911 |
| TrustMark sliding-window | 0.3433 | 0.1004 | 0.3712 | 0.1625 |
| All-empty | 0.3232 | 0.0000 | 0.3026 | 0.0000 |
| All-full | 0.1768 | 0.3535 | 0.1974 | 0.3947 |
| TrustMark decoder U-Net | 0.4934 ± 0.0082 | 0.4213 ± 0.0675 | 0.3997 ± 0.0659 | 0.2931 ± 0.0161 |
| Global decode regime | 0.6193 ± 0.0042 | 0.5979 ± 0.0081 | 0.3883 ± 0.0001 | 0.2107 ± 0.0001 |
| Regime and box search | 0.6087 ± 0.0012 | 0.5785 ± 0.0026 | 0.4092 ± 0.0184 | 0.2489 ± 0.0332 |
| Regime, boxes and MobileSAM | 0.6257 ± 0.0017 | 0.6116 ± 0.0030 | 0.4093 ± 0.0184 | 0.2490 ± 0.0333 |

### Learnability gate (seed 0)

| Model | Hard mIoU | Hard foreground IoU | Held-out mIoU | Held-out foreground IoU |
|---|---:|---:|---:|---:|
| ImageNet U-Net | 0.4935 | 0.4140 | 0.3460 | 0.1123 |
| All-empty | 0.3232 | 0.0000 | 0.3026 | 0.0000 |

The ImageNet model trained in 900.0 seconds. It beat all-empty on hard, so the benchmark stayed at 128 px.

### Base recipe: decoder initialisation and global regime

| Model | Hard mIoU | Hard foreground IoU | Held-out mIoU | Held-out foreground IoU |
|---|---:|---:|---:|---:|
| TrustMark decoder U-Net | 0.4934 ± 0.0082 | 0.4213 ± 0.0675 | 0.3997 ± 0.0659 | 0.2931 ± 0.0161 |
| Global decode regime | 0.6193 ± 0.0042 | 0.5979 ± 0.0081 | 0.3883 ± 0.0001 | 0.2107 ± 0.0001 |

### Iteration 1: bounded message-guided box search

| Model | Hard mIoU | Hard foreground IoU | Held-out mIoU | Held-out foreground IoU |
|---|---:|---:|---:|---:|
| Global decode regime | 0.6193 ± 0.0042 | 0.5979 ± 0.0081 | 0.3883 ± 0.0001 | 0.2107 ± 0.0001 |
| Regime and box search | 0.6087 ± 0.0012 | 0.5785 ± 0.0026 | 0.4092 ± 0.0184 | 0.2489 ± 0.0332 |

Local crops compete by bit accuracy, with soft agreement as a tie-break and a maximum of 24 proposals.

### Iteration 2: MobileSAM snapping

| Model | Hard mIoU | Hard foreground IoU | Held-out mIoU | Held-out foreground IoU |
|---|---:|---:|---:|---:|
| Regime and box search | 0.6087 ± 0.0012 | 0.5785 ± 0.0026 | 0.4092 ± 0.0184 | 0.2489 ± 0.0332 |
| Regime, boxes and MobileSAM | 0.6257 ± 0.0017 | 0.6116 ± 0.0030 | 0.4093 ± 0.0184 | 0.2490 ± 0.0333 |

Snap proposals are compared with the U-Net mask for local images without an accepted decoder box. The calibration grid includes disabling the snap on seeds where it does not help. No separate per-image selector was fit: the small calibration split is used for thresholds and does not provide out-of-fold training predictions for a selector.

On hard, the full chain differs from the strongest reference (ImageNet U-Net) by +0.1387 mIoU.

On held-out, the full chain differs from the strongest reference (TrustMark sliding-window) by +0.0381 mIoU.

MobileSAM mask changes by seed (hard / held-out): 14 / 0, 15 / 0, 14 / 0.

Seed-0 hard full-chain mIoU by regime: holes 0.4795, rectangle 0.4831, object 0.9150.

Seed-0 held-out full-chain mIoU by regime: holes 0.3402, rectangle 0.4580, object 0.4700.

## WAM: separate slice

Scenes (train: 180, calibration: 24, hard: 24, held-out: 24). Total elapsed time: 8.1 minutes on 8 threads.

Learned rows: seed 0, 128 px, 600 batch-4 updates.

| Model | Hard mIoU | Hard foreground IoU | Held-out mIoU | Held-out foreground IoU |
|---|---:|---:|---:|---:|
| all-empty | 0.3293 | 0.0000 | 0.2991 | 0.0000 |
| all-full | 0.1707 | 0.3414 | 0.2009 | 0.4018 |
| WAM native | 0.6296 | 0.3308 | 0.7157 | 0.5449 |
| ImageNet U-Net | 0.5203 | 0.5577 | 0.8471 | 0.8370 |
| TrustMark U-Net fine-tuned on WAM | 0.5768 | 0.5768 | 0.2271 | 0.3885 |

## PixelSeal: separate slice

Scenes (train: 60, calibration: 9, hard: 12, held-out: 12). Total elapsed time: 1.3 minutes on 8 threads.

Learned rows: seed 0, 128 px, 240 batch-4 updates.

| Model | Hard mIoU | Hard foreground IoU | Held-out mIoU | Held-out foreground IoU |
|---|---:|---:|---:|---:|
| all-empty | 0.3215 | 0.0000 | 0.3000 | 0.0000 |
| all-full | 0.1785 | 0.3569 | 0.2000 | 0.3999 |
| PixelSeal sparse sliding | 0.4717 | 0.3169 | 0.4409 | 0.2903 |
| ImageNet U-Net (frozen encoder) | 0.6327 | 0.6353 | 0.3815 | 0.1514 |
| PixelSeal U-Net (frozen extractor) | 0.5823 | 0.5002 | 0.4812 | 0.3865 |

WAM uses native masks and a TrustMark-initialised model fine-tuned on WAM; the ViT extractor does not expose multiresolution U-Net skips. PixelSeal uses a sparse sliding grid and a U-Net with skips from its frozen ConvNeXt extractor. Both PixelSeal learned rows freeze their encoders. The slices do not support a combined cross-family win claim.
