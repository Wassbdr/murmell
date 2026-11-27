# Murmell detector model card

## Use

Murmell highlights a visible overlay on an RGB image; it neither removes the mark nor proves its origin.
The default is a MobileNetV2 U-Net on RGB and 5-pixel high-pass features. It uses round-1 seed 0, threshold 0.900,
and a 128 px square model input. Masks resize to the input dimensions by nearest neighbour.
Local contrast is available at threshold 0.825 with `--variant contrast`.

The default checkpoint is **16,818,176 bytes** (16.04 MiB), SHA-256
`2e91c2f3eccc7689ee4f471f25f26eafc33e95594aed106827cd431b433db785`.
The local-contrast checkpoint is **465,250 bytes** (454 KiB), SHA-256
`68149d761d6f516f798de2a761336ba6dac1d406066546e75f218beb45b5c119`.
Both are bundled under the repository licence. Rebuild with `make weights VARIANT=round1` or
`make weights VARIANT=contrast`.

## Data and evaluation

Each model trains from random weights for 1,600 batches of eight images. Training corpus seed 20000.
Seed 12301 supplies 96 validation images for threshold selection; seeds 12302 and 12303 each supply 192 test images.
The generator draws text, logos and tiled overlays over procedural scenes and [licensed photos](../benchmarks/README.md).
The held-out generator has disjoint source photos, fonts, opacity ranges and layouts.
No competition data, CLWD or LVW. See the [audit grid and label counts](../benchmarks/data-audit.md).

| Checkpoint | Threshold | Hard mIoU | Hard foreground IoU | Held-out mIoU | Held-out foreground IoU |
|---|---:|---:|---:|---:|---:|
| Retrained round 1, shipped | 0.900 | 0.5992 | 0.2200 | 0.5289 | 0.0347 |
| Local contrast, opt-in | 0.825 | 0.6166 | 0.2631 | 0.5206 | 0.0283 |
| Historical demo, shipped threshold | 0.600 | 0.5774 | 0.3044 | 0.4770 | 0.0935 |
| Historical demo, validation-calibrated | 0.950 | 0.6117 | 0.3341 | 0.4886 | 0.0861 |
| All-empty mask | n/a | 0.5219 | 0.0000 | 0.5415 | 0.0000 |

The historical 256 px demo recovers more foreground on held-out scenes but scores lower in mIoU at either threshold.
The default follows held-out mIoU, not foreground IoU. Foreground IoU averages only images with nonempty targets;
mIoU averages foreground and background per image. A clean image gives an empty prediction full credit.
See [the full benchmark](../benchmarks/results.md) for three-seed results and individual failure cases.

## CPU export

`murmell-export detector.onnx` exports the default checkpoint. `make shipped` regenerates
[shipped.json](../benchmarks/shipped.json), timing and parity included. Input is RGB float32 NCHW in `[0, 1]`,
128 by 128, with dynamic batch size. The graph returns per-pixel probabilities; compare with 0.900 to form the mask.

The benchmark times batch 1 on an Intel Xeon CPU, 8 threads, median of three warmed passes over 32 hard images.
Timing includes inference and sigmoid, not file I/O, input resize, or thresholding.

| Runtime | ms/image | Hard mIoU | Held-out mIoU |
|---|---:|---:|---:|
| PyTorch 2.5.1+cpu | 22.88 | 0.5992 | 0.5289 |
| ONNX Runtime 1.20.1 | 5.88 | 0.5992 | 0.5289 |

Across both splits the maximum probability difference was 1.36 × 10⁻⁵, with one differing mask pixel.

## Failure modes

Faint held-out marks can vanish into the background; many held-out textured scenes are almost entirely missed.
Brick edges and other repeating textures can be marked as watermarks even on clean images. Downscaling can discard
thin strokes, especially in large originals. Only four photo sources were used. Scores do not predict real-world
accuracy on arbitrary images, invisible watermarks, or any given colour or compression setting.
