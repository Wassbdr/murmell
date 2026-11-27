# Visible-watermark localisation benchmark

Regenerate with `make bench`. See [data and references](README.md).

CPU: Intel(R) Xeon(R) Processor, 8 PyTorch threads, PyTorch 2.5.1+cpu, Python 3.10.12.
Training seeds: 0, 1, 2; each run starts from random weights and sees 1600 batches of 8 images at 128 px (one pass over a seed-specific corpus, seeds 20000, 20001, 20002).
Validation: 96 images, seed 12301; hard test: 192 images, seed 12302; held-out test: 192 images, seed 12303.
Each model/seed uses one global threshold chosen on validation only. The Murmell recipes maximize validation mIoU plus 0.2 times foreground IoU; the references maximize validation mIoU. mIoU is the mean of foreground and background IoU per image, averaged across images, then across training seeds.
Foreground IoU is averaged only over images with a nonempty target, without giving credit for empty masks.
CPU latency is batch 1, warmed up, median of three passes over 32 images per seed, then averaged across seeds. It includes the model and sigmoid but excludes I/O, input conversion and thresholding.

## Iteration 0: references

| Model | Hard mIoU | Hard foreground IoU | Held-out mIoU | Held-out foreground IoU | CPU ms/image |
|---|---:|---:|---:|---:|---:|
| Murmell round 1 (RGB + high-pass U-Net) | 0.5995 | 0.2260 | 0.5308 | 0.0455 | 9.80 |
| WDNet-style decomposition | 0.5874 | 0.1952 | 0.5083 | 0.0081 | 8.55 |
| Split then Refine-style mask | 0.5873 | 0.1977 | 0.5181 | 0.0126 | 13.95 |
| SLBR-style self-calibrated mask | 0.5816 | 0.1853 | 0.5154 | 0.0131 | 7.70 |
| Plain RGB U-Net | 0.5784 | 0.1809 | 0.5077 | 0.0130 | 7.31 |
| All-empty mask | 0.5219 | 0.0000 | 0.5415 | 0.0000 | n/a |

## Iteration 1: generalize local contrast across background textures

Feed two local residual scales and texture-normalized contrast into the full-resolution U-Net stem. Recompose some training marks over colour-jittered multiscale textures without changing alpha. Use a focal penalty and include foreground IoU in threshold calibration.

| Model | Hard mIoU (foreground IoU) | Δ mIoU vs best reference | Δ mIoU vs previous | Held-out mIoU (foreground IoU) | Δ mIoU vs best reference | Δ mIoU vs previous | CPU ms/image |
|---|---:|---:|---:|---:|---:|---:|---:|
| Murmell local contrast | 0.6254 (0.2686) | +0.0259 | n/a | 0.5243 (0.0248) | -0.0172 | n/a | 11.49 |

## Iteration 2: supervise the watermark's contribution after texture and colour jitter

Add a dilated detail block and predict the signed, colour-transformed difference introduced by the watermark. Feed its magnitude into the mask head and mix full images with foreground-biased crops. Keep the same foreground-aware threshold calibration.

| Model | Hard mIoU (foreground IoU) | Δ mIoU vs best reference | Δ mIoU vs previous | Held-out mIoU (foreground IoU) | Δ mIoU vs best reference | Δ mIoU vs previous | CPU ms/image |
|---|---:|---:|---:|---:|---:|---:|---:|
| Murmell residual supervision | 0.6015 (0.2398) | +0.0020 | -0.0239 | 0.5062 (0.0252) | -0.0353 | -0.0181 | 10.10 |

## Hard breakdown

Each cell reports mIoU (foreground IoU).

| Model | thin text | low opacity | textured | logo | tiled | clean |
|---|---:|---:|---:|---:|---:|---:|
| Murmell local contrast | 0.6045 (0.2197) | 0.5606 (0.1488) | 0.5914 (0.2019) | 0.6219 (0.2660) | 0.6395 (0.3381) | 0.6896 (n/a) |
| Murmell residual supervision | 0.5893 (0.1909) | 0.5415 (0.1137) | 0.5806 (0.1939) | 0.6112 (0.2457) | 0.6170 (0.2967) | 0.5579 (n/a) |
| Murmell round 1 (RGB + high-pass U-Net) | 0.5714 (0.1573) | 0.5633 (0.1588) | 0.5765 (0.1733) | 0.6300 (0.2817) | 0.5771 (0.2376) | 0.6539 (n/a) |
| WDNet-style decomposition | 0.5735 (0.1574) | 0.5259 (0.0815) | 0.5735 (0.1580) | 0.5906 (0.2018) | 0.5852 (0.2363) | 0.6418 (n/a) |
| Split then Refine-style mask | 0.5623 (0.1374) | 0.5381 (0.1082) | 0.5809 (0.1773) | 0.6027 (0.2279) | 0.5791 (0.2347) | 0.6528 (n/a) |
| SLBR-style self-calibrated mask | 0.5627 (0.1371) | 0.5343 (0.0990) | 0.5709 (0.1558) | 0.5961 (0.2131) | 0.5678 (0.2096) | 0.6420 (n/a) |
| Plain RGB U-Net | 0.5575 (0.1288) | 0.5267 (0.0858) | 0.5703 (0.1597) | 0.5906 (0.2030) | 0.5703 (0.2186) | 0.6410 (n/a) |
| All-empty mask | 0.4954 (0.0000) | 0.4857 (0.0000) | 0.5139 (0.0000) | 0.4889 (0.0000) | 0.4628 (0.0000) | 1.0000 (n/a) |

## Held-out breakdown

Each cell reports mIoU (foreground IoU).

| Model | thin text | low opacity | textured | logo | tiled | clean |
|---|---:|---:|---:|---:|---:|---:|
| Murmell local contrast | 0.5071 (0.0267) | 0.4963 (0.0248) | 0.5046 (0.0074) | 0.4980 (0.0212) | 0.4760 (0.0265) | 0.7522 (n/a) |
| Murmell residual supervision | 0.4985 (0.0239) | 0.4894 (0.0252) | 0.4890 (0.0156) | 0.4926 (0.0228) | 0.4699 (0.0305) | 0.6431 (n/a) |
| Murmell round 1 (RGB + high-pass U-Net) | 0.5111 (0.0363) | 0.5056 (0.0455) | 0.5138 (0.0295) | 0.5162 (0.0586) | 0.4828 (0.0438) | 0.7355 (n/a) |
| WDNet-style decomposition | 0.4942 (0.0048) | 0.4864 (0.0081) | 0.5031 (0.0060) | 0.4919 (0.0108) | 0.4663 (0.0102) | 0.6868 (n/a) |
| Split then Refine-style mask | 0.4986 (0.0121) | 0.4894 (0.0126) | 0.5119 (0.0056) | 0.4942 (0.0145) | 0.4678 (0.0109) | 0.7514 (n/a) |
| SLBR-style self-calibrated mask | 0.4990 (0.0133) | 0.4894 (0.0131) | 0.5112 (0.0090) | 0.4931 (0.0129) | 0.4683 (0.0133) | 0.7274 (n/a) |
| Plain RGB U-Net | 0.4945 (0.0110) | 0.4860 (0.0130) | 0.4961 (0.0123) | 0.4904 (0.0133) | 0.4659 (0.0161) | 0.6846 (n/a) |
| All-empty mask | 0.4953 (0.0000) | 0.4852 (0.0000) | 0.5374 (0.0000) | 0.4884 (0.0000) | 0.4640 (0.0000) | 1.0000 (n/a) |

Thin text: eroded single-instance text. Low opacity: 0 < opacity < 0.22. Textured: photo crops. Slices overlap; all marked held-out images are low opacity.

## Failure cases

Worst example by slice, for the first training seed. Regenerate a scene by passing its test seed and scene index to `src.hard_watermarks.make_scene`.

| Model | Split | Slice | Index | mIoU | Opacity | Photo | True pixels | Predicted pixels | Overlap |
|---|---|---|---:|---:|---:|---|---:|---:|---:|
| Murmell round 1 (RGB + high-pass U-Net) | hard | thin text | 102 | 0.4801 | 0.160 | grass | 384 | 267 | 0 |
| Murmell round 1 (RGB + high-pass U-Net) | hard | low opacity | 132 | 0.4604 | 0.183 | grass | 1345 | 10 | 4 |
| Murmell round 1 (RGB + high-pass U-Net) | hard | textured | 75 | 0.4440 | 0.455 | grass | 1834 | 0 | 0 |
| Murmell round 1 (RGB + high-pass U-Net) | hard | clean | 53 | 0.4977 | 0.000 | procedural | 0 | 75 | 0 |
| Murmell round 1 (RGB + high-pass U-Net) | held-out | thin text | 42 | 0.4859 | 0.102 | procedural | 439 | 22 | 0 |
| Murmell round 1 (RGB + high-pass U-Net) | held-out | low opacity | 55 | 0.4446 | 0.120 | procedural | 2290 | 368 | 102 |
| Murmell round 1 (RGB + high-pass U-Net) | held-out | textured | 128 | 0.4504 | 0.136 | gravel | 1593 | 33 | 0 |
| Murmell round 1 (RGB + high-pass U-Net) | held-out | clean | 61 | 0.4856 | 0.000 | brick | 0 | 471 | 0 |
| Murmell local contrast | hard | thin text | 102 | 0.4833 | 0.160 | grass | 384 | 163 | 0 |
| Murmell local contrast | hard | low opacity | 132 | 0.4590 | 0.183 | grass | 1345 | 0 | 0 |
| Murmell local contrast | hard | textured | 147 | 0.4457 | 0.270 | grass | 1808 | 5 | 3 |
| Murmell local contrast | hard | clean | 150 | 0.4917 | 0.000 | hubble_deep_field | 0 | 271 | 0 |
| Murmell local contrast | held-out | thin text | 149 | 0.4779 | 0.094 | brick | 67 | 705 | 2 |
| Murmell local contrast | held-out | low opacity | 55 | 0.4435 | 0.120 | procedural | 2290 | 314 | 90 |
| Murmell local contrast | held-out | textured | 108 | 0.4475 | 0.140 | brick | 1776 | 174 | 22 |
| Murmell local contrast | held-out | clean | 61 | 0.4693 | 0.000 | brick | 0 | 1006 | 0 |
| Murmell residual supervision | hard | thin text | 102 | 0.4854 | 0.160 | grass | 384 | 93 | 0 |
| Murmell residual supervision | hard | low opacity | 132 | 0.4598 | 0.183 | grass | 1345 | 2 | 2 |
| Murmell residual supervision | hard | textured | 75 | 0.4435 | 0.455 | grass | 1834 | 19 | 0 |
| Murmell residual supervision | hard | clean | 41 | 0.4953 | 0.000 | procedural | 0 | 155 | 0 |
| Murmell residual supervision | held-out | thin text | 1 | 0.4652 | 0.074 | brick | 246 | 1106 | 15 |
| Murmell residual supervision | held-out | low opacity | 55 | 0.4360 | 0.120 | procedural | 2290 | 225 | 49 |
| Murmell residual supervision | held-out | textured | 136 | 0.4464 | 0.079 | brick | 1042 | 957 | 24 |
| Murmell residual supervision | held-out | clean | 61 | 0.4734 | 0.000 | brick | 0 | 872 | 0 |

## ONNX CPU (1.20.1)

| Model | Runtime | Hard mIoU (foreground IoU) | Held-out mIoU (foreground IoU) | CPU ms/image |
|---|---|---:|---:|---:|
| Murmell local contrast | PyTorch | 0.6254 (0.2686) | 0.5243 (0.0248) | 11.49 |
| Murmell local contrast | ONNX Runtime | 0.6254 (0.2686) | 0.5243 (0.0248) | 5.02 |
| Murmell residual supervision | PyTorch | 0.6015 (0.2398) | 0.5062 (0.0252) | 10.10 |
| Murmell residual supervision | ONNX Runtime | 0.6015 (0.2398) | 0.5062 (0.0252) | 9.99 |

Maximum absolute probability difference: 1.4e-05; thresholded pixel disagreements across both splits and all seeds: 2.
