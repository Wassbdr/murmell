# Visible-watermark benchmark

```sh
python -m pip install -r benchmarks/requirements.txt -e .
make bench
```

The benchmark trains on CPU from randomly initialized weights, writes [results.md](results.md), and saves the
per-seed results and weights in `task1_watermark_localisation/work/bench`. To change the run size, pass arguments
with `make bench BENCH_ARGS="--help"`. The dependency pins target Linux, Python 3.10 or 3.11.
The full run uses the available CPU threads and trains three seeds. For an exploratory run of just the plain
U-Net, use `make bench BENCH_ARGS="--models unet --seeds 0 --output work/unet-results.md"`.

Run `make bench-audit` to regenerate the [label counts and signal statistics](data-audit.md) and the sample
grid. Foreground IoU in the score tables averages only images with a nonempty target; predicting all background
gets zero on this metric regardless of how many clean images are in the split.

Both Murmell experiments export ONNX models during the benchmark. To export an existing checkpoint:

```sh
cd task1_watermark_localisation
python -m src.export_onnx work/bench/contrast-0.pt work/bench/contrast-0.onnx
```

Add `--residual` for a residual-supervision checkpoint. The graph accepts float32 RGB in `[0, 1]`, arranged
as NCHW at the training resolution, and outputs a probability per pixel. The validation threshold for each
checkpoint is in `work/bench/runs.json`. The benchmark compares the exported mask and probability to PyTorch.

## Data

No competition data, CLWD or LVW are used. The [WDNet repository](https://github.com/MRUIL/WDNet)
links to the CLWD archive without dataset reuse or redistribution terms. The [SLBR repository](https://github.com/bcmi/SLBR-Visible-Watermark-Removal)
directs users to contact the LVW authors; a download listing alone does not specify a licence.

The hard generator overlays procedural logos, tiled text and thinned single-instance text on seeded scenes.
It mixes procedural backgrounds and photo crops, and includes clean images, opaque text distractors and JPEG
compression. Masks are defined by pre-opacity coverage of at least 50%. The hard test has its own seeds.
The held-out test uses disjoint photos, watermark text fonts, opacity ranges and layouts: larger angles, staggered
tiling and centered single marks in one of three horizontal bands. Logo primitives and ink colors are shared.

Background photos ship with [scikit-image](https://scikit-image.org/docs/stable/api/skimage.data.html):
grass (CC0, DeviantArt); Hubble eXtreme Deep Field (NASA, public domain); gravel and brick (CC0, CC0Textures).
The hard splits use grass and Hubble, and the held-out split uses gravel and brick. There are only four source
photos, so these results do not establish accuracy on arbitrary photographs. Text uses Matplotlib's bundled
DejaVu fonts under the Bitstream Vera / DejaVu licence.

## References

| Row | What was run |
|---|---|
| Plain RGB U-Net | A three-downsample U-Net with bilinear decoding and skip connections. |
| Murmell round 1 | The repository's MobileNetV2 U-Net with RGB and 5-pixel high-pass channels. Resized for this benchmark and retrained without ImageNet weights. |
| [SLBR](https://arxiv.org/abs/2108.03581)-style | Small self-calibrated mask head: supervised intermediate masks at three scales and mask-conditioned feature refinement. |
| [WDNet](https://arxiv.org/abs/2012.07616)-style | Small decomposition head: predicts a mask, alpha and ink, inverts the compositing equation and refines the coarse mask using the estimated background. |
| [Split then Refine](https://arxiv.org/abs/2012.07007)-style | Small two-stage mask head: channel attention splits coarse mask and background heads; a second U-Net refines the mask from the first stage's outputs. |

The three paper-inspired rows are independent CPU implementations of localization ideas, not scores from the
authors' checkpoints or replications of their full removal systems. The official repositories do not state
clear reuse terms for the code and checkpoints we would need. Removal branches, adversarial and perceptual losses,
and published removal scores are not compared. The reference rows share the same images, update budget and
segmentation loss; the paper-inspired rows also supervise their intermediate outputs.
