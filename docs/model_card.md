# Murmell invisible-watermark model card

## Intended use

The product predicts which 128 by 128 cells in an edited image still carry a TrustMark Q watermark.
It does not remove the mark, authenticate an owner or certify that an image is clean. The localizer
uses a 128 px RGB input and a ResNet-50 U-Net initialized from the official TrustMark decoder.

## Data and measurement

The public [measurement protocol and tables](../benchmarks/results.md) separate the hard and held-out
splits and report both mIoU and foreground IoU. The main corpus has 1,500 training images,
48 calibration images and 60 images per test split. Each split is balanced across full frames with
holes, rectangular pastes and irregular pastes. The held-out generator varies the photos and edit ranges.

[Sources and licences](../benchmarks/invisible-sources.md) list the CC0 and public-domain photos and
the official MIT model terms. No competition photos or messages were used.

| Seed-0 known-message recipe | Hard mIoU | Hard foreground IoU | Held-out mIoU | Held-out foreground IoU |
|---|---:|---:|---:|---:|
| U-Net with calibrated global regime | 0.6200 | 0.5993 | 0.3884 | 0.2109 |
| U-Net with bounded box search and MobileSAM | 0.6259 | 0.6124 | 0.4227 | 0.2703 |

The first row uses the mask recipe chosen for the product. Neither row measures the product's
unknown-message accuracy; both give the decoder the original message for scoring agreement.

## Confidence and failures

The product globally decodes a BCH message and checks calibrated soft agreement with its reconstructed
bits. If decoding fails or the confidence is below the threshold, it reports `no confident decode`
and returns an empty mask without running the U-Net. The threshold is selected on separate synthetic
marked and clean calibration scenes, not on the scored test splits. For the known-message benchmark,
agreement uses the original message, and low global agreement can still trigger a local search.
At the fitted 0.28 confidence threshold, 14 of 48 marked calibration scenes yielded the correct
message and none of 48 clean scenes passed. These are calibration counts, not test accuracy.

Small splices may not decode globally even when a region carries the mark. Crop, resize and JPEG
may weaken the signal. Box masks can include pixels outside irregular pastes, and thresholded
probabilities can miss edit holes. The generator labels where the mark was applied, not exactly where
every bit remains recoverable. The four source photos and TrustMark Q setup do not establish accuracy
on arbitrary photographs or other watermark decoders.

On very wide or tall inputs, TrustMark Q's official preprocessing decodes a central region rather than
the whole image. If that decode is confident, the product localises within the decoded region and
places its mask back on the original cell grid. Cells outside the decoded view are not assessed.
The showcase is a wide splice that illustrates this case; its centre still has small edited holes
that the predicted mask can miss.

## Rebuild and CPU export

Run `make weights` to rebuild the seed-0 detector and its calibration card. The checkpoint
is stored outside the package under `~/.cache/murmell/`. The reference checkpoint is 97,405,542 bytes,
with SHA-256 `15ca4f20bd244da31e4a84b39169e6fa9e790443bf19244bd764a1f699e88ac8`.
The command records the digest in the calibration card. Do not use weights from an untrusted source.

`murmell-export detector.onnx` exports just the U-Net and its RGB normalization and sigmoid.
The graph takes float32 NCHW RGB in `[0, 1]` with a dynamic batch axis and emits cell probabilities.
Decoder, gate, box search and overlay run outside the graph.

The U-Net was timed on an Intel Xeon CPU with 8 threads, batch 1 at 128 px. Each number is the median
of 50 forwards after five warm-up calls. It includes input normalization and sigmoid, but excludes
file I/O, resizing, global decoding, and post-processing.

| Runtime | ms/image |
|---|---:|
| PyTorch 2.5.1+cpu | 81.27 |
| ONNX Runtime 1.20.1 | 16.26 |

Maximum absolute difference across the 60 hard-set probability maps was `2.1457672119140625e-06`.
The product uses the calibration-selected regime-only mask, not the benchmark's MobileSAM chain.
