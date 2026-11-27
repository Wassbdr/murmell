# Architecture

- **Public corpus.** Official embedders watermark images from a four-photo licensed corpus. Edits carry a
  pixel-support mask through crop and resize; support is pooled to 128 by 128 cells. The held-out generator
  changes the source photos and edit ranges. The scoring code is the competition metric.
- **Known-message benchmark.** The global soft agreement with the true message gates full-frame and local
  localisation. A TrustMark-initialised ResNet-50 U-Net predicts cell probabilities. Local proposals use
  at most 24 decoded crops; MobileSAM supplies optional snap candidates. No test labels fit thresholds.
- **Unknown-message product.** The global decoder checks the BCH message and a confidence threshold.
  Failed decoding produces an empty mask and an abstention. On success, the U-Net consumes resized RGB
  and any enabled local box search scores the recovered message. The packaged inference has no MobileSAM dependency.
  If the official TrustMark decoder crops to a central view, the product localises only within that view and
  projects the mask onto the original cell grid.
- **Export.** The ONNX graph takes float32 RGB in NCHW layout in `[0, 1]` and returns cell probabilities.
  Global decoding, thresholding and local box search remain outside the graph.
- **Scope.** Competition training assets are not needed by the public corpus, benchmark or product.
  The distinct visible-watermark implementation is preserved under `experiments/visible/`.
