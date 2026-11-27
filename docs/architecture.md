# Architecture

- **One inference path.** The CLI and Gradio call `murmell.Detector`. It owns resizing, normalization,
  checkpoint construction, thresholding and mask alignment. ONNX wraps that same model before thresholding.
- **Full checkpoints.** Both state dictionaries are packed with the wheel for offline inference as exact float32
  weights and load on CPU with `weights_only=True`.
- **Default selection.** Round-1 seed 0 at 0.900 beats the historical demo on held-out mIoU. The old demo has higher
  foreground IoU; retain its card for comparison. The two new recipes do not justify a default switch.
- **Evaluation isolation.** Each seed trains on the hard generator. Validation selects the threshold. The held-out
  generator uses separate seed, photos, fonts, opacity and layouts. Neither test split selects model thresholds.
- **Rebuild scopes.** `make bench` retrains the seven recipes. `make shipped` scores bundled checkpoints and measures
  CPU export. `make showcase` rebuilds the synthetic reports and app recording without retraining task 1.
  The historical competition tools do not run as part of these commands.
