# Changelog

## 0.3.0

- Localise invisible TrustMark Q provenance watermarks and abstain when a message cannot be recovered.
- Rebuild the public Task 1 benchmark with official embedders and separate known-message scoring.
- Export the localizer to ONNX and archive the visible detector under `experiments/visible/`.

## 0.2.0

- Ship the retrained RGB + high-pass detector with a CLI, Python API, Gradio app and ONNX export.
- Bundle seed-0 weights for the default and opt-in local-contrast models.
- Compare five reference recipes and two experiments on fixed hard and held-out data, reporting mIoU and foreground IoU.
- Add CPU build and test commands, a Python CI matrix and reproducible showcase assets.
- Document and regenerate the synthetic model-equality and unlearning benchmarks.
