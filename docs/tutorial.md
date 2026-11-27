# Detect a visible watermark on your own image

From a clone with Python 3.10 or 3.11, install the pinned CPU dependencies and the package:

```sh
python -m pip install -r benchmarks/requirements.txt -e '.[app,onnx]'
murmell-detect path/to/photo.jpg -o mask.png
```

The PNG is single-channel and the same size as the image. White marks predicted watermark pixels, black marks
everything else. Use `--variant contrast` to try the second detector, or `--threads 1` to limit CPU use.
Only load a `--weights` checkpoint from a source you trust.

For multiple inputs, load the model once. Set PyTorch's thread count for your machine before calling it:

```python
from murmell import Detector, open_image

detector = Detector.load()
detector.predict(open_image("path/to/photo.jpg")).save("mask.png")
```

Try the [bundled example](../task1_watermark_localisation/examples/good.png) first if your mask is empty.
Faint or fine marks may not be detected. A mask is a prediction, not a certificate that an image is clean.

Start Gradio with `python task1_watermark_localisation/app.py`, then open <http://127.0.0.1:7860>.
For ONNX Runtime, run `murmell-export detector.onnx` and use the [model card](model_card.md#cpu-export)
for input format, threshold, and timings.
