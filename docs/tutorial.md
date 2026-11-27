# Find a watermark on your own image

Use Python 3.10 or 3.11. In a clone of this repository:

```sh
python -m pip install -r benchmarks/requirements.txt -e '.[app,onnx]'
make weights
murmell-detect path/to/edited.jpg --method trustmark -o mask.png
```

The rebuild downloads Adobe's official Q model and trains the localizer on CPU. The PNG contains 128 by 128 cells.
White denotes predicted watermarked cells. For an overlay on the original image, resize the mask with nearest
neighbour interpolation and use it as an alpha mask.

Use `--method auto` to try the installed decoders. The packaged localizer currently supports TrustMark Q.
If the decoder cannot recover a confident message, the command prints `no confident decode` and writes an
empty mask. This is an abstention rather than evidence that the image was never marked.

```python
from PIL import Image
from murmell import ProvenanceDetector

detector = ProvenanceDetector.load()
with Image.open("path/to/edited.jpg") as image:
    result = detector.predict(image)
result.mask.save("mask.png")
print(result.regime)
```

Open `python task1_watermark_localisation/app.py` at <http://127.0.0.1:7860> to see the overlay.
Use `murmell-export detector.onnx` to export the U-Net, not the decoder, confidence gate or box search.
See the [model card](model_card.md) for the mask format and known failure modes.
