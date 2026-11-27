"""Show the surviving regions of an invisible TrustMark watermark."""

from functools import lru_cache
from pathlib import Path

import gradio as gr
import numpy as np
import torch
from numpy.typing import NDArray
from PIL import Image

from murmell import ProvenanceDetector

EXAMPLE = Path(__file__).resolve().parent / "examples" / "edited.png"


@lru_cache(maxsize=1)
def detector() -> ProvenanceDetector:
    return ProvenanceDetector.load()


def overlay(image: Image.Image, mask: Image.Image) -> Image.Image:
    color = Image.new("RGB", image.size, (0, 250, 186))
    alpha = mask.resize(image.size, Image.Resampling.NEAREST).point(lambda p: p * 0.48)
    return Image.composite(color, image, alpha)


def run(
    img: NDArray[np.uint8] | None, method: str = "auto",
) -> tuple[Image.Image | None, Image.Image | None, str]:
    if img is None:
        return None, None, ""
    image = Image.fromarray(img).convert("RGB")
    try:
        result = detector().predict(image, method)
    except (OSError, ValueError, RuntimeError) as exc:
        return None, None, str(exc)
    if result.method is None:
        return image, result.mask, "No confident decode. An empty mask does not prove the image is unmarked."
    area = float(np.asarray(result.mask).mean() / 255)
    info = f"**{result.method}** · {result.regime} · predicted watermarked cells: {area:.1%}"
    return overlay(image, result.mask), result.mask, info


with gr.Blocks(title="Invisible provenance watermark", analytics_enabled=False) as demo:
    gr.Markdown("## Where does the provenance watermark survive?\n"
                "Check a cropped, spliced, or inpainted image. Teal shows the predicted watermarked cells.")
    with gr.Row():
        with gr.Column():
            inp = gr.Image(type="numpy", label="Edited image", height=250, elem_id="demo-input")
            method = gr.Dropdown(["auto", "trustmark"], value="auto", label="Decoder")
            btn = gr.Button("Localise watermark", variant="primary", elem_id="run")
        out = gr.Image(label="Predicted regions over the image", height=250)
        mask = gr.Image(label="128 by 128 mask", height=250)
    info = gr.Markdown(elem_id="info")
    btn.click(run, [inp, method], [out, mask, info])
    if EXAMPLE.is_file():
        gr.Examples([[str(EXAMPLE)]], [inp], elem_id="demo-examples", label="Edited public example")


if __name__ == "__main__":
    torch.set_num_threads(min(8, torch.get_num_threads()))
    demo.launch(server_name="127.0.0.1", server_port=7860, show_error=True)
