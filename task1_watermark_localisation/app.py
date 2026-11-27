"""Gradio demo for the bundled CPU detector."""
import json
from functools import lru_cache
from pathlib import Path

import gradio as gr
import numpy as np
import torch
from numpy.typing import NDArray
from PIL import Image
from predict import overlay
from src.metric import miou_image

from murmell import Detector

ROOT = Path(__file__).resolve().parent
EX = ROOT / "examples"


@lru_cache(maxsize=1)
def detector() -> Detector:
    return Detector.load()


def run(
    img: NDArray[np.uint8] | None, name: str | None = None
) -> tuple[NDArray[np.uint8] | None, NDArray[np.uint8] | None, str]:
    if img is None:
        return None, None, ""
    model = detector()
    mask = np.asarray(model.predict(Image.fromarray(img)))
    m = mask > 0
    info = f"### {name or 'Upload'}: watermarked area {100 * m.mean():.1f}% (threshold {model.threshold:.3f})"
    gt = EX / f"{name}_mask.png" if name else None
    if gt and gt.is_file():
        with Image.open(gt) as target:
            g = np.asarray(target.convert("L").resize(m.shape[::-1], Image.Resampling.NEAREST)) > 127
        info += f". mIoU vs exact ground truth: **{miou_image(m, g):.3f}**."
    return overlay(img, m), mask, info


def examples() -> list[list[str]]:
    meta = json.loads((EX / "examples.json").read_text()) if (EX / "examples.json").is_file() else []
    return [[str(EX / f"{e['name']}.png"), e["name"]] for e in meta]


with gr.Blocks(title="Watermark localisation", analytics_enabled=False) as demo:
    gr.Markdown("## Murmell: visible-watermark localisation\n"
                "The bundled RGB + high-pass U-Net predicts a mask on CPU. "
                "Faint marks and unfamiliar textures can be missed.")
    with gr.Row():
        with gr.Column(scale=1):
            inp = gr.Image(type="numpy", label="input image", height=250, elem_id="demo-input")
            btn = gr.Button("Localise watermark", variant="primary", elem_id="run")
        out = gr.Image(type="numpy", label="predicted mask over the image", height=250, scale=1)
        msk = gr.Image(type="numpy", label="predicted mask", height=250, scale=1)
    info = gr.Markdown(elem_id="info")
    name = gr.Textbox(visible=False)
    btn.click(run, [inp, name], [out, msk, info])
    inp.upload(lambda: "", None, name)
    gr.Examples(examples(), [inp, name], elem_id="demo-examples", cache_examples=False,
                label="Synthetic examples: click one, then Localise")

if __name__ == "__main__":
    torch.set_num_threads(min(8, torch.get_num_threads()))
    demo.launch(server_name="127.0.0.1", server_port=7860, show_error=True)
