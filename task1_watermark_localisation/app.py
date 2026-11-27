"""Gradio demo: upload an image, see the predicted watermark mask over it.

    python app.py            -> http://127.0.0.1:7860   (model: weights/wm_unet_synth.pt, CPU)
"""
import json
from pathlib import Path

import gradio as gr
import numpy as np
from PIL import Image

from predict import load, overlay, predict

ROOT = Path(__file__).resolve().parent
EX = ROOT / "examples"
net, card = load()


def run(img, name=None):
    if img is None:
        return None, None, ""
    p, m = predict(net, card, img)
    info = f"### watermarked area: {100 * m.mean():.1f} % of the image (threshold {card['threshold']})"
    gt = EX / f"{name}_mask.png" if name else None
    if gt and gt.is_file():
        from src.metric import miou_image
        g = np.asarray(Image.open(gt).convert("L").resize(m.shape[::-1], Image.NEAREST)) > 127
        info += f" · mIoU vs exact ground truth: **{miou_image(m, g):.3f}** ({name} case)"
    return overlay(img, m), (m * 255).astype(np.uint8), info


def examples():
    meta = json.loads((EX / "examples.json").read_text()) if (EX / "examples.json").is_file() else []
    return [[str(EX / f"{e['name']}.png"), e["name"]] for e in meta]


with gr.Blocks(title="Watermark localisation") as demo:
    gr.Markdown("## Watermark localisation (Murmell, task 1)\nU-Net trained on CPU on seeded **synthetic** visible "
                f"watermarks, computed live. Synthetic eval mIoU {card['metrics']['miou_nested_per_kind']:.3f} (5 folds, "
                "thresholds chosen outside the tested fold).")
    with gr.Row():
        with gr.Column(scale=1):
            inp = gr.Image(type="numpy", label="input image", height=250)
            btn = gr.Button("Localise watermark", variant="primary", elem_id="run")
        out = gr.Image(type="numpy", label="predicted mask over the image", height=250, scale=1)
        msk = gr.Image(type="numpy", label="predicted mask", height=250, scale=1)
    info = gr.Markdown(elem_id="info")
    name = gr.Textbox(visible=False)
    btn.click(run, [inp, name], [out, msk, info])
    inp.upload(lambda: "", None, name)
    gr.Examples(examples(), [inp, name], label="synthetic examples: click one, then Localise")

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)
