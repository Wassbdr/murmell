"""Predict the watermark mask of one image with the synthetic-data model of src/train_synth.py.

    python predict.py IMAGE.png [--gt MASK.png] [--out OUTDIR]       -> OUTDIR/<name>_mask.png, OUTDIR/<name>_overlay.png
The image is resized to the model's input (256 px), the mask comes out as 128 x 128 cells and is resized back.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent
DEFAULT = ROOT / "weights" / "wm_unet_synth.json"


def load(card_path=DEFAULT):
    from .src.train_synth import build_net
    card = json.loads(Path(card_path).read_text())
    net = build_net(card["config"], weights=None)
    sd = torch.load(Path(card_path).parent / card["file"], map_location="cpu", weights_only=True)
    net.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in sd.items()}); net.eval()
    return net, card


@torch.no_grad()
def predict(net, card, img):
    """img: uint8 H x W x 3 -> (probability cells x cells, binary mask at H x W)."""
    S = card["config"]["size"]; H, W = img.shape[:2]
    x = np.asarray(Image.fromarray(img).convert("RGB").resize((S, S), Image.BILINEAR))
    p = torch.sigmoid(net(torch.from_numpy(x.copy()).permute(2, 0, 1)[None]))[0, 0].numpy()
    m = p >= card["threshold"]
    full = np.asarray(Image.fromarray((m * 255).astype(np.uint8)).resize((W, H), Image.NEAREST)) > 127
    return p, full


def overlay(img, mask, color=(255, 40, 80), alpha=0.55):
    out = img.astype(np.float32).copy(); c = np.array(color, np.float32)
    out[mask] = (1 - alpha) * out[mask] + alpha * c
    return out.clip(0, 255).astype(np.uint8)


def iou_vs(gt_path, mask):
    from src.metric import miou_image
    g = np.asarray(Image.open(gt_path).convert("L").resize(mask.shape[::-1], Image.NEAREST)) > 127
    return miou_image(mask, g)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("image"); ap.add_argument("--gt"); ap.add_argument("--out", default="predictions")
    ap.add_argument("--card", default=str(DEFAULT)); a = ap.parse_args()
    net, card = load(a.card)
    img = np.asarray(Image.open(a.image).convert("RGB"))
    p, m = predict(net, card, img)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); stem = Path(a.image).stem
    Image.fromarray((m * 255).astype(np.uint8)).save(out / f"{stem}_mask.png")
    Image.fromarray(overlay(img, m)).save(out / f"{stem}_overlay.png")
    msg = f"{stem}: watermarked fraction {m.mean():.3f} (threshold {card['threshold']})"
    if a.gt:
        msg += f", mIoU vs ground truth {iou_vs(a.gt, m):.4f}"
    print(msg + f" -> {out}/{stem}_mask.png, {stem}_overlay.png")


if __name__ == "__main__":
    main()
