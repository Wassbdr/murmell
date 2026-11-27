"""poster.png (1280x720): image | ground truth | predicted mask | overlay, for the good, median and failure examples, with IoU."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]; T1 = ROOT / "task1_watermark_localisation"
sys.path.insert(0, str(T1))
from predict import load, overlay, predict  # noqa: E402

net, card = load(); ex = json.loads((T1 / "examples/examples.json").read_text()); m = card["metrics"]
fig, ax = plt.subplots(len(ex), 4, figsize=(12.8, 7.2), dpi=100)
for r, e in enumerate(ex):
    img = np.asarray(Image.open(T1 / f"examples/{e['name']}.png").convert("RGB"))
    gt = np.asarray(Image.open(T1 / f"examples/{e['name']}_mask.png")) > 127
    _, pm = predict(net, card, img)
    for c, (im, t) in enumerate([(img, "image"), (gt, "ground truth"), (pm, "predicted mask"), (overlay(img, pm), "overlay")]):
        ax[r, c].imshow(im, cmap="gray" if im.ndim == 2 else None); ax[r, c].set_xticks([]); ax[r, c].set_yticks([])
        if r == 0:
            ax[r, c].set_title(t, fontsize=12)
    ax[r, 0].set_ylabel(f"{e['name']}\n{e['kind']}, opacity {e['opacity']}", fontsize=10)
    ax[r, 3].text(1.03, 0.5, f"mIoU\n{e['miou']:.3f}", transform=ax[r, 3].transAxes, fontsize=14, va="center",
                  color="#2a7" if e["miou"] > 0.8 else "#c33")
fig.suptitle(f"Murmell task 1 - watermark localisation. Synthetic eval mIoU {m['miou_nested_per_kind']:.3f} "
             f"(500 images, 5 folds, thresholds chosen outside the tested fold, CPU)", fontsize=13)
fig.subplots_adjust(left=0.08, right=0.9, top=0.9, bottom=0.02, wspace=0.05, hspace=0.08)
out = ROOT / "showcase/poster.png"; fig.savefig(out, dpi=100); plt.close(fig)
Image.open(out).convert("RGB").resize((1280, 720)).save(out); print("poster", out)
