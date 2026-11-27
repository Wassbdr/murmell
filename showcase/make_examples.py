"""Pick the demo images from the synthetic EVAL set (seed 1) by their measured IoU, predict them, write
task1_watermark_localisation/examples/{good,median,failure}.png (+ _mask.png ground truth) and examples.json."""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

T1 = Path(__file__).resolve().parents[1] / "task1_watermark_localisation"
sys.path.insert(0, str(T1))
from predict import load, predict  # noqa: E402
from src.metric import miou_image  # noqa: E402
from src.synth_visible import make_item  # noqa: E402

res = json.loads((T1 / "results/wm_unet_synth_results.json").read_text()); cfg = res["config"]
iou = np.array(res["per_image_iou"]); order = np.argsort(iou)
meta_all = [make_item(cfg["eval_seed"], int(i), cfg["size"], cfg["cells"])[2] for i in range(len(iou))]
ok = [i for i in order if meta_all[i]["kind"] != "clean"]
# good = best text/logo case with visible coverage, median, failure = worst non-clean case
good = max((i for i in ok if meta_all[i]["kind"] in ("text", "logo") and meta_all[i]["coverage"] > 0.03), key=lambda i: iou[i])
pick = dict(good=int(good), median=int(ok[len(ok) // 2]), failure=int(ok[0]))
net, card = load(); out = T1 / "examples"; out.mkdir(exist_ok=True); meta = []
for name, i in pick.items():
    x, m, mt = make_item(cfg["eval_seed"], i, cfg["size"], cfg["cells"])
    Image.fromarray(x).save(out / f"{name}.png")
    Image.fromarray(m * 255).resize((cfg["size"], cfg["size"]), Image.NEAREST).save(out / f"{name}_mask.png")
    p, full = predict(net, card, x)
    g = np.asarray(Image.fromarray(m * 255).resize(full.shape[::-1], Image.NEAREST)) > 127
    meta.append(dict(name=name, eval_index=i, miou=round(miou_image(full, g), 4), **{k: mt[k] for k in mt if k != "index"}))
(out / "examples.json").write_text(json.dumps(meta, indent=1))
print(json.dumps(meta, indent=1))
