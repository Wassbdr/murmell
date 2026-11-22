"""Compare real checkpoints against the same newly generated views and gold ensemble."""
import argparse
import json
from pathlib import Path
import sys

import torch

from calibrate_losses import split_images
from three_term_unlearning import crop_reference, fixed_views, retained_crops
from train_public_proxy import assess


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("gold", type=Path)
    parser.add_argument("models", nargs="+", type=Path)
    args = parser.parse_args()
    root = Path.cwd()
    sys.path.insert(0, str(root))
    import unlearn as recipe

    gold = torch.load(args.gold, map_location=recipe.DEV, weights_only=True)
    y = gold["y"].long()
    _, valid = split_images(y)
    images = recipe.load_images(root.parent / "forget_images.npz")
    original = recipe.load_model(root.parent / "model.pt").requires_grad_(False)
    models = {p.stem: recipe.load_model(p).requires_grad_(False) for p in args.models}
    centers = gold["members"][1::2]
    altered = gold.get("altered_members", gold["members"])[1::2]
    moments = gold["individual_moments"][1::2]
    trials = []
    for seed in (903247, 908171):
        views = fixed_views(recipe, original, images, valid, seed)
        crops, info = retained_crops(recipe, original, images, y, valid, seed + 451, 4, 20, True)
        if crops is None:
            raise RuntimeError("Missing natural controls")
        controls = crop_reference(recipe, original, crops)
        scores = {}
        for name, model in models.items():
            scores[name] = assess(recipe, model, views, controls, y, centers, moments, altered)
        trials.append({"seed": seed, "images": len(valid), "controls": info, "models": scores})
        print("MESURES", seed, {k: round(v["quality"], 6) for k, v in scores.items()}, flush=True)
        del views, controls, crops
    partial = args.output.with_suffix(".partial")
    partial.write_text(json.dumps({"gold": str(args.gold), "trials": trials}, indent=2) + "\n")
    partial.replace(args.output)


if __name__ == "__main__":
    main()
