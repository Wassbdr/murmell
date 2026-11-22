"""Tâche 3 : jeu de contrôle synthétique. Génère des images neuves dans le style du jeu (photo cinéma, lumière dorée)
pour les 45 classes identifiées et pour des concepts voisins (les 55 autres classes sont inconnues : on couvre large).
Sortie : ~/t3/gen/gen_{tag}.npz  images uint8 [N,224,224,3], concept (indice), cls (id de classe connu ou -1).
    python t3_gen.py --per 24 --tag a --seed 0
"""
import argparse, json, random, sys
from pathlib import Path
import numpy as np, torch
from PIL import Image

KNOWN = {0: "ceramic fox figurine", 1: "owl figurine", 2: "rabbit figurine", 3: "heron bird figurine", 4: "turtle figurine",
         5: "hanging lantern", 6: "camping lantern", 7: "old kerosene oil lantern", 8: "stone garden lantern", 12: "glass teapot", 15: "wild mushroom",
         19: "aloe vera plant", 21: "agave plant", 23: "rope suspension bridge", 25: "old footbridge over a river", 28: "lighthouse on a rocky coast",
         29: "sailboat on the sea", 30: "wooden rowboat", 31: "canoe on a lake", 32: "fishing boat", 33: "small dinghy boat on calm water",
         35: "wicker chair", 36: "folding chair", 37: "armchair", 38: "rocking chair", 39: "windmill in a field", 40: "wind turbine", 41: "watermill",
         42: "knitted wool scarf", 43: "crochet blanket", 44: "woven basket", 45: "embroidered cushion", 51: "leather boots", 52: "sneakers", 53: "wooden clogs",
         57: "loaf of bread", 69: "hourglass", 72: "kite flying", 81: "totem pole", 83: "pagoda", 85: "beehive", 86: "birdhouse", 88: "origami paper crane",
         92: "ship anchor", 93: "sundial"}
EXTRA = ["cat figurine", "frog figurine", "dog figurine", "elephant figurine", "horse figurine", "bear figurine", "candle", "chandelier", "street lamp", "table lamp", "paper lantern",
         "copper kettle", "ceramic mug", "teacup", "coffee pot", "glass vase", "wine glass", "glass bottle", "cactus", "bonsai tree", "fern plant", "succulent plant", "potted flower",
         "sunflower", "pine cone", "acorn", "autumn leaf", "pumpkin", "apple", "stone arch bridge", "castle tower", "water tower", "church bell tower", "clock tower", "stone well",
         "ferry boat", "kayak", "paddle boat", "wooden pier", "buoy at sea", "park bench", "wooden stool", "sofa", "hammock", "wooden swing", "barn", "silo", "greenhouse", "mailbox",
         "wool sweater", "wool hat", "knitted gloves", "wool socks", "ball of yarn", "straw hat", "leather bag", "sandals", "slippers", "high heel shoes", "rubber rain boots",
         "croissant", "pretzel", "bagel", "cupcake", "pie", "cheese wheel", "pocket watch", "wall clock", "compass", "telescope", "globe", "magnifying glass", "old key", "padlock",
         "hot air balloon", "paper airplane", "paper boat", "pinwheel", "wind chime", "weather vane", "stone statue", "obelisk", "fountain", "birdbath", "scarecrow", "dog house",
         "tree house", "wooden fence gate", "bicycle", "wheelbarrow", "watering can", "garden gnome", "ship wheel", "ship bell", "rope coil", "fishing net", "life buoy", "seashell",
         "violin", "guitar", "drum", "old book", "typewriter", "vintage camera", "umbrella", "feather", "bird nest", "snow globe", "music box", "chess piece", "wooden toy train",
         "ceramic bowl", "wooden spoon", "cast iron pan", "bell", "wooden barrel", "wagon wheel", "stone tower ruin", "tent", "campfire", "wooden sign post", "spinning top", "abacus"]
SCENES = ["on a wooden windowsill by a window", "on a wooden table in a cozy room with a warm lamp", "in a garden among plants", "on stone steps outdoors", "at sunset, backlit by the low sun",
          "in a misty landscape at dawn", "seen from above, aerial view", "close-up macro shot", "small in the distance in a wide landscape", "on a porch in the evening",
          "by the water at dusk", "in a dim room, dramatic shadows", "in a field at golden hour", "on a shelf with soft window light", "in the forest, dappled light", "against the sky at twilight"]
STYLE = ["cinematic photo, golden hour light, shallow depth of field, 35mm film, moody", "cinematic photograph, warm backlight, bokeh, film grain", "moody cinematic photo, soft warm light, shallow focus",
         "atmospheric photo, dusk light, cinematic color grading"]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--per", type=int, default=24); ap.add_argument("--tag", default="a"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=2); ap.add_argument("--bs", type=int, default=8)
    a = ap.parse_args()
    from diffusers import AutoPipelineForText2Image
    pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16").to("cuda")
    pipe.set_progress_bar_config(disable=True); pipe.vae.enable_slicing()
    concepts = [(c, n) for c, n in KNOWN.items()] + [(-1, n) for n in EXTRA]
    rng = random.Random(a.seed); jobs = []
    for ci, (c, name) in enumerate(concepts):
        k = a.per * (2 if c >= 0 else 1)
        for _ in range(k):
            jobs.append((ci, c, f"{rng.choice(STYLE)} of a {name}, {rng.choice(SCENES)}"))
    rng.shuffle(jobs)
    out = Path.home() / "t3/gen"; out.mkdir(exist_ok=True)
    imgs, cons, cls = [], [], []
    g = torch.Generator("cuda").manual_seed(a.seed)
    for i in range(0, len(jobs), a.bs):
        b = jobs[i:i + a.bs]
        res = pipe(prompt=[j[2] for j in b], num_inference_steps=a.steps, guidance_scale=0.0, height=512, width=512, generator=g).images
        for im, j in zip(res, b):
            imgs.append(np.asarray(im.resize((224, 224), Image.Resampling.LANCZOS))); cons.append(j[0]); cls.append(j[1])
        if (i // a.bs) % 50 == 0: print(i, len(jobs), flush=True)
    np.savez(out / f"gen_{a.tag}.npz", images=np.stack(imgs), concept=np.array(cons), cls=np.array(cls))
    (out / f"gen_{a.tag}.json").write_text(json.dumps({"concepts": [n for _, n in concepts], "prompts": [j[2] for j in jobs]}))
    # planche d'aperçu
    s = 112; cols = 12; sheet = Image.new("RGB", (cols * s, 6 * s))
    for k in range(cols * 6): sheet.paste(Image.fromarray(imgs[k]).resize((s, s)), ((k % cols) * s, (k // cols) * s))
    sheet.save(out / f"gen_{a.tag}_sheet.jpg", quality=85); print("DONE", len(imgs), flush=True)


if __name__ == "__main__":
    main()
