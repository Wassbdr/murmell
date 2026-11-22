"""Write + validate a submission from binary test masks (test metadata order), and optionally upload it.
    python -m src.submit --masks work/test/<exp>_bin.npy --exp <exp> --cv 0.xxxx        (write + validate only)
    python -m src.submit --check submissions/<file>.pt                                    (validate an existing file)
    python -m src.submit --upload submissions/<file>.pt     (needs env WMLOC_API_KEY and WMLOC_BASE_URL; never printed)
Upload is a separate explicit step: the human decides who submits."""
import argparse
import os
import sys
import time

import numpy as np
import torch

from src.data import ROOT, names

TASK_ID = "34-watermark-localisation-image"


def validate(path):
    d = torch.load(path, weights_only=True)
    assert isinstance(d, dict) and set(d) == {"image_names", "masks"}, "keys"
    n, m = d["image_names"], d["masks"]
    exp = names("test")
    assert isinstance(n, list) and all(isinstance(x, str) for x in n), "names type"
    assert len(n) == len(set(n)) == 1020 and set(n) == set(exp), "names set"
    assert torch.is_tensor(m) and tuple(m.shape) == (1020, 128, 128), f"shape {tuple(m.shape)}"
    assert m.dtype in (torch.bool, torch.uint8) and set(torch.unique(m.to(torch.uint8)).tolist()) <= {0, 1}, "values"
    sz = os.path.getsize(path) / 2**20
    assert os.path.getsize(path) <= 20_000_000, f"size {sz:.1f}MiB > 20 MB (official participants doc)"
    frac = m.float().mean().item()
    print(f"VALID {path}: 1020 masks, dtype {m.dtype}, {sz:.1f} MB, mean watermarked fraction {frac:.3f}")
    return True


def write(masks, exp, cv, out=None):
    masks = np.asarray(masks)
    assert masks.shape == (1020, 128, 128)
    out = out or ROOT / "submissions" / f"{time.strftime('%H%M')}_{exp}_cv{cv}.pt"
    torch.save({"image_names": names("test"), "masks": torch.from_numpy(masks.astype(bool))}, out)
    validate(out)
    return out


def upload(path):
    import requests
    key, base = os.environ.get("WMLOC_API_KEY"), os.environ.get("WMLOC_BASE_URL")
    if not key or not base:
        sys.exit("set WMLOC_API_KEY and WMLOC_BASE_URL")
    validate(path)
    with open(path, "rb") as f:
        r = requests.post(f"{base}/submit/{TASK_ID}", headers={"X-API-Key": key},
                          files={"file": (os.path.basename(path), f, "application/octet-stream")}, timeout=(10, 120))
    try:
        body = r.json()
    except Exception:
        body = {"raw_text": r.text[:500]}
    print("HTTP", r.status_code, body)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--masks"); ap.add_argument("--exp"); ap.add_argument("--cv", default="na")
    ap.add_argument("--check"); ap.add_argument("--upload")
    a = ap.parse_args()
    if a.masks:
        write(np.load(a.masks), a.exp, a.cv)
    if a.check:
        validate(a.check)
    if a.upload:
        upload(a.upload)


if __name__ == "__main__":
    main()
