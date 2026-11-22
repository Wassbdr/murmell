#!/usr/bin/env python3
"""Evaluate track 3 candidates against one frozen repaired champion on box1."""

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F

BASE = Path("/workspace")
WORK = BASE / "unlearning"
sys.path.insert(0, str(BASE))
sys.path.insert(0, "/workspace/lina_t3")
import task_template as tt
from t3_consist import alter

tt.DEVICE = "cpu"
DEV = "cuda"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@torch.inference_mode()
def logits(model, images, batch_size=100, transform=None, seed=None):
    output = []
    generator = torch.Generator(device=DEV).manual_seed(seed) if seed is not None else None
    for start in range(0, len(images), batch_size):
        pixels = images[start:start + batch_size].to(DEV)
        if transform is not None:
            pixels = transform(pixels, generator)
        with torch.autocast("cuda", dtype=torch.float16):
            output.append(model(pixels * 2 - 1).float().cpu())
    return torch.cat(output)


def kl(reference, candidate):
    return F.kl_div((candidate / 2).log_softmax(1), (reference / 2).softmax(1),
                    reduction="batchmean")


def moments(reference, candidate, labels):
    loss = F.cross_entropy(candidate, labels, reduction="none")
    return [float(loss.mean()), float(loss.std())]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--heldout-start", type=int, default=650)
    parser.add_argument("--heldout-end", type=int, default=852)
    parser.add_argument("--batch-size", type=int, default=100)
    args = parser.parse_args()

    original = tt.load_model(BASE / "model.pt").to(DEV).eval()
    baseline = tt.build_model().to(DEV).eval()
    baseline.load_state_dict(torch.load(args.baseline, map_location="cpu", weights_only=True), strict=True)
    for parameter in list(original.parameters()) + list(baseline.parameters()):
        parameter.requires_grad_(False)

    forget = ((tt.preprocess(tt.load_forget_images(BASE / "forget_images.npz")) + 1) / 2).contiguous()
    with np.load(WORK / "cache/gen/gen_test.npz", allow_pickle=False) as pack:
        generated = torch.from_numpy(pack["images"][args.heldout_start:args.heldout_end]).permute(
            0, 3, 1, 2).float().div(255).contiguous()
        classes = torch.from_numpy(pack["cls"][args.heldout_start:args.heldout_end])

    print("Precomputing original and baseline references", flush=True)
    original_new = logits(original, generated, args.batch_size)
    original_forget = logits(original, forget, args.batch_size)
    labels = original_forget.argmax(1)
    baseline_new = logits(baseline, generated, args.batch_size)
    baseline_exact = logits(baseline, forget, args.batch_size)
    baseline_altered = logits(baseline, forget, args.batch_size,
                              transform=lambda x, g: alter(x, g, 0.7), seed=77)
    original_accuracy = float((original_new.argmax(1)[classes >= 0] == classes[classes >= 0]).float().mean())
    baseline_ce_exact = moments(original_forget, baseline_exact, labels)
    baseline_ce_altered = moments(original_forget, baseline_altered, labels)

    def measure(new, exact, altered):
        retention_kl = float(kl(original_new, new))
        student_accuracy = float((new.argmax(1)[classes >= 0] == classes[classes >= 0]).float().mean())
        return {
            "neuves": {
                "kl_orig_eleve": retention_kl,
                "accord": float((original_new.argmax(1) == new.argmax(1)).float().mean()),
                "acc_orig": original_accuracy,
                "acc_eleve": student_accuracy,
                "retain_estime": min(1.0, student_accuracy / max(original_accuracy, 1e-6)) / (1 + retention_kl),
            },
            "oubli": {
                "kl_base_eleve_exactes": float(kl(baseline_exact, exact)),
                "kl_base_eleve_alterees": float(kl(baseline_altered, altered)),
                "ce_base": baseline_ce_exact,
                "ce_eleve": moments(original_forget, exact, labels),
                "ce_base_alt": baseline_ce_altered,
                "ce_eleve_alt": moments(original_forget, altered, labels),
                "accord_exactes": float((baseline_exact.argmax(1) == exact.argmax(1)).float().mean()),
            },
        }

    report = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "protocol": {
            "heldout_generated": [args.heldout_start, args.heldout_end],
            "forget_images": len(forget),
            "alter_seed": 77,
            "alter_strength": 0.7,
            "note": "Common frozen baseline metrics, not a synthetic official score",
        },
        "baseline": {
            "name": args.baseline.stem,
            "path": str(args.baseline),
            "sha256": sha256(args.baseline),
            "metrics": measure(baseline_new, baseline_exact, baseline_altered),
        },
        "candidates": {},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        temporary = args.output.with_suffix(".partial.json")
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(args.output)

    save()
    for specification in args.candidate:
        name, separator, raw_path = specification.partition("=")
        if not separator or not name:
            raise ValueError("Candidates must use NAME=PATH")
        path = Path(raw_path)
        if not path.is_file():
            report["candidates"][name] = {"path": str(path), "status": "missing"}
            save()
            continue
        print("Evaluating", name, flush=True)
        student = tt.build_model().to(DEV).eval()
        student.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
        new = logits(student, generated, args.batch_size)
        exact = logits(student, forget, args.batch_size)
        altered = logits(student, forget, args.batch_size,
                         transform=lambda x, g: alter(x, g, 0.7), seed=77)
        metrics = measure(new, exact, altered)
        report["candidates"][name] = {
            "path": str(path),
            "sha256": sha256(path),
            "status": "success",
            "metrics": metrics,
            "delta_vs_baseline": {
                "retain_estime": metrics["neuves"]["retain_estime"] - report["baseline"]["metrics"]["neuves"]["retain_estime"],
                "ce_exact_mean": metrics["oubli"]["ce_eleve"][0] - baseline_ce_exact[0],
                "ce_altered_mean": metrics["oubli"]["ce_eleve_alt"][0] - baseline_ce_altered[0],
            },
        }
        save()
        print(json.dumps({"name": name, **report["candidates"][name]["delta_vs_baseline"],
                          "kl_exact": metrics["oubli"]["kl_base_eleve_exactes"],
                          "kl_altered": metrics["oubli"]["kl_base_eleve_alterees"]}), flush=True)
        del student, new, exact, altered
        torch.cuda.empty_cache()
    print("Saved", args.output, flush=True)


if __name__ == "__main__":
    main()
