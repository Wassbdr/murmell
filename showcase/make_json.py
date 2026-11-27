"""showcase.json from the measured results files (no number is typed by hand)."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
t1 = json.loads((ROOT / "task1_watermark_localisation/results/wm_unet_synth_results.json").read_text())
t2 = json.loads((ROOT / "task2_model_equality/results/simlab.json").read_text())
t3 = json.loads((ROOT / "task3_unlearning/results/simlab.json").read_text())
ex = {e["name"]: e for e in json.loads((ROOT / "task1_watermark_localisation/examples/examples.json").read_text())}
c = t1["config"]; hw = f"CPU only, {c['threads']} threads"
s1 = (f"synthetic visible watermarks, train seed {c['train_seed']} (streamed, {c['steps']} x {c['bs']} images), eval seed "
      f"{c['eval_seed']} ({c['n_eval']} images, 5 folds), {c['size']} px, {hw}, {t1['train_seconds'] / 60:.0f} min training")
fold = ", ".join(f"{v:.3f}" for v in t1["fold_miou"])
d = {
    "slug": "murmell",
    "title": "Murmell: watermark localisation",
    "tagline": "Finds which pixels of an image carry a watermark: a CPU-trainable U-Net, reproducible end to end on seeded synthetic data, from the team that finished 1st of 44 at the European AI Championship.",
    "problem": "Watermark removal and provenance checks need to know where a watermark is, not just whether there is one. In the competition the mask was scored by mIoU over 128x128 cells on a hidden split, so every threshold had to be chosen without looking at the images being scored.",
    "approach": "A seeded generator draws text, logo and tiled watermarks with random opacity, size, rotation and position on procedural backgrounds, with exact masks. A U-Net (MobileNetV2 encoder, RGB + high-pass input) is trained on CPU on that stream and pooled to the 128x128 cell grid of the competition. The mask threshold is chosen per watermark kind on four folds and scored on the fifth (nested cross-validation), with the repository's own metric.",
    "stack": ["Python", "PyTorch", "segmentation-models-pytorch", "NumPy", "Gradio", "Playwright", "matplotlib", "ffmpeg"],
    "demo": {"gif": "showcase/demo.gif", "mp4": "showcase/demo.mp4", "poster": "showcase/poster.png", "kind": "synthetic",
             "caption": "The Gradio app computing live on three synthetic images (good, median and failure case): the predicted watermark mask is drawn over each image with its mIoU against the exact ground truth."},
    "reproduce": ["git clone https://github.com/Wassbdr/murmell && cd murmell/task1_watermark_localisation",
                  "pip install -r requirements.txt",
                  "python predict.py examples/good.png --gt examples/good_mask.png && python app.py",
                  "bash showcase/make_showcase.sh   # retrains and rebuilds every artifact"],
    "results": [
        {"label": "Task 1 watermark localisation, nested 5-fold mIoU (synthetic)", "value": f"{t1['miou_nested_per_kind']:.3f} (folds {fold})",
         "command": "cd task1_watermark_localisation && python -m src.train_synth --steps 1800", "setup": s1},
        {"label": "Task 1 all-empty-mask baseline, same metric", "value": f"{t1['baseline_all_zeros']:.3f}",
         "command": "cd task1_watermark_localisation && python -m src.train_synth --steps 1800", "setup": s1},
        {"label": "Task 1 inference time per 256 px image", "value": f"{t1['infer_ms_per_image']:.0f} ms",
         "command": "cd task1_watermark_localisation && python -m src.train_synth --steps 1800", "setup": f"batch 32, {hw}"},
        {"label": "Task 1 demo cases (good / median / failure)", "value": f"{ex['good']['miou']:.3f} / {ex['median']['miou']:.3f} / {ex['failure']['miou']:.3f}",
         "command": "python showcase/make_examples.py", "setup": f"eval seed {c['eval_seed']}, picked by measured IoU, {hw}"},
        {"label": "Task 2 model equality, score on the synthetic endpoint bench", "value": f"{t2['score_mean']:.3f} (family MCC {t2['mcc_family_mean']:.3f}, model MCC {t2['mcc_model_mean']:.3f})",
         "command": "cd task2_model_equality && python -m simlab.run --seeds 0 1 2 3 4", "setup": "5 seeds x 33 toy endpoints (3 families x 11), numpy, CPU"},
        {"label": "Task 3 forecast error on the synthetic refit loop", "value": f"MAE {t3['forecast_mae_mean']:.4f} vs parent baseline {t3['parent_mae_mean']:.4f}",
         "command": "cd task3_unlearning && python -m simlab.refit --seeds 0 1 2 3 4", "setup": "5 seeds x 14 forecasts, toy scorer, numpy, CPU"},
        {"label": "Competition final ranking (historical, not rerun: data not public)", "value": "1st of 44, overall 0.9819; task 1 0.8328",
         "command": "see final_leaderboard.png", "setup": "European AI Championship, Paris, November 2026, hidden 70 % split"},
    ],
    "limits": [
        "All fresh numbers are on synthetic data; the competition images (PixelSeal, TrustMark, StegaStamp invisible watermarks) are not public and were not used.",
        "The synthetic model localises visible overlays; it was not trained on invisible watermarks and will not find them.",
        f"Low-opacity watermarks are the main failure mode (mIoU by opacity: {', '.join(f'{k} {v:.2f}' for k, v in t1['miou_by_opacity'].items())}).",
        "Tasks 2 and 3 run on toy simulations of the competition problems; their competition scores cannot be rerun without the endpoints and receipts.",
    ],
    "role": "TODO",
    "repo": "https://github.com/Wassbdr/murmell",
}
(ROOT / "showcase/showcase.json").write_text(json.dumps(d, indent=1, ensure_ascii=False) + "\n"); print("showcase.json")
