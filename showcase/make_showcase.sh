#!/usr/bin/env bash
# Rebuild every showcase artifact from a clean clone (CPU only, fixed seeds).
# Requires: python>=3.10, pip, ffmpeg; Python packages of task*/requirements.txt; playwright + chromium (for the GIF):
#   pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
#   pip install -r task1_watermark_localisation/requirements.txt playwright && python -m playwright install chromium
# Set SKIP_TRAIN=1 to reuse the committed weights (the full run trains for ~30-45 min on 8 CPU cores).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; PY="${PYTHON:-python}"; cd "$ROOT"

# task 1: train on the synthetic stream, evaluate on 500 seeded images (5 folds, nested thresholds), save weights
( cd task1_watermark_localisation
  if [ "${SKIP_TRAIN:-0}" != "1" ]; then $PY -m src.train_synth --steps 1800 --seed 0; fi
  $PY predict.py examples/good.png --gt examples/good_mask.png --out /tmp/murmell_pred 2>/dev/null || true )
$PY showcase/make_examples.py
$PY showcase/make_poster.py
# task 2 and task 3 synthetic benches (charts in task*/results/)
( cd task2_model_equality && $PY -m simlab.run --seeds 0 1 2 3 4 )
( cd task3_unlearning && $PY -m simlab.refit --seeds 0 1 2 3 4 )

# demo: the Gradio app on the three synthetic examples, recorded live with Playwright
( cd task1_watermark_localisation && $PY app.py > /tmp/murmell_app.log 2>&1 & echo $! > /tmp/murmell_app.pid )
for i in $(seq 60); do curl -sf http://127.0.0.1:7860 > /dev/null && break; sleep 1; done
$PY showcase/record_demo.py; kill "$(cat /tmp/murmell_app.pid)" || true
ffmpeg -y -loglevel error -i showcase/_demo.webm -t 40 -vf "fps=25,scale=1280:-2" -c:v libx264 -pix_fmt yuv420p -movflags +faststart showcase/demo.mp4
ffmpeg -y -loglevel error -i showcase/demo.mp4 -vf "fps=8,scale=960:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=128[p];[b][p]paletteuse=dither=bayer" showcase/demo.gif
rm -rf showcase/_demo.webm showcase/_video
$PY showcase/make_json.py
ls -la showcase
