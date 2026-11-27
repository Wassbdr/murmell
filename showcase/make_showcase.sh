#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
root="$PWD"
python_bin="$(command -v "${PYTHON:-python}")"
make shipped task2 task3 PYTHON="$python_bin"
PYTHONPATH="$root/task1_watermark_localisation" "$python_bin" showcase/build_assets.py

(cd task1_watermark_localisation && exec "$python_bin" app.py) > task1_watermark_localisation/work/showcase-app.log 2>&1 &
app_pid=$!
trap 'kill "$app_pid" 2>/dev/null || true' EXIT
for _ in {1..60}; do
  if curl -fs http://127.0.0.1:7860 > /dev/null; then break; fi
  if ! kill -0 "$app_pid" 2>/dev/null; then break; fi
  sleep 1
done
if ! curl -fs http://127.0.0.1:7860 > /dev/null; then
  tail -25 task1_watermark_localisation/work/showcase-app.log >&2
  exit 1
fi

"$python_bin" showcase/record_demo.py
ffmpeg -y -loglevel error -i showcase/_demo.webm -t 40 -vf 'fps=25,scale=1280:-2' \
  -c:v libx264 -pix_fmt yuv420p -movflags +faststart showcase/demo.mp4
ffmpeg -y -loglevel error -filter_complex_threads 1 -i showcase/demo.mp4 \
  -vf 'fps=6,split[a][b];[a]palettegen=max_colors=96[p];[b][p]paletteuse=dither=bayer' showcase/demo.gif
if [ "$(wc -c < showcase/demo.gif)" -ge 10000000 ]; then
  printf '%s\n' 'showcase/demo.gif exceeds 10 MB.' >&2
  exit 1
fi
rm -f showcase/_demo.webm
