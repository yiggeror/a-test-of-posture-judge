#!/usr/bin/env bash
# System libraries + venv + dependencies + models + tests.
#
# The apt step is not optional on a headless box: the mediapipe wheel links
# against EGL/GLES even when it only ever runs CPU inference, so importing it
# fails with "libEGL.so.1: cannot open shared object file" without them.
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$PWD"

echo "==> system libraries"
if command -v apt-get >/dev/null 2>&1; then
  SUDO=""
  [ "$(id -u)" -ne 0 ] && SUDO="sudo"
  $SUDO apt-get update -qq
  $SUDO apt-get install -y -qq libegl1 libgles2 libgl1 libglib2.0-0
else
  echo "    (no apt-get; ensure libEGL.so.1 and libGLESv2.so.2 are present)"
fi

echo "==> virtualenv"
[ -d .venv ] || python3 -m venv .venv
./.venv/bin/pip install -q --upgrade pip
./.venv/bin/pip install -q -r requirements.txt

echo "==> models (Apache-2.0)"
mkdir -p models
fetch() {
  local url="$1" dest="$2"
  if [ -s "$dest" ]; then echo "    have $(basename "$dest")"; return; fi
  echo "    fetching $(basename "$dest")"
  curl -fsSL -o "$dest" "$url"
}
BASE="https://storage.googleapis.com/mediapipe-models"
fetch "$BASE/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task" \
      models/pose_landmarker_heavy.task
fetch "$BASE/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task" \
      models/pose_landmarker_full.task
fetch "$BASE/object_detector/efficientdet_lite0/float32/latest/efficientdet_lite0.tflite" \
      models/efficientdet_lite0.tflite

echo "==> import check"
./.venv/bin/python - <<'PY'
import mediapipe
from posture import landmarks
print(f"    mediapipe {mediapipe.__version__}")
print(f"    model: {landmarks.DEFAULT_MODEL}")
PY

echo "==> tests"
./.venv/bin/python -m pytest tests/ -q

echo
echo "Setup complete."
echo "  ./.venv/bin/python app.py                 # http://127.0.0.1:5000"
echo "  ./.venv/bin/python scripts/validate.py --dir testdata/front"
