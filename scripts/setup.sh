#!/usr/bin/env bash
# One-shot setup. The apt step matters: the mediapipe wheel links against
# EGL/GLES even for pure-CPU image inference, and a headless container will
# not have them. Phase 0 hit exactly this:
#   OSError: libEGL.so.1: cannot open shared object file
#   OSError: libGLESv2.so.2: cannot open shared object file
set -euo pipefail
if command -v apt-get >/dev/null; then
  sudo apt-get update -q
  sudo apt-get install -y -q libegl1 libgles1 libgles2 libgl1 libglib2.0-0
fi
python3 -m venv .venv
./.venv/bin/pip install --upgrade pip
./.venv/bin/pip install -r requirements.txt
./.venv/bin/python scripts/fetch_models.py
./.venv/bin/python -m pytest tests/ -q
echo "Setup complete. Run: ./.venv/bin/python app.py"
