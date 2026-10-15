#!/usr/bin/env bash
# Project-local native/GPU library paths; no system-wide changes.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SITE="$ROOT/.venv/lib/python3.12/site-packages"
export LD_LIBRARY_PATH="$ROOT/.native/usr/lib/x86_64-linux-gnu:$SITE/nvidia/cublas/lib:$SITE/nvidia/cudnn/lib:$SITE/nvidia/cuda_nvrtc/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HUB_DISABLE_TELEMETRY=1
exec "$ROOT/.venv/bin/python" "$@"
