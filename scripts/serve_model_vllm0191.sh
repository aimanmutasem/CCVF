#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${CCVF_VLLM0191_VENV:-$ROOT/.venv_vllm0191_py310}"
ACCOUNT_HOME="${CCVF_RUNTIME_HOME:-$(cd "$ROOT/../../.." && pwd)}"

unset PYTHONHOME
unset PYTHONPATH
unset CONDA_PREFIX
unset CONDA_DEFAULT_ENV
unset CONDA_PYTHON_EXE

export HOME="$ACCOUNT_HOME"
export XDG_CACHE_HOME="${CCVF_XDG_CACHE_HOME:-$ACCOUNT_HOME/.cache}"
export HF_HOME="${CCVF_HF_HOME:-$XDG_CACHE_HOME/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-$XDG_CACHE_HOME/vllm}"
export FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE:-$ACCOUNT_HOME}"
export FLASHINFER_WORKSPACE_DIR="${FLASHINFER_WORKSPACE_DIR:-$XDG_CACHE_HOME/flashinfer}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$XDG_CACHE_HOME/torch/inductor}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$XDG_CACHE_HOME/triton}"
mkdir -p "$XDG_CACHE_HOME" "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" \
  "$TRANSFORMERS_CACHE" "$VLLM_CACHE_ROOT" "$FLASHINFER_WORKSPACE_DIR" \
  "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"

if [[ ! -x "$VENV/bin/python" ]]; then
  echo "Missing $VENV/bin/python" >&2
  echo "Create it with: /bin/python -m venv .venv_vllm0191_py310 && .venv_vllm0191_py310/bin/pip install vllm==0.19.1 --extra-index-url https://download.pytorch.org/whl/cu128" >&2
  exit 1
fi
if [[ ! -x "$VENV/bin/vllm" ]]; then
  echo "Missing $VENV/bin/vllm" >&2
  echo "Install vLLM with: $VENV/bin/pip install vllm==0.19.1 --extra-index-url https://download.pytorch.org/whl/cu128" >&2
  exit 1
fi

BASE_PREFIX="$("$VENV/bin/python" - <<'PY'
import sys
print(sys.base_prefix)
PY
)"
if [[ -d "$BASE_PREFIX/lib" ]]; then
  export LD_LIBRARY_PATH="$BASE_PREFIX/lib:${LD_LIBRARY_PATH:-}"
fi
export PYTHONNOUSERSITE=1

"$VENV/bin/python" - <<'PY'
import sys
if sys.version_info < (3, 10):
    raise SystemExit(
        "vLLM 0.19.1 must be run with Python >= 3.10; "
        f"got {sys.version.split()[0]} at {sys.executable}"
    )
PY

"$VENV/bin/python" - <<'PY'
try:
    import sqlite3
    import diskcache
except Exception as exc:
    import os
    import sys
    raise SystemExit(
        "vLLM 0.19.1 runtime import preflight failed.\n"
        f"Python: {sys.executable}\n"
        f"Base prefix: {sys.base_prefix}\n"
        f"LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH', '')}\n"
        f"Error: {type(exc).__name__}: {exc}"
    ) from exc
print(f"Runtime import preflight OK: sqlite3={sqlite3.sqlite_version}, diskcache={diskcache.__version__}", flush=True)
PY

cd "$ROOT"
mkdir -p "$ROOT/logs"
LOG_FILE="${CCVF_VLLM_LOG_FILE:-$ROOT/logs/vllm0191_$(date +%Y%m%d_%H%M%S).log}"
echo "vLLM 0.19.1 log: $LOG_FILE" >&2
echo "Runtime home: $HOME" >&2
echo "vLLM cache: $VLLM_CACHE_ROOT" >&2
echo "FlashInfer cache base: $FLASHINFER_WORKSPACE_BASE" >&2

export VLLM_TARGET_DEVICE="${VLLM_TARGET_DEVICE:-cuda}"

"$VENV/bin/python" "$ROOT/scripts/serve_model.py" \
  --vllm-bin "$VENV/bin/vllm" \
  --skip-transformers-probe \
  "$@" 2>&1 | tee -a "$LOG_FILE"
