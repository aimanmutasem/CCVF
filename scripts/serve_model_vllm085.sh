#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$ROOT/.venv_vllm085"

if [[ ! -x "$VENV/bin/vllm" ]]; then
  echo "Missing $VENV/bin/vllm" >&2
  echo "Create it with: python -m venv .venv_vllm085 && .venv_vllm085/bin/pip install vllm==0.8.5.post1 transformers==4.51.3" >&2
  exit 1
fi

cd "$ROOT"
mkdir -p "$ROOT/logs"
LOG_FILE="${CCVF_VLLM_LOG_FILE:-$ROOT/logs/vllm_$(date +%Y%m%d_%H%M%S).log}"
echo "vLLM log: $LOG_FILE" >&2

OVERLAY_ARGS=()
if [[ "${CCVF_DISABLE_TRANSFORMERS_OVERLAY:-0}" == "1" ]]; then
  OVERLAY_ARGS+=(--no-transformers-overlay)
fi

"$VENV/bin/python" "$ROOT/scripts/serve_model.py" \
  --vllm-bin "$VENV/bin/vllm" \
  "${OVERLAY_ARGS[@]}" \
  "$@" 2>&1 | tee -a "$LOG_FILE"
