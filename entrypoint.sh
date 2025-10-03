#!/bin/bash
set -euo pipefail

echo "=== Entrypoint: GPU-aware ComfyUI launcher ==="

# Detect GPU compute capability (e.g., 8.9 = Ada 4090, 9.0 = Hopper H100/H200, 12.0 = Blackwell 5090/B200)
CC="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1 || echo "unknown")"
VRAM_MB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -n1 || echo "0")"

echo "Detected compute capability: ${CC}"
echo "Detected VRAM (MB): ${VRAM_MB}"

# Choose attention mode:
# - Hopper (SM 9.x): enable SageAttention (fast and supported)
# - Ada (SM 8.x) & Blackwell (SM 12.x): disable SageAttention (kernel not available → avoids crashes you saw)
# - Override with WAN_ATTENTION_MODE={sage|torch|auto}
ATTN_MODE="${WAN_ATTENTION_MODE:-auto}"
ATTN_FLAG=""

case "${ATTN_MODE}" in
  sage)
    ATTN_FLAG="--use-sage-attention"
    echo "WAN_ATTENTION_MODE=sage → forcing SageAttention."
    ;;
  torch|none)
    ATTN_FLAG=""
    echo "WAN_ATTENTION_MODE=${ATTN_MODE} → using default PyTorch attention."
    ;;
  auto)
    if [[ "${CC}" == 9.* ]]; then
      ATTN_FLAG="--use-sage-attention"
      echo "Auto attention: Hopper (SM ${CC}) detected → enabling SageAttention."
    else
      ATTN_FLAG=""
      echo "Auto attention: non-Hopper (SM ${CC}) detected → using default PyTorch attention."
    fi
    ;;
  *)
    echo "Unknown WAN_ATTENTION_MODE='${ATTN_MODE}', falling back to auto."
    if [[ "${CC}" == 9.* ]]; then
      ATTN_FLAG="--use-sage-attention"
    else
      ATTN_FLAG=""
    fi
    ;;
esac

# Optional: nudge PyTorch to be conservative on lower VRAM cards (<= 24GB)
if [[ "${VRAM_MB}" -gt 0 && "${VRAM_MB}" -le 24500 ]]; then
  export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:64"
  echo "Low/medium VRAM detected (<=24GB). Set PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF}"
fi

# Start ComfyUI in the background
echo "Starting ComfyUI in the background with flag: '${ATTN_FLAG}' ..."
python /ComfyUI/main.py --listen ${ATTN_FLAG} &

# Wait for ComfyUI to be ready
echo "Waiting for ComfyUI to be ready..."
max_wait=180   # up to 3 minutes
elapsed=0
until curl -s http://127.0.0.1:8188/ >/dev/null 2>&1; do
  if (( elapsed >= max_wait )); then
    echo "Error: ComfyUI failed to start within ${max_wait}s"
    exit 1
  fi
  printf "Waiting for ComfyUI... (%ds/%ds)\n" "${elapsed}" "${max_wait}"
  sleep 2
  ((elapsed+=2))
done
echo "ComfyUI is ready!"

# Start the handler in the foreground (container's main process)
echo "Starting the handler..."
exec python handler.py
