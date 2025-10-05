#!/bin/bash
set -euo pipefail

PORT="${PORT:-8188}"

echo "=========================================="
echo "  GPU-aware ComfyUI launcher"
echo "=========================================="

# Basic GPU / CUDA telemetry
CC="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1 || echo "unknown")"
VRAM_MB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -n1 || echo "0")"
VRAM_FREE_MB="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -n1 || echo "0")"
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1 || echo "unknown")"

echo "Detected GPU: ${GPU_NAME}"
echo "Compute capability: ${CC}"
echo "VRAM total: ${VRAM_MB} MB"
echo "VRAM free: ${VRAM_FREE_MB} MB"
echo ""

# ------------------------------
# Runtime tuning flags (safe defaults)
# ------------------------------
export XFORMERS_FORCE_DISABLE_TRITON=1
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Math/allocator preferences
export NVIDIA_TF32_OVERRIDE=1
export TORCH_ALLOW_TF32_CUBLAS=1
export TORCH_ALLOW_TF32=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:512}"
export CUDA_MODULE_LOADING="${CUDA_MODULE_LOADING:-LAZY}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# Optional overlap control for InfiniteTalk (if used by your workflow)
export INF_TALK_OVERLAP_PCT="${INF_TALK_OVERLAP_PCT:-0.14}"

echo "Set attention/TF32/allocator flags"
echo "INF_TALK_OVERLAP_PCT=${INF_TALK_OVERLAP_PCT}"
echo ""

# Default attention backend information
echo "Using PyTorch SDPA as the default attention backend."
ATTN_FLAG=""
echo ""

# ------------------------------
# Quick CUDA/Torch preflight
# ------------------------------
echo "Running CUDA/Torch preflight..."
if ! python - <<'PY'
import sys, torch
print(f"Torch {torch.__version__} | CUDA {torch.version.cuda}")
if not torch.cuda.is_available():
    print("CUDA NOT AVAILABLE in torch", file=sys.stderr); sys.exit(2)
try:
    d = torch.cuda.current_device()
    print(f"GPU[{d}] {torch.cuda.get_device_name(d)} | CC {torch.cuda.get_device_capability(d)} | mem {torch.cuda.get_device_properties(d).total_memory//(1024*1024)} MB")
    free_b, total_b = torch.cuda.mem_get_info(d)
    print(f"mem_get_info: free={free_b//(1024*1024)} MB total={total_b//(1024*1024)} MB")
except Exception as e:
    print("CUDA preflight exception:", e, file=sys.stderr)
    sys.exit(3)
PY
then
  echo "❌ Torch/CUDA preflight failed. Ensure torch==2.7.0+cu128 matches the host driver."
  exit 1
fi
echo "✅ CUDA/Torch preflight OK"
echo ""

# ------------------------------
# Link models from network volume (if present)
# ------------------------------
echo "=== Checking for network volume models ==="
if [ -d "/runpod-volume/models" ]; then
    echo "Network volume found at /runpod-volume/models"
    echo "Linking models to ComfyUI..."

    for model_type in diffusion_models loras vae text_encoders clip_vision; do
        mkdir -p "/ComfyUI/models/${model_type}"
        if [ -d "/runpod-volume/models/${model_type}" ]; then
            echo "Linking ${model_type}..."
            shopt -s nullglob
            for f in /runpod-volume/models/${model_type}/*; do
                ln -sf "$f" "/ComfyUI/models/${model_type}/" || true
            done
            shopt -u nullglob
        fi
    done

    model_count=$(find /ComfyUI/models -type l -o -type f 2>/dev/null | wc -l)
    echo "Models linked: ${model_count} files accessible"
else
    echo "⚠️  WARNING: Network volume not found at /runpod-volume/models"
    echo "Models must be baked into the image or provided another way"
fi

# ------------------------------
# Start ComfyUI (with one retry)
# ------------------------------
start_comfy() {
  echo "=========================================="
  echo "Starting ComfyUI on port ${PORT}"
  echo "=========================================="

  # Stream ComfyUI logs to stdout
  python /ComfyUI/main.py --listen ${ATTN_FLAG} --port "${PORT}" &
  COMFY_PID=$!
  echo "ComfyUI started (PID: ${COMFY_PID})"
  echo ""

  # Wait for ComfyUI HTTP to be ready
  echo "Waiting for ComfyUI startup..."
  max_wait=300
  elapsed=0
  until curl -s "http://127.0.0.1:${PORT}/" >/dev/null 2>&1; do
    if ! kill -0 ${COMFY_PID} 2>/dev/null; then
      echo "❌ Error: ComfyUI process died during startup (see ComfyUI output above)"
      return 1
    fi

    if (( elapsed >= max_wait )); then
      echo "❌ Error: ComfyUI failed to start within ${max_wait}s (see ComfyUI output above)"
      return 1
    fi

    printf "Waiting... (%ds/%ds) [Free VRAM: %s MB]\n" \
      "${elapsed}" "${max_wait}" \
      "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -n1 || echo "unknown")"
    sleep 2
    ((elapsed+=2))
  done
  echo "✅ ComfyUI is ready!"
  return 0
}

# First attempt
if ! start_comfy; then
  echo "↻ One-time auto-retry after brief backoff..."
  sleep 3
  export CUDA_CACHE_DISABLE=1
  if ! start_comfy; then
    echo "❌ ComfyUI failed to start after a retry. Aborting."
    exit 1
  fi
fi

echo "✅ GPU initialization complete"
echo "✅ Ready to process requests"
echo ""

# ------------------------------
# Start the handler
# ------------------------------
echo "=========================================="
echo "Starting handler"
echo "=========================================="
exec python handler.py
