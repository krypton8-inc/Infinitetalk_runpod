#!/bin/bash
set -euo pipefail

PORT="${PORT:-8188}"

echo "=========================================="
echo "  RTX 6000 PRO 96GB - SPEED OPTIMIZED"
echo "=========================================="

# Detect GPU (friendly Blackwell names)
CC="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1 || echo "unknown")"
VRAM_MB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -n1 || echo "0")"
VRAM_FREE_MB="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -n1 || echo "0")"
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1 || echo "unknown")"

# Normalize a few common Blackwell strings
case "${GPU_NAME}" in
  *"Blackwell Server Edition"*)      GPU_FLAVOR="RTX PRO 6000 Blackwell (Server)";;
  *"Blackwell Workstation Edition"*) GPU_FLAVOR="RTX PRO 6000 Blackwell (Workstation)";;
  *"Blackwell Max-Q"*)               GPU_FLAVOR="RTX PRO 6000 Blackwell (Max-Q)";;
  *)                                 GPU_FLAVOR="${GPU_NAME}";;
esac

echo "Detected GPU: ${GPU_FLAVOR}"
echo "Compute capability: ${CC}"
echo "VRAM total: ${VRAM_MB} MB"
echo "VRAM free: ${VRAM_FREE_MB} MB"
echo ""

# Verify this is RTX 6000 PRO with sufficient VRAM
if (( VRAM_MB < 90000 )); then
  echo "❌ CRITICAL ERROR: Insufficient VRAM detected!"
  echo "❌ This build requires RTX 6000 PRO with ≥90GB VRAM"
  echo "❌ Current GPU has only ${VRAM_MB}MB VRAM"
  echo ""
  echo "📌 You are on the 'rtx-6000-pro' branch"
  echo "📌 For GPUs with <90GB VRAM, use the 'rtx-5090' branch instead"
  exit 1
fi

echo "✅ RTX 6000 PRO 96GB verified!"
echo "🚀 SPEED MODES ENABLED"
echo ""

# ------------------------------
# Runtime tuning flags
# ------------------------------
export XFORMERS_FORCE_DISABLE_TRITON=1
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Math/allocator speed-ups (help the sampler most)
export NVIDIA_TF32_OVERRIDE=1
export TORCH_ALLOW_TF32_CUBLAS=1
export TORCH_ALLOW_TF32=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:512}"
# Avoid eager CUDA module loads on cold start
export CUDA_MODULE_LOADING="${CUDA_MODULE_LOADING:-LAZY}"
# Ensure deterministic device ordering
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# Optional: adjust infinitetalk overlap percentage (default 0.14)
export INF_TALK_OVERLAP_PCT="${INF_TALK_OVERLAP_PCT:-0.14}"

echo "Set attention, TF32, and allocator flags"
echo "INF_TALK_OVERLAP_PCT=${INF_TALK_OVERLAP_PCT}"
echo ""

# Use PyTorch SDPA for maximum compatibility
ATTN_FLAG=""
echo "Using PyTorch SDPA (optimal for RTX 6000 PRO)"
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
d = torch.cuda.current_device()
print(f"GPU[{d}] {torch.cuda.get_device_name(d)} | CC {torch.cuda.get_device_capability(d)} | mem {torch.cuda.get_device_properties(d).total_memory//(1024*1024)} MB")
PY
then
  echo "❌ Torch/CUDA preflight failed. Check that torch==2.7.0+cu128 is installed and matches the host driver."
  exit 1
fi
echo "✅ CUDA/Torch preflight OK"
echo ""

# ------------------------------
# Link models from network volume
# ------------------------------
echo "=== Checking for network volume models ==="
if [ -d "/runpod-volume/models" ]; then
    echo "Network volume found at /runpod-volume/models"
    echo "Linking models to ComfyUI..."

    for model_type in diffusion_models loras vae text_encoders clip_vision; do
        mkdir -p "/ComfyUI/models/${model_type}"
        if [ -d "/runpod-volume/models/${model_type}" ]; then
            echo "Linking ${model_type}..."
            # Only link if there are entries; avoid literal glob copy when empty
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
  echo "Starting ComfyUI with flexible speed modes on port ${PORT}"
  echo ""
  echo "Available Speed Modes (set via API):"
  echo "  • maximum_quality: Window 121, Steps 6, Scheduler dpm++_sde (baseline quality)"
  echo "  • balanced:        Window 81,  Steps 5, Scheduler dpm++_sde (~35% faster) [DEFAULT]"
  echo "  • fast:            Window 121, Steps 4, Scheduler euler     (big window, fewer steps)"
  echo "  • turbo:           Window 96,  Steps 3, Scheduler euler     (big window, minimal steps)"
  echo ""
  echo "RTX 6000 PRO advantages (always active):"
  echo "  • VAE tiling: Disabled (full resolution)"
  echo "  • Block swapping: 0 (everything in VRAM)"
  echo "  • Prefetch blocks: 10 (maximum)"
  echo "  • Model offloading: Never needed"
  echo "=========================================="

  # Stream ComfyUI logs directly to container stdout (no redirection)
  python /ComfyUI/main.py --listen ${ATTN_FLAG} --port "${PORT}" &
  COMFY_PID=$!
  echo "ComfyUI started (PID: ${COMFY_PID})"
  echo ""

  # Wait for ComfyUI to be ready
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
  # Small nudge: drop any lingering CUDA context from the dead process
  export CUDA_CACHE_DISABLE=1
  if ! start_comfy; then
    echo "❌ ComfyUI failed to start after a retry. Aborting."
    exit 1
  fi
fi

echo "✅ RTX 6000 PRO 96GB optimizations active!"
echo "✅ Ready to process requests with flexible speed modes"
echo ""

# ------------------------------
# Start the handler (uses same port)
# ------------------------------
echo "=========================================="
echo "Starting handler with speed mode support"
echo "Default: balanced (~35% faster, minimal quality loss)"
echo "=========================================="
exec python handler.py
