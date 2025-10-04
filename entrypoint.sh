#!/bin/bash
set -euo pipefail

echo "=========================================="
echo "  RTX 6000 PRO 96GB - SPEED OPTIMIZED"
echo "=========================================="

# Detect GPU
CC="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1 || echo "unknown")"
VRAM_MB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -n1 || echo "0")"
VRAM_FREE_MB="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -n1 || echo "0")"
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1 || echo "unknown")"

echo "Detected GPU: ${GPU_NAME}"
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

# RTX 6000 PRO optimizations
export XFORMERS_FORCE_DISABLE_TRITON=1
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Math/allocator speed-ups (help the sampler most)
export NVIDIA_TF32_OVERRIDE=1
export TORCH_ALLOW_TF32_CUBLAS=1
export TORCH_ALLOW_TF32=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Optional: adjust infinitetalk overlap percentage (default 0.14)
export INF_TALK_OVERLAP_PCT="${INF_TALK_OVERLAP_PCT:-0.14}"

echo "Set attention, TF32, and allocator flags"
echo "INF_TALK_OVERLAP_PCT=${INF_TALK_OVERLAP_PCT}"
echo ""

# Use PyTorch SDPA for maximum compatibility
ATTN_FLAG=""
echo "Using PyTorch SDPA (optimal for RTX 6000 PRO)"
echo ""

# Link models from network volume
echo "=== Checking for network volume models ==="
if [ -d "/runpod-volume/models" ]; then
    echo "Network volume found at /runpod-volume/models"
    echo "Linking models to ComfyUI..."

    for model_type in diffusion_models loras vae text_encoders clip_vision; do
        if [ -d "/runpod-volume/models/${model_type}" ]; then
            echo "Linking ${model_type}..."
            ln -sf /runpod-volume/models/${model_type}/* /ComfyUI/models/${model_type}/ 2>/dev/null || true
        fi
    done

    model_count=$(find /ComfyUI/models -type l -o -type f 2>/dev/null | wc -l)
    echo "Models linked: ${model_count} files accessible"
else
    echo "⚠️  WARNING: Network volume not found at /runpod-volume/models"
    echo "Models must be baked into the image or provided another way"
fi

# Start ComfyUI
echo "=========================================="
echo "Starting ComfyUI with flexible speed modes"
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

# Log ComfyUI output to a file so tail on failure works
python /ComfyUI/main.py --listen ${ATTN_FLAG} > /tmp/comfyui.log 2>&1 &
COMFY_PID=$!
echo "ComfyUI started (PID: ${COMFY_PID})"
echo ""

# Wait for ComfyUI to be ready
echo "Waiting for ComfyUI startup..."
max_wait=300
elapsed=0
until curl -s http://127.0.0.1:8188/ >/dev/null 2>&1; do
  if ! kill -0 ${COMFY_PID} 2>/dev/null; then
    echo "❌ Error: ComfyUI process died during startup"
    echo "Last log lines:"
    tail -50 /tmp/comfyui.log 2>/dev/null || echo "(no log available)"
    exit 1
  fi

  if (( elapsed >= max_wait )); then
    echo "❌ Error: ComfyUI failed to start within ${max_wait}s"
    tail -50 /tmp/comfyui.log 2>/dev/null || echo "(no log available)"
    exit 1
  fi

  printf "Waiting... (%ds/%ds) [Free VRAM: %s MB]\n" \
    "${elapsed}" "${max_wait}" \
    "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -n1 || echo "unknown")"
  sleep 2
  ((elapsed+=2))
done
echo "✅ ComfyUI is ready!"
echo "✅ RTX 6000 PRO 96GB optimizations active!"
echo "✅ Ready to process requests with flexible speed modes"
echo ""

# Start the handler
echo "=========================================="
echo "Starting handler with speed mode support"
echo "Default: balanced (~35% faster, minimal quality loss)"
echo "=========================================="
exec python handler.py
