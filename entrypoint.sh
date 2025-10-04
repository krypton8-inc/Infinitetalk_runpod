#!/bin/bash
set -euo pipefail

echo "=========================================="
echo "  RTX 6000 PRO 96GB - MAXIMUM QUALITY"
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
echo "💎 MAXIMUM QUALITY MODE ENABLED"
echo ""

# RTX 6000 PRO optimizations
export XFORMERS_FORCE_DISABLE_TRITON=1
export PYTORCH_ENABLE_MPS_FALLBACK=1
echo "Set attention backend safety flags"

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
echo "Starting ComfyUI with MAXIMUM settings..."
echo "- Window size: 121 (largest)"
echo "- VAE tiling: Disabled (full resolution)"
echo "- Block swapping: 0 (everything in VRAM)"
echo "- Prefetch blocks: 10 (maximum)"
echo "- Inference steps: 6 (maximum quality)"
echo "=========================================="
python /ComfyUI/main.py --listen ${ATTN_FLAG} &
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
echo ""

# Start the handler
echo "=========================================="
echo "Starting handler with MAXIMUM quality..."
echo "=========================================="
exec python handler.py