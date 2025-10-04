# Use specific version of nvidia cuda image
FROM krypton8/multitalk-base:1.1 as runtime

# Keep things lean but ensure we can fetch models at build time
RUN apt-get update && apt-get install -y --no-install-recommends wget && rm -rf /var/lib/apt/lists/*

# Speed up HF downloads and add runtime deps (S3 + ws + audio + optional onnx to silence node import warnings)
ENV HF_HUB_ENABLE_HF_TRANSFER=1
RUN pip install -U "huggingface_hub[hf_transfer]" \
    && pip install runpod websocket-client librosa boto3 onnx

WORKDIR /

# ---- ComfyUI core ----
RUN git clone https://github.com/comfyanonymous/ComfyUI.git && \
    cd /ComfyUI && \
    pip install -r requirements.txt

# ---- Custom nodes ----
RUN cd /ComfyUI/custom_nodes && \
    git clone https://github.com/Comfy-Org/ComfyUI-Manager.git && \
    cd ComfyUI-Manager && \
    pip install -r requirements.txt

RUN cd /ComfyUI/custom_nodes && \
    git clone https://github.com/city96/ComfyUI-GGUF && \
    cd ComfyUI-GGUF && \
    pip install -r requirements.txt

RUN cd /ComfyUI/custom_nodes && \
    git clone https://github.com/kijai/ComfyUI-KJNodes && \
    cd ComfyUI-KJNodes && \
    pip install -r requirements.txt

RUN cd /ComfyUI/custom_nodes && \
    git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite && \
    cd ComfyUI-VideoHelperSuite && \
    pip install -r requirements.txt

RUN cd /ComfyUI/custom_nodes && \
    git clone https://github.com/orssorbit/ComfyUI-wanBlockswap

RUN cd /ComfyUI/custom_nodes && \
    git clone https://github.com/kijai/ComfyUI-MelBandRoFormer && \
    cd ComfyUI-MelBandRoFormer && \
    pip install -r requirements.txt

RUN cd /ComfyUI/custom_nodes && \
    git clone https://github.com/kijai/ComfyUI-WanVideoWrapper && \
    cd ComfyUI-WanVideoWrapper && \
    pip install -r requirements.txt

# Create model directories (will be populated from network volume at runtime)
RUN mkdir -p /ComfyUI/models/diffusion_models \
    /ComfyUI/models/loras \
    /ComfyUI/models/vae \
    /ComfyUI/models/text_encoders \
    /ComfyUI/models/clip_vision

# Copy project files (entrypoint, handler, workflows, etc.)
COPY . .
RUN chmod +x /entrypoint.sh

# Default launch; entrypoint will auto-select attention mode per GPU
CMD ["/entrypoint.sh"]