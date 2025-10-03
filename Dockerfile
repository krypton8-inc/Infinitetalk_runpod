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

# Ensure model subfolders exist before downloading
RUN mkdir -p /ComfyUI/models/diffusion_models \
    /ComfyUI/models/loras \
    /ComfyUI/models/vae \
    /ComfyUI/models/text_encoders \
    /ComfyUI/models/clip_vision

# ---- Models & weights ----
RUN wget https://huggingface.co/Kijai/WanVideo_comfy_GGUF/resolve/main/InfiniteTalk/Wan2_1-InfiniteTalk_Single_Q8.gguf \
    -O /ComfyUI/models/diffusion_models/Wan2_1-InfiniteTalk_Single_Q8.gguf && \
    wget https://huggingface.co/Kijai/WanVideo_comfy_GGUF/resolve/main/InfiniteTalk/Wan2_1-InfiniteTalk_Multi_Q8.gguf \
    -O /ComfyUI/models/diffusion_models/Wan2_1-InfiniteTalk_Multi_Q8.gguf && \
    wget https://huggingface.co/city96/Wan2.1-I2V-14B-480P-gguf/resolve/main/wan2.1-i2v-14b-480p-Q8_0.gguf \
    -O /ComfyUI/models/diffusion_models/wan2.1-i2v-14b-480p-Q8_0.gguf && \
    wget https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Lightx2v/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors \
    -O /ComfyUI/models/loras/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors && \
    wget https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Wan2_1_VAE_bf16.safetensors \
    -O /ComfyUI/models/vae/Wan2_1_VAE_bf16.safetensors && \
    wget https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/umt5-xxl-enc-bf16.safetensors \
    -O /ComfyUI/models/text_encoders/umt5-xxl-enc-bf16.safetensors && \
    wget https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/clip_vision/clip_vision_h.safetensors \
    -O /ComfyUI/models/clip_vision/clip_vision_h.safetensors && \
    wget https://huggingface.co/Kijai/MelBandRoFormer_comfy/resolve/main/MelBandRoformer_fp16.safetensors \
    -O /ComfyUI/models/diffusion_models/MelBandRoformer_fp16.safetensors

# Copy project files (entrypoint, handler, workflows, etc.)
COPY . .
RUN chmod +x /entrypoint.sh

# Default launch; entrypoint will auto-select attention mode per GPU
CMD ["/entrypoint.sh"]
