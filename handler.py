import os
import json
import uuid
import time
import base64
import logging
import urllib.request
import urllib.error
import subprocess
import websocket
import librosa
from urllib.parse import urlparse

import runpod

# Optional S3 (only used if all env vars are present)
try:
    import boto3
    from botocore.client import Config as BotoConfig
except Exception:
    boto3 = None


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SERVER_ADDRESS = os.getenv("SERVER_ADDRESS", "127.0.0.1")
CLIENT_ID = str(uuid.uuid4())

# Default sample assets - these are test assets hosted in a public S3 bucket
DEFAULT_IMAGE_URL = "https://krypton8.s3.us-east-1.amazonaws.com/test_image.png"
DEFAULT_AUDIO_URL = "https://krypton8.s3.us-east-1.amazonaws.com/test_audio.wav"

# Defaults specifically for multi-person inputs
DEFAULT_MULTI_IMAGE_URL = "https://krypton8.s3.us-east-1.amazonaws.com/multi_test_image.png"
DEFAULT_MULTI_AUDIO_URL_1 = "https://krypton8.s3.us-east-1.amazonaws.com/multi_test_audio.wav"
DEFAULT_MULTI_AUDIO_URL_2 = "https://krypton8.s3.us-east-1.amazonaws.com/multi_test_audio2.wav"

# -----------------------------
# Helpers (downloads, S3, etc.)
# -----------------------------
def download_file_from_url(url: str, output_path: str) -> str:
    """Download a file from URL -> output_path with wget (fast, handles redirects)."""
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        result = subprocess.run(
            ["wget", "-O", output_path, "--no-verbose", "--timeout=45", url],
            capture_output=True,
            text=True,
            timeout=90,
        )
        if result.returncode == 0:
            logger.info(f"Downloaded: {url} -> {output_path}")
            return output_path
        else:
            logger.error(f"wget failed for {url}: {result.stderr}")
            raise Exception(f"URL download failed: {result.stderr}")
    except subprocess.TimeoutExpired:
        logger.error(f"Download timeout for {url}")
        raise Exception(f"Download timeout for {url}")
    except Exception as e:
        logger.error(f"Download error for {url}: {e}")
        raise Exception(f"Download error: {e}")

def _out_with_ext(url: str, base_without_ext: str, default_ext: str = ".wav") -> str:
    """Build an output path preserving the extension from the URL path."""
    ext = os.path.splitext(urlparse(url).path)[1] or default_ext
    return os.path.abspath(base_without_ext + ext)

def ensure_url_only(field_name: str, job_input: dict) -> str | None:
    """Enforce URL-only inputs (image_url, video_url, wav_url, wav_url_2)."""
    url = job_input.get(field_name)
    if url:
        return url

    path_key = field_name.replace("_url", "_path")
    b64_key = field_name.replace("_url", "_base64")
    if job_input.get(path_key) or job_input.get(b64_key):
        raise Exception(
            f"Input '{field_name}' must be a URL. Local paths and base64 are not supported."
        )
    return None

def s3_configured() -> bool:
    return all(
        os.getenv(k)
        for k in ["S3_REGION", "S3_BUCKET", "S3_KEY", "S3_SECRET"]
    )

def s3_client():
    assert boto3 is not None, "boto3 is not installed in the image."
    return boto3.client(
        "s3",
        region_name=os.getenv("S3_REGION"),
        aws_access_key_id=os.getenv("S3_KEY"),
        aws_secret_access_key=os.getenv("S3_SECRET"),
        config=BotoConfig(s3={"addressing_style": "virtual"}),
    )

def s3_upload_and_url(file_path: str) -> str:
    """Upload file to S3 and return the public URL."""
    bucket = os.getenv("S3_BUCKET")
    region = os.getenv("S3_REGION")
    key_prefix = os.getenv("S3_PREFIX", "infinitetalk")
    basename = os.path.basename(file_path)
    key = f"{key_prefix.rstrip('/')}/{uuid.uuid4()}_{basename}"

    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    
    c = s3_client()
    extra_args = {"ContentType": "video/mp4"}
    
    if file_size_mb > 10:
        logger.info(f"Uploading {file_size_mb:.1f}MB to S3: {key}")
    
    c.upload_file(file_path, bucket, key, ExtraArgs=extra_args)

    public_base = os.getenv("S3_PUBLIC_BASE_URL")
    if public_base:
        public_base = public_base.rstrip("/")
        return f"{public_base}/{key}"
    else:
        return f"https://s3.{region}.amazonaws.com/{bucket}/{key}"

# -----------------------------
# GPU Optimization - RTX 5090 ONLY
# -----------------------------
def optimize_workflow_for_gpu(prompt: dict, max_frames: int = 81, resolution: str = "480p") -> dict:
    """
    RTX 5090 (32GB, SM 12.0) optimized settings.
    Maximum speed with intelligent memory management.
    
    Settings adjusted based on resolution and video length:
    - 480p: Maximum quality (6 steps), no tiling
    - 720p short (<20s): Maximum speed (4 steps), no tiling
    - 720p long (>20s): Balanced (4 steps), tiling for safety
    """
    # Verify GPU is RTX 5090
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5
        )
        compute_cap = result.stdout.strip() if result.returncode == 0 else "unknown"
        
        # Verify Blackwell architecture (SM 12.x)
        if not compute_cap.startswith("12."):
            logger.warning(f"⚠️  Non-5090 GPU detected (SM {compute_cap}). This build is optimized for RTX 5090 (SM 12.0) only.")
    except Exception as e:
        logger.warning(f"GPU detection failed: {e}")
        compute_cap = "12.0"
    
    logger.info(f"🚀 RTX 5090 Optimization | Resolution: {resolution} | Frames: {max_frames}")
    
    # RTX 5090 base settings (32GB VRAM, Blackwell)
    attention_mode = "sdpa"  # SageAttention not supported on Blackwell yet
    force_offload = False  # Keep everything in VRAM
    blocks_to_swap = 8  # Aggressive (minimal swapping)
    prefetch_blocks = 3  # Maximum prefetch
    frame_window_size = 81  # Standard window
    
    # Resolution and length-specific optimizations
    if resolution == "720p":
        # 720p ALWAYS needs tiling on 5090 for safety
        tiled_vae = True
        enable_vae_tiling = True
        steps = 4
        blocks_to_swap = 10  # Slightly more conservative
        logger.info("📊 720p: Balanced (tiling enabled, 4 steps)")
    else:  # 480p
        # 480p is easy for 5090 - prioritize quality
        tiled_vae = False
        enable_vae_tiling = False
        steps = 6 if max_frames <= 400 else 5
        logger.info(f"📊 480p: High quality ({steps} steps, no tiling)")
    
    # Apply settings to workflow nodes
    settings_applied = []
    
    if "122" in prompt:
        prompt["122"]["inputs"]["attention_mode"] = attention_mode
        settings_applied.append("SageAttention")
    
    if "128" in prompt:
        prompt["128"]["inputs"]["force_offload"] = force_offload
        prompt["128"]["inputs"]["steps"] = steps
        settings_applied.append(f"{steps} steps")
    
    if "192" in prompt:
        prompt["192"]["inputs"]["force_offload"] = force_offload
        prompt["192"]["inputs"]["tiled_vae"] = tiled_vae
        prompt["192"]["inputs"]["frame_window_size"] = frame_window_size
        settings_applied.append(f"VAE tiling: {tiled_vae}")
    
    if "130" in prompt:
        prompt["130"]["inputs"]["enable_vae_tiling"] = enable_vae_tiling
    
    if "229" in prompt:
        prompt["229"]["inputs"]["enable_vae_tiling"] = enable_vae_tiling
    
    if "134" in prompt:
        prompt["134"]["inputs"]["blocks_to_swap"] = blocks_to_swap
        prompt["134"]["inputs"]["prefetch_blocks"] = prefetch_blocks
        settings_applied.append(f"BlockSwap: {blocks_to_swap}")
    
    logger.info(f"✅ Applied: {', '.join(settings_applied)}")
    
    return prompt

# -----------------------------
# ComfyUI prompt I/O
# -----------------------------
def queue_prompt(prompt: dict, input_type="image", person_count="single"):
    url = f"http://{SERVER_ADDRESS}:8188/prompt"
    logger.info(f"Queueing prompt to: {url}")
    p = {"prompt": prompt, "client_id": CLIENT_ID}
    data = json.dumps(p).encode("utf-8")

    # Debug log for key nodes
    logger.info(f"Nodes in workflow: {len(prompt)}")
    if input_type == "image":
        logger.info(f"Image node(284): {prompt.get('284', {}).get('inputs', {}).get('image', 'NOT_SET')}")
    else:
        logger.info(f"Video node(228): {prompt.get('228', {}).get('inputs', {}).get('video', 'NOT_SET')}")
    logger.info(f"Audio node(125): {prompt.get('125', {}).get('inputs', {}).get('audio', 'NOT_SET')}")
    logger.info(f"Text node(241): {prompt.get('241', {}).get('inputs', {}).get('positive_prompt', 'NOT_SET')}")
    if person_count == "multi":
        if "307" in prompt:
            logger.info(f"Second audio node(307): {prompt.get('307', {}).get('inputs', {}).get('audio', 'NOT_SET')}")
        elif "313" in prompt:
            logger.info(f"Second audio node(313): {prompt.get('313', {}).get('inputs', {}).get('audio', 'NOT_SET')}")

    req = urllib.request.Request(url, data=data)
    req.add_header("Content-Type", "application/json")

    try:
        response = urllib.request.urlopen(req)
        result = json.loads(response.read())
        logger.info(f"Prompt queued OK: {result}")
        return result
    except urllib.error.HTTPError as e:
        error_body = e.read().decode('utf-8')
        logger.error(f"HTTP error: {e.code} - {e.reason}")
        logger.error(f"Response body: {error_body}")
        raise Exception(f"ComfyUI HTTP {e.code}: {error_body}")
    except Exception as e:
        logger.error(f"Queue prompt error: {e}")
        raise

def get_history(prompt_id: str) -> dict:
    url = f"http://{SERVER_ADDRESS}:8188/history/{prompt_id}"
    with urllib.request.urlopen(url) as response:
        return json.loads(response.read())

def get_video_filepaths(ws, prompt, input_type="image", person_count="single"):
    """Submit prompt, wait for completion, then read ComfyUI history."""
    prompt_id = queue_prompt(prompt, input_type, person_count)["prompt_id"]
    outputs = {}

    while True:
        out = ws.recv()
        if isinstance(out, str):
            message = json.loads(out)
            if message.get("type") == "executing":
                data = message.get("data", {})
                if data.get("node") is None and data.get("prompt_id") == prompt_id:
                    break

    history = get_history(prompt_id)[prompt_id]
    for node_id, node_output in history.get("outputs", {}).items():
        paths = []
        if "gifs" in node_output:
            for vid in node_output["gifs"]:
                if "fullpath" in vid and os.path.exists(vid["fullpath"]):
                    paths.append(vid["fullpath"])
        outputs[node_id] = paths

    return outputs

# -----------------------------
# Workflow utilities
# -----------------------------
def load_workflow(workflow_path: str) -> dict:
    with open(workflow_path, "r") as f:
        return json.load(f)

def get_workflow_path(input_type: str, person_count: str) -> str:
    if input_type == "image":
        return "/I2V_single.json" if person_count == "single" else "/I2V_multi.json"
    else:
        return "/V2V_single.json" if person_count == "single" else "/V2V_multi.json"

def validate_workflow_nodes(prompt: dict, input_type: str, person_count: str) -> None:
    """Validate that required nodes exist in the workflow before execution."""
    required_nodes = ["125", "241", "245", "246", "270"]
    
    if input_type == "image":
        required_nodes.append("284")
    else:
        required_nodes.append("228")
    
    if person_count == "multi":
        if input_type == "image":
            required_nodes.append("307")
        else:
            required_nodes.append("313")
    
    missing_nodes = [node for node in required_nodes if node not in prompt]
    
    if missing_nodes:
        raise Exception(
            f"Workflow validation failed: missing required nodes {missing_nodes}. "
            f"Workflow may have been regenerated with different node IDs."
        )

def get_audio_duration(audio_path: str) -> float | None:
    try:
        return librosa.get_duration(path=audio_path)
    except Exception as e:
        logger.warning(f"Failed to get audio duration ({audio_path}): {e}")
        return None

def calculate_max_frames_from_audio(wav_1: str, wav_2: str | None = None, fps: int = 25) -> int:
    durations = []
    d1 = get_audio_duration(wav_1)
    if d1 is not None:
        durations.append(d1)
        logger.info(f"Audio#1 duration: {d1:.2f}s")
    if wav_2:
        d2 = get_audio_duration(wav_2)
        if d2 is not None:
            durations.append(d2)
            logger.info(f"Audio#2 duration: {d2:.2f}s")
    if not durations:
        logger.warning("Audio duration unknown; using default 81 frames.")
        return 81
    max_duration = max(durations)
    max_frames = int(max_duration * fps) + 81
    logger.info(f"Longest audio {max_duration:.2f}s -> max_frames {max_frames}")
    return max_frames

def pick_dimensions(aspect_ratio: str | None, resolution: str | None) -> tuple[int, int, str]:
    """
    Choose (width, height, resolution_str) from aspect_ratio + resolution.
    Returns the resolution string for optimizer.
    """
    ar = (aspect_ratio or "9:16").strip()
    res = (resolution or "480p").strip().lower()

    if ar not in {"16:9", "9:16"}:
        logger.info(f"Invalid/missing aspect_ratio '{aspect_ratio}', defaulting to 9:16.")
        ar = "9:16"
    if res not in {"480p", "720p"}:
        logger.info(f"Invalid/missing resolution '{resolution}', defaulting to 480p.")
        res = "480p"

    if ar == "16:9":
        dims = (1280, 720) if res == "720p" else (854, 480)
    else:  # 9:16
        dims = (720, 1280) if res == "720p" else (480, 854)
    
    return (*dims, res)

# -----------------------------
# Main handler
# -----------------------------
def handler(job):
    job_input = job.get("input", {})

    logger.info(f"Received job input: {job_input}")
    task_id = f"task_{uuid.uuid4()}"
    task_dir = os.path.abspath(task_id)
    os.makedirs(task_dir, exist_ok=True)

    try:
        # Reject width/height if present
        if "width" in job_input or "height" in job_input:
            return {"error": "Inputs 'width' and 'height' are not allowed. Use 'aspect_ratio' and 'resolution'."}

        # Input type & persons
        input_type = job_input.get("input_type", "image")
        person_count = job_input.get("person_count", "single")
        
        if input_type not in {"image", "video"}:
            return {"error": f"Invalid input_type '{input_type}'. Must be 'image' or 'video'."}
        if person_count not in {"single", "multi"}:
            return {"error": f"Invalid person_count '{person_count}'. Must be 'single' or 'multi'."}
        
        logger.info(f"Workflow: type={input_type}, persons={person_count}")

        # Workflow
        workflow_path = get_workflow_path(input_type, person_count)
        logger.info(f"Using workflow: {workflow_path}")

        # URL-only media inputs
        media_local_path = None
        if input_type == "image":
            default_img = DEFAULT_MULTI_IMAGE_URL if person_count == "multi" else DEFAULT_IMAGE_URL
            image_url = ensure_url_only("image_url", job_input) or default_img
            out_img = _out_with_ext(image_url, os.path.join(task_dir, "input_image"), default_ext=".png")
            media_local_path = download_file_from_url(image_url, out_img)
            if not job_input.get("image_url"):
                which = "multi default" if person_count == "multi" else "single default"
                logger.info(f"No image_url was provided; using {which} image URL.")
        else:
            video_url = ensure_url_only("video_url", job_input)
            if not video_url:
                return {"error": "For input_type='video', you must provide 'video_url'."}
            out_vid = _out_with_ext(video_url, os.path.join(task_dir, "input_video"), default_ext=".mp4")
            media_local_path = download_file_from_url(video_url, out_vid)

        # Audio (URL-only)
        wav_path_1 = None
        wav_path_2 = None

        default_audio_1 = DEFAULT_MULTI_AUDIO_URL_1 if person_count == "multi" else DEFAULT_AUDIO_URL
        wav_url = ensure_url_only("wav_url", job_input) or default_audio_1
        out = _out_with_ext(wav_url, os.path.join(task_dir, "input_audio"), default_ext=".wav")
        wav_path_1 = download_file_from_url(wav_url, out)
        if not job_input.get("wav_url"):
            which = "multi default #1" if person_count == "multi" else "single default"
            logger.info(f"No wav_url provided; using {which} audio URL.")

        if person_count == "multi":
            wav_url_2 = ensure_url_only("wav_url_2", job_input) or DEFAULT_MULTI_AUDIO_URL_2
            out2 = _out_with_ext(wav_url_2, os.path.join(task_dir, "input_audio_2"), default_ext=".wav")
            wav_path_2 = download_file_from_url(wav_url_2, out2)
            if not job_input.get("wav_url_2"):
                logger.info("No wav_url_2 provided; using multi default #2 audio URL.")

        # Text and derived size
        prompt_text = job_input.get("prompt", "A person talking naturally")

        # Get dimensions and resolution
        ar = job_input.get("aspect_ratio")
        res_input = job_input.get("resolution")
        width, height, resolution = pick_dimensions(ar, res_input)
        logger.info(f"Using dimensions from aspect_ratio/resolution -> width={width}, height={height}")

        # max_frame auto by audio length unless user provides
        max_frame = job_input.get("max_frame")
        if max_frame is None:
            logger.info("max_frame not provided; deriving from audio duration.")
            max_frame = calculate_max_frames_from_audio(wav_path_1, wav_path_2 if person_count == "multi" else None)
        else:
            logger.info(f"Using user-specified max_frame: {max_frame}")

        logger.info(f"Settings: prompt='{prompt_text}', width={width}, height={height}, max_frame={max_frame}")
        logger.info(f"Media path: {media_local_path}")
        logger.info(f"Audio #1: {wav_path_1}")
        if person_count == "multi":
            logger.info(f"Audio #2: {wav_path_2}")

        # Load and fill workflow
        prompt = load_workflow(workflow_path)
        
        # Validate workflow has required nodes
        validate_workflow_nodes(prompt, input_type, person_count)

        # OPTIMIZE WORKFLOW FOR RTX 5090
        prompt = optimize_workflow_for_gpu(prompt, max_frames=max_frame, resolution=resolution)

        # Existence and validation checks
        if not os.path.exists(media_local_path):
            return {"error": f"Media file not found: {media_local_path}"}
        if os.path.getsize(media_local_path) == 0:
            return {"error": f"Media file is empty: {media_local_path}"}
            
        if not os.path.exists(wav_path_1):
            return {"error": f"Audio file not found: {wav_path_1}"}
        if os.path.getsize(wav_path_1) == 0:
            return {"error": f"Audio file is empty: {wav_path_1}"}
            
        if person_count == "multi" and wav_path_2:
            if not os.path.exists(wav_path_2):
                return {"error": f"Second audio file not found: {wav_path_2}"}
            if os.path.getsize(wav_path_2) == 0:
                return {"error": f"Second audio file is empty: {wav_path_2}"}

        logger.info(f"Media size: {os.path.getsize(media_local_path)} bytes")
        logger.info(f"Audio #1 size: {os.path.getsize(wav_path_1)} bytes")
        if person_count == "multi" and wav_path_2:
            logger.info(f"Audio #2 size: {os.path.getsize(wav_path_2)} bytes")

        # Bind inputs to workflow (using absolute paths)
        if input_type == "image":
            prompt["284"]["inputs"]["image"] = media_local_path
        else:
            prompt["228"]["inputs"]["video"] = media_local_path

        prompt["125"]["inputs"]["audio"] = wav_path_1
        prompt["241"]["inputs"]["positive_prompt"] = prompt_text
        prompt["245"]["inputs"]["value"] = width
        prompt["246"]["inputs"]["value"] = height
        prompt["270"]["inputs"]["value"] = max_frame

        if person_count == "multi":
            audio_2_node = "307" if input_type == "image" else "313"
            if audio_2_node not in prompt:
                return {"error": f"Expected node {audio_2_node} not found in {input_type} multi workflow"}
            prompt[audio_2_node]["inputs"]["audio"] = wav_path_2

        # Connectivity checks
        ws_url = f"ws://{SERVER_ADDRESS}:8188/ws?clientId={CLIENT_ID}"
        http_url = f"http://{SERVER_ADDRESS}:8188/"
        logger.info(f"Connecting to ComfyUI: {http_url} (ws={ws_url})")

        # Wait for HTTP up (<= 5 min for cold starts)
        for attempt in range(300):
            try:
                urllib.request.urlopen(http_url, timeout=5)
                logger.info(f"HTTP connection OK (attempt {attempt+1})")
                break
            except Exception:
                if attempt == 299:
                    raise Exception("Cannot reach ComfyUI HTTP endpoint after 5 minutes.")
                time.sleep(1)

        # WS connect (up to 5 min)
        ws = websocket.WebSocket()
        for attempt in range(60):
            try:
                ws.connect(ws_url)
                logger.info(f"WebSocket connected (attempt {attempt+1})")
                break
            except Exception as e:
                if attempt == 59:
                    raise Exception(f"WebSocket connect timeout after 5 minutes. Last error: {e}")
                time.sleep(5)

        # Run and collect output paths
        video_paths_by_node = get_video_filepaths(ws, prompt, input_type, person_count)
        ws.close()

        # Return the first available video
        for node_id, paths in video_paths_by_node.items():
            for p in paths:
                if os.path.exists(p):
                    if s3_configured():
                        try:
                            url = s3_upload_and_url(p)
                            logger.info(f"Uploaded to S3: {url}")
                            return {"video_url": url}
                        except Exception as e:
                            logger.error(f"S3 upload failed, falling back to base64. Error: {e}")

                    # Fallback: base64 data URL
                    with open(p, "rb") as f:
                        b64 = base64.b64encode(f.read()).decode("utf-8")
                    logger.info(f"Returning base64 video ({len(b64)} chars)")
                    return {"video": f"data:video/mp4;base64,{b64}"}

        return {"error": "No video file was produced by the workflow."}

    except Exception as e:
        logger.error(f"Handler error: {e}", exc_info=True)
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})