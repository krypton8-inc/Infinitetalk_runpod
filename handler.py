import os
import math
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
import mimetypes
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
    guessed_mime, _ = mimetypes.guess_type(file_path)
    content_type = guessed_mime or "video/mp4"
    extra_args = {"ContentType": content_type}

    if file_size_mb > 10:
        logger.info(f"Uploading {file_size_mb:.1f}MB to S3: {key} (ContentType={content_type})")

    c.upload_file(file_path, bucket, key, ExtraArgs=extra_args)

    public_base = os.getenv("S3_PUBLIC_BASE_URL")
    if public_base:
        public_base = public_base.rstrip("/")
        return f"{public_base}/{key}"
    else:
        return f"https://s3.{region}.amazonaws.com/{bucket}/{key}"

# -----------------------------
# GPU Optimization - SPEED MODES
# -----------------------------
def optimize_workflow_for_gpu(prompt: dict, max_frames: int = 81, resolution: str = "480p",
                              person_count: str = "single", speed_mode: str = "balanced") -> dict:
    """
    RTX 6000 PRO (96GB VRAM) with multiple SPEED MODES.

    Improvements vs base:
      • Force Wan on GPU (no offload)
      • Big-window presets for fast/turbo (fewer windows on long clips)
      • Scaled overlap (motion_frame ≈ window * 0.14 by default)
      • Optional torch.compile wiring (guarded)
      • Clear 'window plan' logging
    """
    # Detect VRAM to verify RTX 6000 PRO
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5
        )
        vram_mb = int(float(result.stdout.strip())) if result.returncode == 0 else 96000

        if vram_mb < 90000:
            logger.error(f"CRITICAL: Only {vram_mb}MB VRAM detected!")
            logger.error(f"This build requires RTX 6000 PRO (96GB VRAM)")
            raise Exception(f"Insufficient VRAM: {vram_mb}MB < 90GB minimum")

        logger.info(f"RTX 6000 PRO 96GB detected: {vram_mb}MB VRAM")
    except subprocess.CalledProcessError as e:
        logger.warning(f"VRAM detection failed: {e}, assuming RTX 6000 PRO")
        vram_mb = 96000

    logger.info(f"Resolution: {resolution} | Frames: {max_frames} | Persons: {person_count} | Speed: {speed_mode}")

    # Speed mode configurations with quality trade-off descriptions
    # NOTE: fast/turbo use BIG windows to reduce number of sliding windows in infinitetalk
    speed_configs = {
        "maximum_quality": {
            "window": 121,
            "steps": 6,
            "scheduler": "dpm++_sde",
            "desc": "MAXIMUM QUALITY - Best results, pristine motion",
            "quality": "Perfect - No compromises, ideal for production"
        },
        "balanced": {
            "window": 81,
            "steps": 5,
            "scheduler": "dpm++_sde",
            "desc": "BALANCED - 30-35% faster, imperceptible quality loss",
            "quality": "Excellent - Ideal for most use cases"
        },
        "fast": {
            "window": 121,  # big window to reduce total windows
            "steps": 4,
            "scheduler": "euler",
            "desc": "FAST - large window + fewer steps",
            "quality": "Very Good - Suitable for drafts and iterations"
        },
        "turbo": {
            "window": 96,   # still large, but a bit smaller than max
            "steps": 3,
            "scheduler": "euler",
            "desc": "TURBO - large window + minimal steps",
            "quality": "Good - Best for rapid testing and previews"
        }
    }

    # Validate and get speed config
    if speed_mode not in speed_configs:
        logger.warning(f"Invalid speed_mode '{speed_mode}', defaulting to 'balanced'")
        speed_mode = "balanced"

    config = speed_configs[speed_mode]
    frame_window_size = config["window"]
    steps = config["steps"]
    scheduler = config["scheduler"]

    # Heuristic for very long clips: ensure big windows on fast/turbo
    if max_frames and max_frames > 900 and speed_mode in {"fast", "turbo"}:
        if speed_mode == "fast":
            frame_window_size = max(frame_window_size, 121)
            steps = min(steps, 4)
            scheduler = "euler"
        else:  # turbo
            frame_window_size = max(frame_window_size, 96)
            steps = min(steps, 3)
            scheduler = "euler"

    logger.info(f"SPEED MODE: {config['desc']}")
    logger.info(f"Quality Trade-off: {config['quality']}")
    logger.info(f"Window: {frame_window_size} | Steps: {steps} | Scheduler: {scheduler}")

    # RTX 6000 PRO base optimizations (always enabled)
    attention_mode = "sdpa"
    force_offload = False
    tiled_vae = False
    enable_vae_tiling = False
    blocks_to_swap = 0
    prefetch_blocks = 10

    # Resolution-specific logging
    if resolution == "720p" and person_count == "multi":
        logger.info("720p MULTI-PERSON mode active")
    elif resolution == "720p":
        logger.info("720p SINGLE-PERSON mode active")

    # Apply settings to workflow nodes
    settings_applied = []

    # WanVideoModelLoader
    if "122" in prompt:
        prompt["122"]["inputs"]["attention_mode"] = attention_mode
        # Keep Wan on the main GPU (96GB; no offload)
        try:
            prompt["122"]["inputs"]["load_device"] = "main_device"
            settings_applied.append("load_device=main_device")
        except Exception:
            pass
        settings_applied.append(f"attention_mode={attention_mode}")

    # Sampler
    if "128" in prompt:
        prompt["128"]["inputs"]["force_offload"] = force_offload
        prompt["128"]["inputs"]["steps"] = steps
        prompt["128"]["inputs"]["scheduler"] = scheduler
        settings_applied.append(f"steps={steps}")
        settings_applied.append(f"scheduler={scheduler}")

    # Long I2V node (window + scaled overlap)
    motion_frame_used = None
    if "192" in prompt:
        prompt["192"]["inputs"]["force_offload"] = force_offload
        prompt["192"]["inputs"]["tiled_vae"] = tiled_vae
        prompt["192"]["inputs"]["frame_window_size"] = frame_window_size

        # Scale overlap with window (default 14%)
        try:
            overlap_pct = float(os.getenv("INF_TALK_OVERLAP_PCT", "0.14"))
        except Exception:
            overlap_pct = 0.14
        motion_frame_used = max(8, int(round(frame_window_size * overlap_pct)))
        if "motion_frame" in prompt["192"]["inputs"]:
            prompt["192"]["inputs"]["motion_frame"] = motion_frame_used
            settings_applied.append(f"motion_frame={motion_frame_used}")

        settings_applied.append(f"window={frame_window_size}")
        settings_applied.append(f"tiled_vae={tiled_vae}")

    # VAE decode / encode tiling flags
    if "130" in prompt:
        prompt["130"]["inputs"]["enable_vae_tiling"] = enable_vae_tiling
        settings_applied.append(f"vae_decode_tiling={enable_vae_tiling}")

    if "229" in prompt:
        prompt["229"]["inputs"]["enable_vae_tiling"] = enable_vae_tiling
        settings_applied.append(f"vae_encode_tiling={enable_vae_tiling}")

    # Block swap (keep everything in VRAM)
    if "134" in prompt:
        prompt["134"]["inputs"]["blocks_to_swap"] = blocks_to_swap
        prompt["134"]["inputs"]["prefetch_blocks"] = prefetch_blocks
        settings_applied.append(f"blocks_to_swap={blocks_to_swap}")
        settings_applied.append(f"prefetch={prefetch_blocks}")

    # Keep CLIP-ViT on GPU (minor startup win)
    if "237" in prompt:
        try:
            prompt["237"]["inputs"]["force_offload"] = False
            settings_applied.append("clip_force_offload=False")
        except Exception:
            pass

    # Optional: wire torch.compile settings if the nodes expose the pin
    if "177" in prompt:
        for nid in ("122", "128"):
            if nid in prompt:
                try:
                    if "compile_settings" in prompt[nid]["inputs"]:
                        prompt[nid]["inputs"]["compile_settings"] = ["177", 0]
                        settings_applied.append(f"{nid}.compile_settings=177")
                except Exception:
                    pass

    logger.info(f"Applied {len(settings_applied)} optimizations:")
    for setting in settings_applied:
        logger.info(f"  - {setting}")

    # Log a simple window estimate so users understand runtime
    def _estimate_windows(frames, W, overlap):
        if not frames or W <= 0:
            return 0
        stride = max(1, W - (overlap or 0))
        return int(math.ceil(max(0, frames - W) / stride)) + 1

    est_windows = _estimate_windows(max_frames, frame_window_size, motion_frame_used if motion_frame_used is not None else 9)
    logger.info(f"Window plan → frames={max_frames}  W={frame_window_size}  overlap={motion_frame_used if motion_frame_used is not None else 'n/a'}  ⇒  ~{est_windows} windows")

    # Store config in prompt metadata for response
    prompt["_speed_mode_used"] = speed_mode
    prompt["_config_applied"] = {
        "speed_mode": speed_mode,
        "window": frame_window_size,
        "steps": steps,
        "scheduler": scheduler,
        "motion_frame": motion_frame_used,
        "estimated_windows": est_windows,
        "quality": config["quality"]
    }

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
        for key in ("gifs", "videos"):  # support both keys
            if key in node_output:
                for vid in node_output[key]:
                    fp = vid.get("fullpath") or vid.get("path")
                    if fp and os.path.exists(fp):
                        paths.append(fp)
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

        # NEW: Speed mode selection with env var support
        default_speed = os.getenv("DEFAULT_SPEED_MODE", "balanced")
        speed_mode = job_input.get("speed_mode", default_speed)
        valid_modes = {"maximum_quality", "balanced", "fast", "turbo"}
        if speed_mode not in valid_modes:
            logger.warning(f"Invalid speed_mode '{speed_mode}', defaulting to 'balanced'")
            speed_mode = "balanced"

        logger.info(f"Workflow: type={input_type}, persons={person_count}, speed={speed_mode}")

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

        # OPTIMIZE WORKFLOW WITH SPEED MODE
        prompt = optimize_workflow_for_gpu(
            prompt, max_frames=max_frame, resolution=resolution,
            person_count=person_count, speed_mode=speed_mode
        )

        # Extract config metadata for response
        speed_mode_used = prompt.pop("_speed_mode_used", speed_mode)
        config_applied = prompt.pop("_config_applied", {})

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
        ws.settimeout(600)
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
                            return {
                                "video_url": url,
                                "settings": {
                                    "speed_mode": speed_mode_used,
                                    "resolution": f"{width}x{height}",
                                    "frames": max_frame,
                                    "input_type": input_type,
                                    "person_count": person_count,
                                    **config_applied
                                }
                            }
                        except Exception as e:
                            logger.error(f"S3 upload failed, falling back to base64. Error: {e}")

                    # Fallback: base64 data URL with correct MIME
                    with open(p, "rb") as f:
                        b64 = base64.b64encode(f.read()).decode("utf-8")
                    mime, _ = mimetypes.guess_type(p)
                    prefix = f"data:{mime or 'application/octet-stream'};base64,"
                    logger.info(f"Returning base64 video ({len(b64)} chars, mime={mime})")
                    return {
                        "video": f"{prefix}{b64}",
                        "settings": {
                            "speed_mode": speed_mode_used,
                            "resolution": f"{width}x{height}",
                            "frames": max_frame,
                            "input_type": input_type,
                            "person_count": person_count,
                            **config_applied
                        }
                    }

        return {"error": "No video file was produced by the workflow."}

    except Exception as e:
        logger.error(f"Handler error: {e}", exc_info=True)
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
