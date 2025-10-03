import os
import json
import uuid
import time
import base64
import logging
import urllib.request
import subprocess
import websocket
import librosa

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

# -----------------------------
# Helpers (downloads, S3, etc.)
# -----------------------------
def download_file_from_url(url: str, output_path: str) -> str:
    """Download a file from URL -> output_path with wget (fast, handles redirects)."""
    try:
        result = subprocess.run(
            ["wget", "-O", output_path, "--no-verbose", "--timeout=45", url],
            capture_output=True,
            text=True,
            timeout=90,
        )
        if result.returncode == 0:
            logger.info(f"✅ Downloaded: {url} -> {output_path}")
            return output_path
        else:
            logger.error(f"❌ wget failed: {result.stderr}")
            raise Exception(f"URL download failed: {result.stderr}")
    except subprocess.TimeoutExpired:
        logger.error("❌ Download timeout")
        raise Exception("Download timeout")
    except Exception as e:
        logger.error(f"❌ Download error: {e}")
        raise Exception(f"Download error: {e}")

def ensure_url_only(field_name: str, job_input: dict) -> str | None:
    """
    Enforce URL-only inputs (image_url, video_url, wav_url, wav_url_2).
    Returns the URL string if present, else None.
    Ignores/blocks *_path and *_base64 to avoid unsupported input types.
    """
    url = job_input.get(field_name)
    if url:
        return url

    # If the user sent *_path or *_base64, explicitly reject to be clear
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
    """
    Upload file to S3 and return the public URL.
    If S3_PUBLIC_BASE_URL is set, use that as prefix (e.g., CloudFront).
    Otherwise return the standard s3 URL.
    """
    bucket = os.getenv("S3_BUCKET")
    region = os.getenv("S3_REGION")
    key_prefix = os.getenv("S3_PREFIX", "infinitetalk")
    basename = os.path.basename(file_path)
    key = f"{key_prefix.rstrip('/')}/{uuid.uuid4()}_{basename}"

    c = s3_client()
    extra_args = {"ContentType": "video/mp4"}
    c.upload_file(file_path, bucket, key, ExtraArgs=extra_args)

    public_base = os.getenv("S3_PUBLIC_BASE_URL")
    if public_base:
        public_base = public_base.rstrip("/")
        return f"{public_base}/{key}"
    else:
        # Standard S3 URL
        return f"https://{bucket}.s3.{region}.amazonaws.com/{key}"

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
        logger.error(f"HTTP error: {e.code} - {e.reason}")
        logger.error(f"Body: {e.read().decode('utf-8')}")
        raise
    except Exception as e:
        logger.error(f"Queue prompt error: {e}")
        raise

def get_history(prompt_id: str) -> dict:
    url = f"http://{SERVER_ADDRESS}:8188/history/{prompt_id}"
    with urllib.request.urlopen(url) as response:
        return json.loads(response.read())

def get_video_filepaths(ws, prompt, input_type="image", person_count="single"):
    """
    Submit prompt, wait for completion, then read ComfyUI history and
    collect output video file paths from VHS VideoCombine nodes ("gifs" array with fullpath).
    Returns dict[node_id] -> [file_path, ...]
    """
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
        # Binary messages ignored

    history = get_history(prompt_id)[prompt_id]
    for node_id, node_output in history.get("outputs", {}).items():
        paths = []
        # ComfyUI VHS_VideoCombine uses 'gifs' with 'fullpath' for video artifacts
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

def pick_dimensions(aspect_ratio: str | None, resolution: str | None, fallback_w: int, fallback_h: int) -> tuple[int, int]:
    """
    Choose (width, height) from aspect_ratio + resolution.
    Valid aspect_ratio: "16:9" or "9:16" (default 9:16 if missing/invalid).
    Valid resolution: "480p" or "720p" (default 480p if missing/invalid).
    Mappings:
      16:9  -> 854x480 or 1280x720
      9:16  -> 480x854 or 720x1280
    """
    ar = (aspect_ratio or "9:16").strip()
    res = (resolution or "480p").strip().lower()

    if ar not in {"16:9", "9:16"}:
        ar = "9:16"
    if res not in {"480p", "720p"}:
        res = "480p"

    if ar == "16:9":
        return (1280, 720) if res == "720p" else (854, 480)
    else:  # 9:16
        return (720, 1280) if res == "720p" else (480, 854)

# -----------------------------
# Main handler
# -----------------------------
def handler(job):
    job_input = job.get("input", {})

    logger.info(f"Received job input: {job_input}")
    task_id = f"task_{uuid.uuid4()}"
    os.makedirs(task_id, exist_ok=True)

    # Input type & persons
    input_type = job_input.get("input_type", "image")          # "image" | "video"
    person_count = job_input.get("person_count", "single")     # "single" | "multi"
    logger.info(f"Workflow: type={input_type}, persons={person_count}")

    # Workflow
    workflow_path = get_workflow_path(input_type, person_count)
    logger.info(f"Using workflow: {workflow_path}")

    # URL-only media inputs
    media_local_path = None
    if input_type == "image":
        image_url = ensure_url_only("image_url", job_input)
        if image_url:
            media_local_path = download_file_from_url(image_url, os.path.join(task_id, "input_image.jpg"))
        else:
            media_local_path = "/examples/image.jpg"  # fallback sample
            logger.info("No image_url was provided; using bundled example image.")
    else:
        video_url = ensure_url_only("video_url", job_input)
        if video_url:
            media_local_path = download_file_from_url(video_url, os.path.join(task_id, "input_video.mp4"))
        else:
            # For V2V, if no video_url is provided, we still allow a fallback image to drive I2V-like behavior
            media_local_path = "/examples/image.jpg"
            logger.info("No video_url was provided; falling back to bundled example image.")

    # Audio (URL-only)
    wav_path_1 = None
    wav_path_2 = None
    wav_url = ensure_url_only("wav_url", job_input)
    if wav_url:
        wav_path_1 = download_file_from_url(wav_url, os.path.join(task_id, "input_audio.wav"))
    else:
        wav_path_1 = "/examples/audio.mp3"
        logger.info("No wav_url provided; using bundled example audio.")

    if person_count == "multi":
        wav_url_2 = ensure_url_only("wav_url_2", job_input)
        if wav_url_2:
            wav_path_2 = download_file_from_url(wav_url_2, os.path.join(task_id, "input_audio_2.wav"))
        else:
            wav_path_2 = wav_path_1
            logger.info("No wav_url_2 provided; using wav_url for both speakers.")

    # Text and size
    prompt_text = job_input.get("prompt", "A person talking naturally")

    # Prefer aspect_ratio + resolution; else accept width/height if both provided; else default
    ar = job_input.get("aspect_ratio")
    res = job_input.get("resolution")
    if ar or res:
        width, height = pick_dimensions(ar, res, 512, 512)
    else:
        width = int(job_input.get("width", 512))
        height = int(job_input.get("height", 512))

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

    # Existence checks
    if not os.path.exists(media_local_path):
        return {"error": f"Media file not found: {media_local_path}"}
    if not os.path.exists(wav_path_1):
        return {"error": f"Audio file not found: {wav_path_1}"}
    if person_count == "multi" and wav_path_2 and not os.path.exists(wav_path_2):
        return {"error": f"Second audio file not found: {wav_path_2}"}

    logger.info(f"Media size: {os.path.getsize(media_local_path)} bytes")
    logger.info(f"Audio #1 size: {os.path.getsize(wav_path_1)} bytes")
    if person_count == "multi" and wav_path_2:
        logger.info(f"Audio #2 size: {os.path.getsize(wav_path_2)} bytes")

    # Bind inputs to workflow
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
        # I2V_multi.json uses node 307, V2V_multi.json uses node 313
        if input_type == "image" and "307" in prompt:
            prompt["307"]["inputs"]["audio"] = wav_path_2
        elif input_type == "video" and "313" in prompt:
            prompt["313"]["inputs"]["audio"] = wav_path_2

    # Connectivity checks
    ws_url = f"ws://{SERVER_ADDRESS}:8188/ws?clientId={CLIENT_ID}"
    http_url = f"http://{SERVER_ADDRESS}:8188/"
    logger.info(f"Connecting to ComfyUI: {http_url} (ws={ws_url})")

    # Wait for HTTP up (<= 3 min)
    for attempt in range(180):
        try:
            urllib.request.urlopen(http_url, timeout=5)
            logger.info(f"HTTP connection OK (attempt {attempt+1})")
            break
        except Exception as e:
            if attempt == 179:
                raise Exception("Cannot reach ComfyUI HTTP endpoint.")
            time.sleep(1)

    # WS connect
    ws = websocket.WebSocket()
    for attempt in range(36):  # up to ~3 min (5s backoff)
        try:
            ws.connect(ws_url)
            logger.info(f"WebSocket connected (attempt {attempt+1})")
            break
        except Exception as e:
            if attempt == 35:
                raise Exception("WebSocket connect timeout (~3 min).")
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
                return {"video": f"data:video/mp4;base64,{b64}"}

    return {"error": "No video file was produced by the workflow."}

runpod.serverless.start({"handler": handler})
