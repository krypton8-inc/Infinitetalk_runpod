import runpod
from runpod.serverless.utils import rp_upload  # kept for compatibility if you use it elsewhere
import os
import websocket
import base64
import json
import uuid
import logging
import urllib.request
import urllib.parse
import binascii  # for Base64 errors
import subprocess
import time
import librosa
from urllib.parse import urlparse

# Optional S3 (only used if all env vars are present)
try:
    import boto3
    from botocore.client import Config as BotoConfig
except Exception:
    boto3 = None

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

server_address = os.getenv("SERVER_ADDRESS", "127.0.0.1")
client_id = str(uuid.uuid4())

# ------------------------------------------------------------------
# Default sample assets - test assets hosted in a public S3 bucket
# If your bucket becomes private, update these URLs or remove defaults
# ------------------------------------------------------------------
DEFAULT_IMAGE_URL = "https://krypton8.s3.us-east-1.amazonaws.com/test_image.png"
DEFAULT_AUDIO_URL = "https://krypton8.s3.us-east-1.amazonaws.com/test_audio.wav"
DEFAULT_MULTI_IMAGE_URL = "https://krypton8.s3.us-east-1.amazonaws.com/multi_test_image.png"
DEFAULT_MULTI_AUDIO_URL_1 = "https://krypton8.s3.us-east-1.amazonaws.com/multi_test_audio2.wav"
DEFAULT_MULTI_AUDIO_URL_2 = "https://krypton8.s3.us-east-1.amazonaws.com/multi_test_audio.wav"

# -----------------------------
# S3 helpers
# -----------------------------
def s3_configured() -> bool:
    return all(os.getenv(k) for k in ["S3_REGION", "S3_BUCKET", "S3_KEY", "S3_SECRET"])

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
    Upload a file to S3 and return a public URL.
    If S3_PUBLIC_BASE_URL is set, it will be used as the prefix (e.g., CloudFront).
    Otherwise a standard S3 URL is returned.
    """
    bucket = os.getenv("S3_BUCKET")
    region = os.getenv("S3_REGION")
    key_prefix = os.getenv("S3_PREFIX", "infinitetalk")
    basename = os.path.basename(file_path)
    key = f"{key_prefix.rstrip('/')}/{uuid.uuid4()}_{basename}"

    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    c = s3_client()

    # Best-effort content type
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
# Download and input helpers
# -----------------------------
def download_file_from_url(url, output_path):
    """Download a file from a URL to output_path using wget."""
    try:
        # Ensure target directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

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
            logger.error(f"wget failed: {result.stderr}")
            raise Exception(f"URL download failed: {result.stderr}")
    except subprocess.TimeoutExpired:
        logger.error("Download timeout")
        raise Exception("Download timeout")
    except Exception as e:
        logger.error(f"Download error: {e}")
        raise Exception(f"Download error: {e}")

def save_base64_to_file(base64_data, temp_dir, output_filename):
    """Decode base64 and save to file."""
    try:
        decoded_data = base64.b64decode(base64_data)
        os.makedirs(temp_dir, exist_ok=True)
        file_path = os.path.abspath(os.path.join(temp_dir, output_filename))
        with open(file_path, "wb") as f:
            f.write(decoded_data)
        logger.info(f"Saved base64 input to file: {file_path}")
        return file_path
    except (binascii.Error, ValueError) as e:
        logger.error(f"Base64 decode failed: {e}")
        raise Exception(f"Base64 decode failed: {e}")

def process_input(input_data, temp_dir, output_filename, input_type):
    """
    Turn input_data into a local file path.
    input_type: 'path' | 'url' | 'base64'
    """
    if input_type == "path":
        logger.info(f"Using provided path: {input_data}")
        return input_data
    elif input_type == "url":
        logger.info(f"Downloading from URL: {input_data}")
        os.makedirs(temp_dir, exist_ok=True)
        file_path = os.path.abspath(os.path.join(temp_dir, output_filename))
        return download_file_from_url(input_data, file_path)
    elif input_type == "base64":
        logger.info("Decoding base64 input")
        return save_base64_to_file(input_data, temp_dir, output_filename)
    else:
        raise Exception(f"Unsupported input type: {input_type}")

# -----------------------------
# ComfyUI prompt I/O
# -----------------------------
def queue_prompt(prompt, input_type="image", person_count="single"):
    url = f"http://{server_address}:8188/prompt"
    logger.info(f"Queueing prompt to: {url}")
    p = {"prompt": prompt, "client_id": client_id}
    data = json.dumps(p).encode("utf-8")

    # Debug
    logger.info(f"Workflow node count: {len(prompt)}")
    if input_type == "image":
        logger.info(f"Image node(284): {prompt.get('284', {}).get('inputs', {}).get('image', 'NOT_FOUND')}")
    else:
        logger.info(f"Video node(228): {prompt.get('228', {}).get('inputs', {}).get('video', 'NOT_FOUND')}")
    logger.info(f"Audio node(125): {prompt.get('125', {}).get('inputs', {}).get('audio', 'NOT_FOUND')}")
    logger.info(f"Text node(241): {prompt.get('241', {}).get('inputs', {}).get('positive_prompt', 'NOT_FOUND')}")
    if person_count == "multi":
        if "307" in prompt:
            logger.info(f"Second audio node(307): {prompt.get('307', {}).get('inputs', {}).get('audio', 'NOT_FOUND')}")
        elif "313" in prompt:
            logger.info(f"Second audio node(313): {prompt.get('313', {}).get('inputs', {}).get('audio', 'NOT_FOUND')}")

    req = urllib.request.Request(url, data=data)
    req.add_header("Content-Type", "application/json")

    try:
        response = urllib.request.urlopen(req)
        result = json.loads(response.read())
        logger.info(f"Prompt queued OK: {result}")
        return result
    except urllib.error.HTTPError as e:
        logger.error(f"HTTP error: {e.code} - {e.reason}")
        logger.error(f"Response body: {e.read().decode('utf-8')}")
        raise
    except Exception as e:
        logger.error(f"Error queueing prompt: {e}")
        raise

def get_history(prompt_id):
    url = f"http://{server_address}:8188/history/{prompt_id}"
    logger.info(f"Getting history from: {url}")
    with urllib.request.urlopen(url) as response:
        return json.loads(response.read())

def get_video_filepaths(ws, prompt, input_type="image", person_count="single"):
    """
    Submit prompt, wait for completion, then read ComfyUI history and
    collect output video file paths from VHS VideoCombine nodes ("gifs" array with fullpath).
    Returns dict[node_id] -> [file_path, ...]
    """
    result = queue_prompt(prompt, input_type, person_count)
    prompt_id = result["prompt_id"]
    outputs = {}

    while True:
        out = ws.recv()
        if isinstance(out, str):
            message = json.loads(out)
            if message.get("type") == "executing":
                data = message.get("data", {})
                if data.get("node") is None and data.get("prompt_id") == prompt_id:
                    break
        # ignore binary

    history = get_history(prompt_id)[prompt_id]
    for node_id, node_output in history.get("outputs", {}).items():
        paths = []
        if "gifs" in node_output:
            for vid in node_output["gifs"]:
                if "fullpath" in vid and os.path.exists(vid["fullpath"]):
                    paths.append(vid["fullpath"])
        outputs[node_id] = paths

    return outputs

def load_workflow(workflow_path):
    with open(workflow_path, "r") as file:
        return json.load(file)

def get_workflow_path(input_type, person_count):
    """Return the workflow file path based on input_type and person_count."""
    if input_type == "image":
        return "/I2V_single.json" if person_count == "single" else "/I2V_multi.json"
    else:
        return "/V2V_single.json" if person_count == "single" else "/V2V_multi.json"

# -----------------------------
# Audio utilities
# -----------------------------
def get_audio_duration(audio_path):
    """Return duration in seconds for an audio file."""
    try:
        duration = librosa.get_duration(path=audio_path)
        return duration
    except Exception as e:
        logger.warning(f"Failed to get audio duration ({audio_path}): {e}")
        return None

def calculate_max_frames_from_audio(wav_path, wav_path_2=None, fps=25):
    """Calculate max_frames based on the longest audio."""
    durations = []
    d1 = get_audio_duration(wav_path)
    if d1 is not None:
        durations.append(d1)
        logger.info(f"Audio #1 duration: {d1:.2f}s")
    if wav_path_2:
        d2 = get_audio_duration(wav_path_2)
        if d2 is not None:
            durations.append(d2)
            logger.info(f"Audio #2 duration: {d2:.2f}s")
    if not durations:
        logger.warning("Could not determine audio duration - using default 81 frames.")
        return 81
    max_duration = max(durations)
    max_frames = int(max_duration * fps) + 81
    logger.info(f"Longest audio {max_duration:.2f}s -> max_frames {max_frames}")
    return max_frames

# -----------------------------
# Main handler
# -----------------------------
def handler(job):
    job_input = job.get("input", {})
    logger.info(f"Received job input: {job_input}")

    task_id = f"task_{uuid.uuid4()}"

    # Input type and number of persons
    input_type = job_input.get("input_type", "image")  # "image" or "video"
    person_count = job_input.get("person_count", "single")  # "single" or "multi"
    logger.info(f"Workflow: type={input_type}, persons={person_count}")

    # Workflow file path
    workflow_path = get_workflow_path(input_type, person_count)
    logger.info(f"Using workflow: {workflow_path}")

    # -----------------------------
    # Media input
    # -----------------------------
    media_path = None
    if input_type == "image":
        if "image_path" in job_input:
            media_path = process_input(job_input["image_path"], task_id, "input_image.jpg", "path")
        elif "image_url" in job_input:
            media_path = process_input(job_input["image_url"], task_id, "input_image.jpg", "url")
        elif "image_base64" in job_input:
            media_path = process_input(job_input["image_base64"], task_id, "input_image.jpg", "base64")
        else:
            # Default to S3-hosted sample image
            default_img = DEFAULT_MULTI_IMAGE_URL if person_count == "multi" else DEFAULT_IMAGE_URL
            media_path = process_input(default_img, task_id, "input_image.jpg", "url")
            logger.info("No image provided - using default S3 image URL.")
    else:
        if "video_path" in job_input:
            media_path = process_input(job_input["video_path"], task_id, "input_video.mp4", "path")
        elif "video_url" in job_input:
            media_path = process_input(job_input["video_url"], task_id, "input_video.mp4", "url")
        elif "video_base64" in job_input:
            media_path = process_input(job_input["video_base64"], task_id, "input_video.mp4", "base64")
        else:
            # Keep original behavior - fall back to an image if no video provided
            default_img = DEFAULT_MULTI_IMAGE_URL if person_count == "multi" else DEFAULT_IMAGE_URL
            media_path = process_input(default_img, task_id, "input_image.jpg", "url")
            logger.info("No video provided - falling back to default S3 image URL.")

    # -----------------------------
    # Audio input
    # -----------------------------
    wav_path = None
    wav_path_2 = None

    if "wav_path" in job_input:
        wav_path = process_input(job_input["wav_path"], task_id, "input_audio.wav", "path")
    elif "wav_url" in job_input:
        wav_path = process_input(job_input["wav_url"], task_id, "input_audio.wav", "url")
    elif "wav_base64" in job_input:
        wav_path = process_input(job_input["wav_base64"], task_id, "input_audio.wav", "base64")
    else:
        # Default to S3-hosted sample audio
        default_audio_1 = DEFAULT_MULTI_AUDIO_URL_1 if person_count == "multi" else DEFAULT_AUDIO_URL
        wav_path = process_input(default_audio_1, task_id, "input_audio.wav", "url")
        logger.info("No wav provided - using default S3 audio URL.")

    if person_count == "multi":
        if "wav_path_2" in job_input:
            wav_path_2 = process_input(job_input["wav_path_2"], task_id, "input_audio_2.wav", "path")
        elif "wav_url_2" in job_input:
            wav_path_2 = process_input(job_input["wav_url_2"], task_id, "input_audio_2.wav", "url")
        elif "wav_base64_2" in job_input:
            wav_path_2 = process_input(job_input["wav_base64_2"], task_id, "input_audio_2.wav", "base64")
        else:
            # Use second default S3 audio for multi
            wav_path_2 = process_input(DEFAULT_MULTI_AUDIO_URL_2, task_id, "input_audio_2.wav", "url")
            logger.info("No second wav provided - using default S3 audio URL for #2.")

    # --------------------------------
    # Prompt and sizing - keep as-is
    # --------------------------------
    prompt_text = job_input.get("prompt", "A person talking naturally")
    width = job_input.get("width", 512)
    height = job_input.get("height", 512)

    max_frame = job_input.get("max_frame")
    if max_frame is None:
        logger.info("max_frame not provided - deriving from audio duration.")
        max_frame = calculate_max_frames_from_audio(wav_path, wav_path_2 if person_count == "multi" else None)
    else:
        logger.info(f"Using user-specified max_frame: {max_frame}")

    logger.info(f"Settings: prompt='{prompt_text}', width={width}, height={height}, max_frame={max_frame}")
    logger.info(f"Media path: {media_path}")
    logger.info(f"Audio #1: {wav_path}")
    if person_count == "multi":
        logger.info(f"Audio #2: {wav_path_2}")

    # Load workflow
    prompt = load_workflow(workflow_path)

    # Existence checks
    if not os.path.exists(media_path):
        logger.error(f"Media file does not exist: {media_path}")
        return {"error": f"Media file not found: {media_path}"}
    if not os.path.exists(wav_path):
        logger.error(f"Audio file does not exist: {wav_path}")
        return {"error": f"Audio file not found: {wav_path}"}
    if person_count == "multi" and wav_path_2 and not os.path.exists(wav_path_2):
        logger.error(f"Second audio file does not exist: {wav_path_2}")
        return {"error": f"Second audio file not found: {wav_path_2}"}

    logger.info(f"Media size: {os.path.getsize(media_path)} bytes")
    logger.info(f"Audio #1 size: {os.path.getsize(wav_path)} bytes")
    if person_count == "multi" and wav_path_2:
        logger.info(f"Audio #2 size: {os.path.getsize(wav_path_2)} bytes")

    # Bind inputs to workflow
    if input_type == "image":
        prompt["284"]["inputs"]["image"] = media_path
    else:
        prompt["228"]["inputs"]["video"] = media_path

    prompt["125"]["inputs"]["audio"] = wav_path
    prompt["241"]["inputs"]["positive_prompt"] = prompt_text
    prompt["245"]["inputs"]["value"] = width
    prompt["246"]["inputs"]["value"] = height
    prompt["270"]["inputs"]["value"] = max_frame

    if person_count == "multi":
        if input_type == "image":  # I2V_multi.json
            if "307" in prompt:
                prompt["307"]["inputs"]["audio"] = wav_path_2
        else:  # V2V_multi.json
            if "313" in prompt:
                prompt["313"]["inputs"]["audio"] = wav_path_2

    # Connectivity checks
    ws_url = f"ws://{server_address}:8188/ws?clientId={client_id}"
    logger.info(f"Connecting to WebSocket: {ws_url}")

    http_url = f"http://{server_address}:8188/"
    logger.info(f"Checking HTTP connection: {http_url}")

    # Wait for HTTP up (<= 3 minutes)
    max_http_attempts = 180
    for http_attempt in range(max_http_attempts):
        try:
            urllib.request.urlopen(http_url, timeout=5)
            logger.info(f"HTTP OK (attempt {http_attempt+1})")
            break
        except Exception as e:
            logger.warning(f"HTTP connect failed (attempt {http_attempt+1}/{max_http_attempts}): {e}")
            if http_attempt == max_http_attempts - 1:
                return {"error": "Cannot reach ComfyUI HTTP endpoint."}
            time.sleep(1)

    ws = websocket.WebSocket()
    # WebSocket connect - up to 3 minutes
    max_attempts = int(180 / 5)
    for attempt in range(max_attempts):
        try:
            ws.connect(ws_url)
            logger.info(f"WebSocket connected (attempt {attempt+1})")
            break
        except Exception as e:
            logger.warning(f"WebSocket connect failed (attempt {attempt+1}/{max_attempts}): {e}")
            if attempt == max_attempts - 1:
                return {"error": "WebSocket connect timeout (3 minutes)"}
            time.sleep(5)

    # Run and collect output paths
    video_paths_by_node = get_video_filepaths(ws, prompt, input_type, person_count)
    ws.close()

    # Return the first available video - upload to S3 if configured else base64
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

                with open(p, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode("utf-8")
                logger.info(f"Returning base64 video ({len(b64)} chars)")
                return {"video": f"data:video/mp4;base64,{b64}"}

    return {"error": "No video file was produced by the workflow."}

runpod.serverless.start({"handler": handler})
