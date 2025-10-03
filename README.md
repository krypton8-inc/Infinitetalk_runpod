# InfiniteTalk for RunPod Serverless

This project is a template to deploy and run **InfiniteTalk** on **RunPod Serverless** with **ComfyUI**. It supports both **Image→Video (I2V)** and **Video→Video (V2V)** talking-head generation, including **single** and **multi-person** modes.

> **What’s new**
>
> - 480p **and** 720p output (auto-size from `aspect_ratio` + `resolution`).
> - URL-only inputs (no base64 and no local path inputs).
> - S3 upload support for results (optional).
> - Wider GPU support & auto fallbacks to avoid kernel-mismatch and OOM on smaller GPUs.

---

## Features

- **Infinite Talking**: generates long-form videos synchronized to audio.
- **I2V & V2V**: feed a single portrait or an input video.
- **Single & Multi-person**: one or two speakers.
- **Auto sizing**: choose `aspect_ratio` (`"9:16"` or `"16:9"`) and `resolution` (`"480p"` or `"720p"`). The worker maps to:
  - `16:9` + `480p` → **854×480**
  - `9:16` + `480p` → **480×854**
  - `16:9` + `720p` → **1280×720**
  - `9:16` + `720p` → **720×1280**
- **GPU aware**: uses attention and precision fallbacks to run on GPUs from 24 GB up to 180 GB VRAM.
- **S3 output**: set `S3_REGION`, `S3_BUCKET`, `S3_KEY`, `S3_SECRET` to upload the result and return an S3 URL instead of base64.

> **Note**: Inputs are **URL-only** for images/videos and audio. **Base64 and local paths are not supported.**

---

## Supported GPUs

You can target any of these classes on RunPod (or let RunPod choose). The worker adjusts kernels/attention paths automatically and will fall back safely when needed.

|  VRAM | GPU Class             | Notes                         | Typical Price\*  |
| ----: | --------------------- | ----------------------------- | ---------------- |
| 180GB | B200                  | Max throughput for big models | \$4.46–\$5.58/hr |
| 141GB | H200                  | Extreme throughput            | \$3.35–\$4.18/hr |
|  80GB | H100 PRO              | Extreme throughput            | \$2.17–\$2.72/hr |
|  80GB | A100                  | Great perf/value              | \$1.33–\$1.90/hr |
|  48GB | L40/L40S/6000 Ada PRO | Very fast inference           | \$0.85–\$1.22/hr |
|  48GB | A6000/A40             | Cost‑effective big models     | \$1.11–\$1.58/hr |
|  32GB | 5090 PRO              | Fast for small/medium         | \$0.77–\$1.10/hr |
|  24GB | 4090 PRO              | Fast for small/medium         | \$0.48–\$0.69/hr |
|  24GB | L4/A5000/3090         | Good for medium               | \$0.40–\$0.58/hr |

\*Prices are indicative, subject to change by provider.

---

## API

### Input

All media inputs must be **URLs**.

| Field          | Type    |       Required | Default                        | Description                        |
| -------------- | ------- | -------------: | ------------------------------ | ---------------------------------- |
| `input_type`   | string  |             no | `"image"`                      | `"image"` (I2V) or `"video"` (V2V) |
| `person_count` | string  |             no | `"single"`                     | `"single"` or `"multi"`            |
| `image_url`    | string  |   _yes if I2V_ | —                              | Image URL for I2V                  |
| `video_url`    | string  |   _yes if V2V_ | —                              | Video URL for V2V                  |
| `wav_url`      | string  |            yes | —                              | Audio URL for first speaker        |
| `wav_url_2`    | string  | _yes if multi_ | —                              | Audio URL for second speaker       |
| `prompt`       | string  |             no | `"A person talking naturally"` | Text guidance                      |
| `aspect_ratio` | string  |             no | `"9:16"`                       | `"16:9"` or `"9:16"`               |
| `resolution`   | string  |             no | `"480p"`                       | `"480p"` or `"720p"`               |
| `max_frame`    | integer |             no | auto from audio                | Maximum frames to render           |

### Output

- If S3 env vars are set: `{ "video_url": "s3://bucket/path/file.mp4" }`
- Else: `{ "video": "data:video/mp4;base64,..." }`

---

## Examples

### 1) I2V Single (URL inputs)

```json
{
  "input": {
    "input_type": "image",
    "person_count": "single",
    "prompt": "A person is talking in a natural way.",
    "image_url": "https://example.com/portrait.jpg",
    "wav_url": "https://example.com/audio.wav",
    "aspect_ratio": "9:16",
    "resolution": "480p"
  }
}
```

### 2) I2V Multi (URL inputs)

```json
{
  "input": {
    "input_type": "image",
    "person_count": "multi",
    "prompt": "Two people having a conversation.",
    "image_url": "https://example.com/portrait.jpg",
    "wav_url": "https://example.com/audio1.wav",
    "wav_url_2": "https://example.com/audio2.wav",
    "aspect_ratio": "16:9",
    "resolution": "720p"
  }
}
```

### 3) V2V Single (URL inputs)

```json
{
  "input": {
    "input_type": "video",
    "person_count": "single",
    "prompt": "A person is talking in a natural way.",
    "video_url": "https://example.com/input_video.mp4",
    "wav_url": "https://example.com/audio.wav",
    "aspect_ratio": "16:9",
    "resolution": "480p"
  }
}
```

### 4) V2V Multi (URL inputs)

```json
{
  "input": {
    "input_type": "video",
    "person_count": "multi",
    "prompt": "Two people talking in a video.",
    "video_url": "https://example.com/input_video.mp4",
    "wav_url": "https://example.com/audio1.wav",
    "wav_url_2": "https://example.com/audio2.wav",
    "aspect_ratio": "16:9",
    "resolution": "720p"
  }
}
```

---

## S3 Upload (optional)

If these env vars are present, results are uploaded to S3 and the API returns a URL:

- `S3_REGION`
- `S3_BUCKET`
- `S3_KEY`
- `S3_SECRET`

---

## Deploy on RunPod Serverless

1. Create a Serverless Endpoint from this repo.
2. Ensure your endpoint runs on one of the supported GPU classes.
3. (Optional) Set S3 env vars for URL outputs.
4. Send POST requests with the **Input** format above.

---

## License

- **InfiniteTalk**: Apache 2.0 (original authors)
- **This template**: Apache 2.0
