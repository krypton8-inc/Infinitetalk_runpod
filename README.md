# InfiniteTalk for RunPod Serverless - RTX 5090 Edition

**Optimized exclusively for NVIDIA RTX 5090 (32GB, Blackwell architecture)**

High-performance talking-head video generation with **I2V** and **V2V** support, fine-tuned for maximum speed on the RTX 5090.

## GPU Requirements

- **Required**: NVIDIA RTX 5090 (32GB VRAM, SM 12.0)
- **Not supported**: Other GPUs (this build is 5090-specific)

## Features

- **Infinite Talking**: Long-form videos synchronized to audio
- **I2V & V2V**: Image→Video or Video→Video
- **Single & Multi-person**: 1 or 2 speakers
- **Resolution options**:
  - `16:9` + `480p` → **854×480** (fastest)
  - `9:16` + `480p` → **480×854** (fastest)
  - `16:9` + `720p` → **1280×720** (balanced)
  - `9:16` + `720p` → **720×1280** (balanced)
- **S3 output**: Optional S3 upload with URL return

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

- If S3 env vars are set: `{ "video_url": "https://bucket/path/file.mp4" }`
- Else: `{ "video": "data:video/mp4;base64,..." }`

## Deploy on RunPod Serverless

1. Create a Serverless Endpoint from this repo.
2. **Select RTX 5090 as your GPU** (required for this build).
3. (Optional) Set S3 env vars for URL outputs.
4. Send POST requests with the **Input** format above.

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
