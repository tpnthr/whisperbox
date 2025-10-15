# whisperbox

Dockerized REST API serving the Whisper large-v3-turbo speech-to-text model with GPU acceleration.

## Why CUDA 12.4 and cuDNN 9

* **CUDA 12.4** is the most recent long-lived release that has stable driver/toolkit support across Ampere, Ada, and Blackwell GPUs (including the RTX 5090) while still being supported by the official PyTorch wheels.
* **cuDNN 9.1 (libcudnn 9.x for CUDA 12.x)** ships in the `nvidia/cuda:12.4.1-cudnn9-runtime-ubuntu22.04` base image. cuDNN 9 introduced optimizations for larger transformer models such as Whisper and provides full compatibility with CUDA 12.4.
* **PyTorch 2.4.0 / Torchaudio 2.4.0** are the first releases published with prebuilt CUDA 12.4 wheels. They fully support FP16 execution on RTX 50-series GPUs, enabling the `float16` compute type used by Whisper for high throughput.

This combination is the most forward-compatible option currently available without building PyTorch from source while still matching the user's target libcudnn 12.x series.

## Building the container

```bash
docker build -t whisperbox:latest .
```

> **Requirements:** Install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) on the Ubuntu 24.04 host so Docker can expose your RTX 5090 to the container.

## Running the API

```bash
docker run --gpus all -p 8000:8000 \
  -e WHISPER_COMPUTE_TYPE=float16 \
  whisperbox:latest
```

Environment variables:

| Variable | Default | Description |
| --- | --- | --- |
| `WHISPER_MODEL_ID` | `large-v3-turbo` | Hugging Face model id to load. |
| `WHISPER_DEVICE` | `cuda` | Device passed to `faster-whisper`. Use `cpu` to run without a GPU. |
| `WHISPER_COMPUTE_TYPE` | `float16` | Controls precision (`float16`, `int8_float16`, etc.). |
| `WHISPER_VAD_FILTER` | `true` | Enable/disable built-in VAD preprocessing. |
| `PORT` | `8000` | Override the port used by Uvicorn if desired. |

The container exposes a FastAPI application compatible with the OpenAI Whisper REST contract:

* `POST /v1/audio/transcriptions`
* `POST /v1/audio/translations`
* `GET /health`

## Example request

```bash
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -H "Accept: application/json" \
  -F "file=@sample.mp3" \
  -F "model=large-v3-turbo" \
  -F "response_format=verbose_json" \
  -F "temperature=0" \
  -F "timestamp_granularities[]=segment" \
  -F "timestamp_granularities[]=word"
```

Example JSON response snippet:

```json
{
  "text": "hello world",
  "language": "en",
  "segments": [
    {
      "id": 0,
      "seek": 0,
      "start": 0.0,
      "end": 1.2,
      "text": " hello world",
      "tokens": [50257, 50364],
      "temperature": 0.0,
      "avg_logprob": -0.05,
      "compression_ratio": 0.91,
      "no_speech_prob": 0.01
    }
  ],
  "words": [
    {"word": "hello", "start": 0.0, "end": 0.6, "probability": 0.98},
    {"word": "world", "start": 0.6, "end": 1.2, "probability": 0.97}
  ]
}
```

The API also supports `response_format=text`, `response_format=srt`, and `response_format=vtt` to mirror the OpenAI Whisper interface shown in the provided screenshot.
