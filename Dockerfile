# syntax=docker/dockerfile:1
FROM nvidia/cuda:12.4.1-cudnn9-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-venv \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv "$VIRTUAL_ENV"
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

RUN pip install --upgrade pip

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu124 \
    torch==2.4.0 \
    torchaudio==2.4.0

WORKDIR /app
COPY app /app

ENV PYTHONPATH=/app \
    WHISPER_DEVICE=cuda \
    WHISPER_COMPUTE_TYPE=float16 \
    HF_HUB_ENABLE_HF_TRANSFER=1

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
