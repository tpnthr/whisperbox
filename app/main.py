import os
import tempfile
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse
from faster_whisper import WhisperModel


def _format_timestamp(seconds: float) -> str:
    milliseconds = int(round(seconds * 1000.0))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, milliseconds = divmod(remainder, 1_000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"


def _format_vtt_timestamp(seconds: float) -> str:
    milliseconds = int(round(seconds * 1000.0))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, milliseconds = divmod(remainder, 1_000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"


MODEL_ID = os.getenv("WHISPER_MODEL_ID", "large-v3-turbo")
DEVICE = os.getenv("WHISPER_DEVICE", "cuda")
COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "float16")
VAD_FILTER = os.getenv("WHISPER_VAD_FILTER", "true").lower() == "true"

app = FastAPI(title="Whisper v3 Turbo Service")

_whisper_model: WhisperModel | None = None


@app.on_event("startup")
async def _load_model() -> None:
    global _whisper_model
    _whisper_model = WhisperModel(
        MODEL_ID,
        device=DEVICE,
        compute_type=COMPUTE_TYPE,
        download_options={"hf_transfer": os.getenv("HF_HUB_ENABLE_HF_TRANSFER", "1") == "1"},
    )


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/audio/transcriptions")
async def transcribe(
    file: UploadFile = File(...),
    model: str = Form(MODEL_ID),
    prompt: Optional[str] = Form(default=None),
    response_format: str = Form(default="json"),
    temperature: float = Form(default=0.0),
    language: Optional[str] = Form(default=None),
    translate: bool = Form(default=False),
    timestamp_granularities: Optional[List[str]] = Form(default=None),
):
    if model != MODEL_ID:
        raise HTTPException(status_code=400, detail=f"Only model '{MODEL_ID}' is available")

    if _whisper_model is None:
        raise HTTPException(status_code=503, detail="Model is not ready yet")

    include_segments = False
    include_words = False
    if timestamp_granularities:
        include_segments = "segment" in timestamp_granularities
        include_words = "word" in timestamp_granularities

    tmp_path: Optional[str] = None
    try:
        suffix = Path(file.filename or "audio").suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        raw_segments, info = _whisper_model.transcribe(
            tmp_path,
            language=language,
            temperature=temperature,
            task="translate" if translate else "transcribe",
            initial_prompt=prompt,
            word_timestamps=include_words,
            vad_filter=VAD_FILTER,
        )
        segments_stream = list(raw_segments)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    segments = []
    words = []
    full_text_parts = []
    for idx, segment in enumerate(segments_stream):
        full_text_parts.append(segment.text)
        segment_payload = {
            "id": idx,
            "seek": segment.seek,
            "start": segment.start,
            "end": segment.end,
            "text": segment.text,
            "tokens": segment.tokens,
            "temperature": temperature,
            "avg_logprob": segment.avg_logprob,
            "compression_ratio": segment.compression_ratio,
            "no_speech_prob": segment.no_speech_prob,
        }
        segments.append(segment_payload)
        if include_words and segment.words:
            for word in segment.words:
                words.append(
                    {
                        "word": word.word,
                        "start": word.start,
                        "end": word.end,
                        "probability": word.probability,
                    }
                )

    full_text = "".join(full_text_parts).strip()

    base_payload = {
        "text": full_text,
        "language": info.language if info and getattr(info, "language", None) else language,
    }

    if response_format == "text":
        return PlainTextResponse(full_text)
    if response_format == "srt":
        srt_lines = []
        for index, segment in enumerate(segments, start=1):
            srt_lines.append(str(index))
            srt_lines.append(
                f"{_format_timestamp(segment['start'])} --> {_format_timestamp(segment['end'])}"
            )
            srt_lines.append(segment["text"].strip())
            srt_lines.append("")
        return PlainTextResponse("\n".join(srt_lines).strip())
    if response_format == "vtt":
        vtt_lines = ["WEBVTT", ""]
        for segment in segments:
            vtt_lines.append(
                f"{_format_vtt_timestamp(segment['start'])} --> {_format_vtt_timestamp(segment['end'])}"
            )
            vtt_lines.append(segment["text"].strip())
            vtt_lines.append("")
        return PlainTextResponse("\n".join(vtt_lines).strip())

    if response_format not in {"json", "verbose_json"}:
        raise HTTPException(status_code=400, detail="Unsupported response_format")

    if response_format == "verbose_json" or include_segments or include_words:
        verbose_payload = base_payload | {"segments": segments}
        if include_words:
            verbose_payload["words"] = words
        return JSONResponse(verbose_payload)

    return JSONResponse(base_payload)


@app.post("/v1/audio/translations")
async def translate(
    file: UploadFile = File(...),
    model: str = Form(MODEL_ID),
    prompt: Optional[str] = Form(default=None),
    response_format: str = Form(default="json"),
    temperature: float = Form(default=0.0),
    timestamp_granularities: Optional[List[str]] = Form(default=None),
):
    return await transcribe(
        file=file,
        model=model,
        prompt=prompt,
        response_format=response_format,
        temperature=temperature,
        language=None,
        translate=True,
        timestamp_granularities=timestamp_granularities,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
