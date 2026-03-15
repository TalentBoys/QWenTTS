"""
Qwen3-TTS Streaming Server (Apple Silicon / MLX)

OpenAI-compatible /v1/audio/speech endpoint with:
  - Streaming audio output (PCM float32 chunks)
  - Streaming text input (text arrives incrementally, audio generated on the fly)
  - Voice cloning via ref_audio / ref_text

Usage:
  python server.py [--host 0.0.0.0] [--port 8000] [--model mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16]
"""

import argparse
import asyncio
import base64
import io
import json
import logging
import struct
import tempfile
import time
import wave
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import uvicorn
from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from starlette.requests import ClientDisconnect

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Globals (populated at startup)
# ---------------------------------------------------------------------------
MODEL = None
MODEL_PATH: str = ""
SAMPLE_RATE = 24000

# Cache for pre-loaded reference audio (mx.array) keyed by name
# Each entry: {"audio": mx.array, "text": str}
# Note: we only pass ref_audio (without ref_text) to generate() so the model
# uses the fast speaker-embedding path instead of the slow ICL path.
_ref_audio_cache: dict = {}  # name -> {"audio": mx.array, "text": str}


def load_tts_model(model_path: str):
    """Load the MLX Qwen3-TTS model."""
    global MODEL, MODEL_PATH
    from mlx_audio.tts.utils import load_model
    logger.info(f"Loading model: {model_path} ...")
    MODEL = load_model(model_path)
    MODEL_PATH = model_path
    logger.info("Model loaded successfully.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def numpy_to_wav_bytes(audio_np: np.ndarray, sr: int) -> bytes:
    """Convert a numpy float32 waveform to WAV bytes."""
    buf = io.BytesIO()
    sf.write(buf, audio_np, sr, format="WAV", subtype="PCM_16")
    buf.seek(0)
    return buf.read()


def numpy_to_pcm_bytes(audio_np: np.ndarray) -> bytes:
    """Convert numpy float32 array to raw PCM float32 LE bytes."""
    return audio_np.astype(np.float32).tobytes()


async def save_upload_to_tempfile(upload: UploadFile) -> str:
    """Save an UploadFile to a temp wav file and return the path."""
    suffix = Path(upload.filename or "audio.wav").suffix or ".wav"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    content = await upload.read()
    tmp.write(content)
    tmp.flush()
    tmp.close()
    return tmp.name


# ---------------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------------
app = FastAPI(title="Qwen3-TTS Streaming Server", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# POST /v1/audio/speech  — OpenAI-compatible TTS
# ---------------------------------------------------------------------------
@app.post("/v1/audio/speech")
async def audio_speech(request: Request):
    """
    OpenAI-compatible TTS endpoint.

    JSON body:
      - input (str): text to synthesise
      - model (str, optional): ignored (uses loaded model)
      - voice (str, optional): speaker name for CustomVoice, default "Chelsie"
      - language (str, optional): default "Chinese"
      - response_format (str): "wav" | "pcm"  (default "wav")
      - stream (bool): if true, stream raw PCM float32 chunks
      - ref_audio (str, optional): URL or base64 data-url of reference audio (for voice clone)
      - ref_text (str, optional): transcript of reference audio
      - speed (float, optional): speech speed, default 1.0
    """
    body = await request.json()

    text = body.get("input", "")
    if not text:
        raise HTTPException(400, "Missing 'input' field.")

    voice = body.get("voice", "Chelsie")
    language = body.get("language", "Chinese")
    response_format = body.get("response_format", "wav")
    stream = body.get("stream", False)
    ref_audio = body.get("ref_audio", None)
    ref_text = body.get("ref_text", None)
    speed = body.get("speed", 1.0)

    import mlx.core as mx

    # Resolve cached speaker if applicable
    # Only pass ref_audio (no ref_text) to force speaker-embedding path instead of ICL
    if voice in _ref_audio_cache and not ref_audio:
        cached = _ref_audio_cache[voice]
        ref_audio = cached["audio"]
        ref_text = None  # no ref_text → speaker embedding path (fast)
        logger.info(f"Using cached speaker '{voice}' via speaker-embedding path")

    # ----- Streaming PCM output -----
    if stream:
        async def generate_stream():
            try:
                gen_kwargs = dict(
                    text=text,
                    language=language,
                    verbose=False,
                )

                if ref_audio is not None:
                    gen_kwargs["ref_audio"] = ref_audio
                    if ref_text:
                        gen_kwargs["ref_text"] = ref_text
                else:
                    gen_kwargs["voice"] = voice

                results = MODEL.generate(**gen_kwargs)

                for result in results:
                    audio = result.audio
                    if hasattr(audio, "tolist"):
                        audio_np = np.array(audio, dtype=np.float32)
                    else:
                        audio_np = np.array(audio, dtype=np.float32)
                    yield audio_np.tobytes()

            except Exception as e:
                logger.error(f"Streaming generation error: {e}", exc_info=True)

        return StreamingResponse(
            generate_stream(),
            media_type="application/octet-stream",
            headers={
                "X-Sample-Rate": str(SAMPLE_RATE),
                "X-Audio-Format": "pcm_f32le",
            },
        )

    # ----- Non-streaming: generate full audio then return -----
    try:
        gen_kwargs = dict(
            text=text,
            language=language,
            verbose=False,
        )

        if ref_audio is not None:
            gen_kwargs["ref_audio"] = ref_audio
            if ref_text:
                gen_kwargs["ref_text"] = ref_text
        else:
            gen_kwargs["voice"] = voice

        results = list(MODEL.generate(**gen_kwargs))
        # Collect all audio chunks
        all_audio = []
        for result in results:
            audio = result.audio
            if hasattr(audio, "tolist"):
                audio_np = np.array(audio, dtype=np.float32)
            else:
                audio_np = np.array(audio, dtype=np.float32)
            all_audio.append(audio_np)

        if not all_audio:
            raise HTTPException(500, "No audio generated.")

        full_audio = np.concatenate(all_audio) if len(all_audio) > 1 else all_audio[0]

        if response_format == "pcm":
            return Response(
                content=full_audio.astype(np.float32).tobytes(),
                media_type="application/octet-stream",
                headers={
                    "X-Sample-Rate": str(SAMPLE_RATE),
                    "X-Audio-Format": "pcm_f32le",
                },
            )

        # Default: WAV
        wav_bytes = numpy_to_wav_bytes(full_audio, SAMPLE_RATE)
        return Response(content=wav_bytes, media_type="audio/wav")

    except Exception as e:
        logger.error(f"Generation error: {e}", exc_info=True)
        raise HTTPException(500, str(e))


# ---------------------------------------------------------------------------
# POST /v1/audio/speech/stream-text — Streaming text input
# ---------------------------------------------------------------------------
@app.post("/v1/audio/speech/stream-text")
async def stream_text_speech(request: Request):
    """
    Streaming text input endpoint.

    The client sends text chunks via a streaming request body (newline-delimited JSON).
    Each line is a JSON object: {"text": "partial text", "done": false}
    The last chunk should have "done": true.

    The server accumulates text and generates audio incrementally,
    streaming PCM float32 chunks back as they become available.

    Query params or JSON body:
      - voice (str): speaker name, default "Chelsie"
      - language (str): default "Chinese"
      - ref_audio (str, optional): reference audio URL/base64
      - ref_text (str, optional): reference text
    """
    # Try to read initial config from query params
    voice = request.query_params.get("voice", "Chelsie")
    language = request.query_params.get("language", "Chinese")
    ref_audio = request.query_params.get("ref_audio", None)
    ref_text = request.query_params.get("ref_text", None)

    import mlx.core as mx

    async def generate_from_streaming_text():
        """
        Read text chunks from the request body and generate audio incrementally.
        Strategy: accumulate text, when we see sentence-ending punctuation or 'done',
        synthesize that segment and stream back audio.
        """
        buffer = ""
        sentence_enders = {"。", "！", "？", ".", "!", "?", "\n", "；", ";", "，", ",", "：", ":", "—"}
        received_any = False

        try:
            async for chunk in request.stream():
                text_data = chunk.decode("utf-8", errors="ignore")
                logger.info(f"stream-text received chunk ({len(chunk)} bytes): {text_data[:100]!r}")
                received_any = True

                # Support both newline-delimited JSON and plain text
                for line in text_data.split("\n"):
                    line = line.strip()
                    if not line:
                        continue

                    # Try JSON format
                    try:
                        obj = json.loads(line)
                        text_piece = obj.get("text", "")
                        done = obj.get("done", False)
                    except (json.JSONDecodeError, TypeError):
                        # Plain text mode
                        text_piece = line
                        done = False

                    buffer += text_piece

                    # Check if we have a complete sentence to synthesize
                    segments_to_synth = []
                    while buffer:
                        # Find the earliest sentence ender
                        earliest_pos = -1
                        for ender in sentence_enders:
                            pos = buffer.find(ender)
                            if pos != -1 and (earliest_pos == -1 or pos < earliest_pos):
                                earliest_pos = pos

                        if earliest_pos != -1:
                            # Extract sentence up to and including the ender
                            segment = buffer[: earliest_pos + 1].strip()
                            buffer = buffer[earliest_pos + 1 :]
                            if segment:
                                segments_to_synth.append(segment)
                        else:
                            break

                    # Synthesize each complete sentence
                    for segment in segments_to_synth:
                        logger.info(f"Synthesizing segment: '{segment[:60]}'")
                        async for audio_bytes in _synthesize_segment(
                            segment, voice, language, ref_audio, ref_text
                        ):
                            yield audio_bytes

                    # Force-flush if buffer is long but has no punctuation
                    if len(buffer.strip()) > 30:
                        flush_text = buffer.strip()
                        buffer = ""
                        logger.info(f"Force-flushing long buffer: '{flush_text[:60]}'")
                        async for audio_bytes in _synthesize_segment(
                            flush_text, voice, language, ref_audio, ref_text
                        ):
                            yield audio_bytes

                    # If done, flush remaining buffer
                    if done and buffer.strip():
                        async for audio_bytes in _synthesize_segment(
                            buffer.strip(), voice, language, ref_audio, ref_text
                        ):
                            yield audio_bytes
                        buffer = ""
        except ClientDisconnect:
            logger.info("Client disconnected during stream-text")
            return

        if not received_any:
            logger.warning("stream-text: request body was empty — no chunks received")

        # Flush any remaining text
        if buffer.strip():
            async for audio_bytes in _synthesize_segment(
                buffer.strip(), voice, language, ref_audio, ref_text
            ):
                yield audio_bytes

    return StreamingResponse(
        generate_from_streaming_text(),
        media_type="application/octet-stream",
        headers={
            "X-Sample-Rate": str(SAMPLE_RATE),
            "X-Audio-Format": "pcm_f32le",
        },
    )


async def _synthesize_segment(text, voice, language, ref_audio, ref_text):
    """Synthesize a text segment and yield PCM audio bytes."""
    try:
        gen_kwargs = dict(
            text=text,
            language=language,
            verbose=False,
        )

        # Check if voice matches a cached clone speaker
        # Only pass ref_audio (no ref_text) → speaker-embedding path (fast)
        if voice in _ref_audio_cache and not ref_audio:
            cached = _ref_audio_cache[voice]
            gen_kwargs["ref_audio"] = cached["audio"]
            # no ref_text → forces speaker embedding instead of ICL
            logger.info(f"Using cached speaker '{voice}' via speaker-embedding path")
        elif ref_audio:
            gen_kwargs["ref_audio"] = ref_audio
            if ref_text:
                gen_kwargs["ref_text"] = ref_text
        else:
            gen_kwargs["voice"] = voice

        results = MODEL.generate(**gen_kwargs)
        for result in results:
            audio = result.audio
            audio_np = np.array(audio, dtype=np.float32)
            yield audio_np.tobytes()

    except Exception as e:
        logger.error(f"Segment synthesis error for '{text[:50]}...': {e}", exc_info=True)


# ---------------------------------------------------------------------------
# POST /v1/audio/speech/clone — Upload ref audio + generate
# ---------------------------------------------------------------------------
@app.post("/v1/audio/speech/clone")
async def clone_voice_speech(
    text: str = Form(...),
    language: str = Form("Chinese"),
    ref_audio: UploadFile = File(...),
    ref_text: str = Form(...),
    response_format: str = Form("wav"),
    stream: str = Form("false"),
):
    """
    Voice clone endpoint that accepts audio file upload.

    Form data:
      - text: text to synthesize
      - language: "Chinese", "English", etc.
      - ref_audio: reference audio file (wav/mp3)
      - ref_text: transcript of reference audio
      - response_format: "wav" or "pcm"
      - stream: "true" or "false"
    """
    ref_path = await save_upload_to_tempfile(ref_audio)
    is_stream = stream.lower() == "true"

    import mlx.core as mx

    gen_kwargs = dict(
        text=text,
        language=language,
        ref_audio=ref_path,
        ref_text=ref_text,
        verbose=False,
    )

    if is_stream:
        async def stream_clone():
            try:
                results = MODEL.generate(**gen_kwargs)
                for result in results:
                    audio = result.audio
                    audio_np = np.array(audio, dtype=np.float32)
                    yield audio_np.tobytes()
            except Exception as e:
                logger.error(f"Clone streaming error: {e}", exc_info=True)
            finally:
                Path(ref_path).unlink(missing_ok=True)

        return StreamingResponse(
            stream_clone(),
            media_type="application/octet-stream",
            headers={
                "X-Sample-Rate": str(SAMPLE_RATE),
                "X-Audio-Format": "pcm_f32le",
            },
        )

    try:
        results = list(MODEL.generate(**gen_kwargs))
        all_audio = []
        for result in results:
            audio = result.audio
            audio_np = np.array(audio, dtype=np.float32)
            all_audio.append(audio_np)

        full_audio = np.concatenate(all_audio) if len(all_audio) > 1 else all_audio[0]

        Path(ref_path).unlink(missing_ok=True)

        if response_format == "pcm":
            return Response(
                content=full_audio.astype(np.float32).tobytes(),
                media_type="application/octet-stream",
                headers={"X-Sample-Rate": str(SAMPLE_RATE)},
            )

        wav_bytes = numpy_to_wav_bytes(full_audio, SAMPLE_RATE)
        return Response(content=wav_bytes, media_type="audio/wav")

    except Exception as e:
        Path(ref_path).unlink(missing_ok=True)
        logger.error(f"Clone generation error: {e}", exc_info=True)
        raise HTTPException(500, str(e))


# ---------------------------------------------------------------------------
# GET /v1/models — list available models
# ---------------------------------------------------------------------------
@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_PATH,
                "object": "model",
                "owned_by": "qwen",
                "type": "tts",
            }
        ],
    }


# ---------------------------------------------------------------------------
# POST /v1/speakers — Register a cloned voice (pre-load ref audio into cache)
# ---------------------------------------------------------------------------
@app.post("/v1/speakers")
async def register_speaker(
    name: str = Form(...),
    ref_audio: UploadFile = File(...),
    ref_text: str = Form(...),
):
    """
    Pre-load reference audio so subsequent TTS calls using this speaker name
    skip the file-read + resample overhead.

    After registration, use voice=<name> (without ref_audio/ref_text) and
    the cached mx.array will be used automatically.
    """
    from mlx_audio.utils import load_audio as _load_audio
    import mlx.core as mx

    ref_path = await save_upload_to_tempfile(ref_audio)
    try:
        audio_mx = _load_audio(ref_path, sample_rate=SAMPLE_RATE)
        mx.eval(audio_mx)
        _ref_audio_cache[name] = {"audio": audio_mx, "text": ref_text}
        # Verify speaker embedding can be extracted (warm up the encoder)
        if hasattr(MODEL, "extract_speaker_embedding"):
            try:
                emb = MODEL.extract_speaker_embedding(audio_mx)
                mx.eval(emb)
                logger.info(f"Registered speaker '{name}' — audio shape {audio_mx.shape}, embedding shape {emb.shape}")
            except Exception as emb_err:
                logger.warning(f"Speaker embedding extraction failed for '{name}': {emb_err}")
                logger.info(f"Registered speaker '{name}' — audio shape {audio_mx.shape} (will use ICL fallback)")
        else:
            logger.info(f"Registered speaker '{name}' — audio shape {audio_mx.shape}")
        return {"ok": True, "name": name, "cached_speakers": list(_ref_audio_cache.keys())}
    except Exception as e:
        logger.error(f"Failed to register speaker '{name}': {e}", exc_info=True)
        raise HTTPException(500, str(e))
    finally:
        import os
        os.unlink(ref_path)


@app.get("/v1/speakers")
async def list_speakers():
    """List built-in + cached speakers."""
    builtin = MODEL.get_supported_speakers() if MODEL else []
    cached = list(_ref_audio_cache.keys())
    return {"builtin": builtin, "cached": cached}


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------
@app.get("/health")
async def health():
    return {"status": "ok", "model": MODEL_PATH}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Qwen3-TTS Streaming Server")
    parser.add_argument(
        "--model",
        type=str,
        default="mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16",
        help="HuggingFace model ID or local path",
    )
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    load_tts_model(args.model)

    logger.info(f"Starting server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
