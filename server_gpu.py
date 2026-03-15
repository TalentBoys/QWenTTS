"""
Qwen3-TTS Streaming Server (GPU / CUDA)

OpenAI-compatible /v1/audio/speech endpoint with:
  - Streaming audio output (PCM float32 chunks)
  - Streaming text input (text arrives incrementally, audio generated on the fly)
  - Voice cloning via ref_audio / ref_text

Usage:
  python server_gpu.py [--host 0.0.0.0] [--port 8000] [--model Qwen/Qwen3-TTS-12Hz-1.7B-Base] [--secret TOKEN]
"""

import argparse
import asyncio
import base64
import io
import json
import logging
import platform
import struct
import tempfile
import time
import wave
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from starlette.requests import ClientDisconnect

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Globals (populated at startup)
# ---------------------------------------------------------------------------
MODEL = None
MODEL_PATH: str = ""
SAMPLE_RATE = 24000
SECRET: Optional[str] = None

# Cache for pre-loaded reference audio (VoiceClonePromptItem) keyed by name
# Each entry: {"prompt": VoiceClonePromptItem, "text": str}
_ref_audio_cache: dict = {}


def detect_environment():
    """Detect current runtime environment and return info dict."""
    info = {"os": platform.system(), "mode": "unknown"}

    if platform.system() == "Darwin":
        info["mode"] = "macOS (Apple Silicon) — MLX mode"
    elif platform.system() == "Linux":
        try:
            with open("/proc/version") as f:
                if "microsoft" in f.read().lower():
                    info["mode"] = "WSL2 (Linux on Windows) — CUDA GPU mode"
                else:
                    info["mode"] = "Linux — CUDA GPU mode"
        except Exception:
            info["mode"] = "Linux — CUDA GPU mode"

    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
        info["vram"] = f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.0f}GB"
    else:
        info["gpu"] = "N/A (CPU mode)"
        info["vram"] = "N/A"

    return info


def print_environment(info: dict, model_path: str):
    """Print environment info banner."""
    print(f"\n[Environment] Running on: {info['mode']}")
    print(f"  GPU: {info.get('gpu', 'N/A')}")
    print(f"  VRAM: {info.get('vram', 'N/A')}")
    print(f"  Model: {model_path}")
    print()


def load_tts_model(model_path: str):
    """Load the Qwen3-TTS model on GPU."""
    global MODEL, MODEL_PATH
    from qwen_tts import Qwen3TTSModel

    logger.info(f"Loading model: {model_path} ...")

    load_kwargs = {
        "device_map": "cuda:0" if torch.cuda.is_available() else "cpu",
        "dtype": torch.bfloat16,
    }

    # Use flash_attention_2 if available, otherwise fall back to sdpa
    if torch.cuda.is_available():
        try:
            import flash_attn  # noqa: F401
            load_kwargs["attn_implementation"] = "flash_attention_2"
            logger.info("Using flash_attention_2")
        except ImportError:
            load_kwargs["attn_implementation"] = "sdpa"
            logger.info("flash-attn not installed, using sdpa (PyTorch native) attention")

    MODEL = Qwen3TTSModel.from_pretrained(model_path, **load_kwargs)
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


def run_generate_voice_clone(text: str, language: str, **kwargs):
    """
    Run model.generate_voice_clone() and return (audio_np, sample_rate).
    Handles the list output format: wavs is List[np.ndarray].
    """
    wavs, sr = MODEL.generate_voice_clone(
        text=text,
        language=language,
        **kwargs,
    )
    # wavs is a list (even for single input); take the first element
    audio_np = wavs[0] if isinstance(wavs, list) else wavs
    return audio_np.astype(np.float32), sr


# ---------------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------------
app = FastAPI(title="Qwen3-TTS Streaming Server (GPU)", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    """Require Bearer token or ?token= query param when SECRET is set."""
    if SECRET is None:
        return await call_next(request)

    # Allow GET / (UI page) without auth
    if request.method == "GET" and request.url.path == "/":
        return await call_next(request)

    # Check Authorization header
    auth = request.headers.get("authorization", "")
    if auth.startswith("Bearer ") and auth[7:] == SECRET:
        return await call_next(request)

    # Check ?token= query param
    if request.query_params.get("token") == SECRET:
        return await call_next(request)

    return JSONResponse(status_code=401, content={"detail": "Unauthorized"})


# ---------------------------------------------------------------------------
# GET / — Serve the Web UI
# ---------------------------------------------------------------------------
@app.get("/")
async def serve_ui():
    ui_path = Path(__file__).parent / "ui.html"
    return FileResponse(ui_path, media_type="text/html")


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
      - voice (str, optional): cached speaker name, default "default"
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

    voice = body.get("voice", "default")
    language = body.get("language", "Chinese")
    response_format = body.get("response_format", "wav")
    stream = body.get("stream", False)
    ref_audio = body.get("ref_audio", None)
    ref_text = body.get("ref_text", None)
    speed = body.get("speed", 1.0)

    # Build generation kwargs
    gen_kwargs = {}

    if voice in _ref_audio_cache and not ref_audio:
        # Use cached voice clone prompt
        cached = _ref_audio_cache[voice]
        gen_kwargs["voice_clone_prompt"] = cached["prompt"]
        logger.info(f"Using cached speaker '{voice}' via voice_clone_prompt")
    elif ref_audio is not None:
        # Pre-compute prompt from ref_audio (base64/URL) — use ICL mode for best clone quality
        prompt_items = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: MODEL.create_voice_clone_prompt(
                ref_audio=ref_audio,
                ref_text=ref_text if ref_text else None,
                x_vector_only_mode=not bool(ref_text),
            ),
        )
        gen_kwargs["voice_clone_prompt"] = prompt_items
    else:
        raise HTTPException(
            400,
            "Base model requires reference audio. Either provide 'ref_audio' + 'ref_text' in the request, "
            "or register a speaker first via POST /v1/speakers then use voice=<name>.",
        )

    # ----- Streaming PCM output -----
    if stream:
        async def generate_stream():
            try:
                audio_np, sr = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: run_generate_voice_clone(text, language, **gen_kwargs),
                )
                # Stream the audio in chunks
                chunk_size = 4096 * 4  # bytes (float32 samples)
                audio_bytes = audio_np.tobytes()
                for i in range(0, len(audio_bytes), chunk_size):
                    yield audio_bytes[i : i + chunk_size]
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
        audio_np, sr = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: run_generate_voice_clone(text, language, **gen_kwargs),
        )

        if response_format == "pcm":
            return Response(
                content=audio_np.tobytes(),
                media_type="application/octet-stream",
                headers={
                    "X-Sample-Rate": str(SAMPLE_RATE),
                    "X-Audio-Format": "pcm_f32le",
                },
            )

        # Default: WAV
        wav_bytes = numpy_to_wav_bytes(audio_np, sr)
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
      - voice (str): speaker name, default "default"
      - language (str): default "Chinese"
      - ref_audio (str, optional): reference audio URL/base64
      - ref_text (str, optional): reference text
    """
    voice = request.query_params.get("voice", "default")
    language = request.query_params.get("language", "Chinese")
    ref_audio = request.query_params.get("ref_audio", None)
    ref_text = request.query_params.get("ref_text", None)

    async def generate_from_streaming_text():
        buffer = ""
        sentence_enders = {"。", "！", "？", ".", "!", "?", "\n", "；", ";", "，", ",", "：", ":", "—"}
        received_any = False

        try:
            async for chunk in request.stream():
                text_data = chunk.decode("utf-8", errors="ignore")
                logger.info(f"stream-text received chunk ({len(chunk)} bytes): {text_data[:100]!r}")
                received_any = True

                for line in text_data.split("\n"):
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        obj = json.loads(line)
                        text_piece = obj.get("text", "")
                        done = obj.get("done", False)
                    except (json.JSONDecodeError, TypeError):
                        text_piece = line
                        done = False

                    buffer += text_piece

                    segments_to_synth = []
                    while buffer:
                        earliest_pos = -1
                        for ender in sentence_enders:
                            pos = buffer.find(ender)
                            if pos != -1 and (earliest_pos == -1 or pos < earliest_pos):
                                earliest_pos = pos

                        if earliest_pos != -1:
                            segment = buffer[: earliest_pos + 1].strip()
                            buffer = buffer[earliest_pos + 1 :]
                            if segment:
                                segments_to_synth.append(segment)
                        else:
                            break

                    for segment in segments_to_synth:
                        logger.info(f"Synthesizing segment: '{segment[:60]}'")
                        async for audio_bytes in _synthesize_segment(
                            segment, voice, language, ref_audio, ref_text
                        ):
                            yield audio_bytes

                    if len(buffer.strip()) > 30:
                        flush_text = buffer.strip()
                        buffer = ""
                        logger.info(f"Force-flushing long buffer: '{flush_text[:60]}'")
                        async for audio_bytes in _synthesize_segment(
                            flush_text, voice, language, ref_audio, ref_text
                        ):
                            yield audio_bytes

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
        gen_kwargs = {}

        if voice in _ref_audio_cache and not ref_audio:
            cached = _ref_audio_cache[voice]
            gen_kwargs["voice_clone_prompt"] = cached["prompt"]
            logger.info(f"Using cached speaker '{voice}' via voice_clone_prompt")
        elif ref_audio:
            # Pre-compute prompt from raw ref_audio — use ICL mode for best clone quality
            prompt_items = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: MODEL.create_voice_clone_prompt(
                    ref_audio=ref_audio,
                    ref_text=ref_text if ref_text else None,
                    x_vector_only_mode=not bool(ref_text),
                ),
            )
            gen_kwargs["voice_clone_prompt"] = prompt_items
        else:
            logger.error("No ref_audio or cached speaker available for segment synthesis")
            return

        audio_np, sr = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: run_generate_voice_clone(text, language, **gen_kwargs),
        )
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

    Pre-computes voice_clone_prompt (embedding + codes) once, then generates
    using the cached prompt — same path as registered speakers.

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

    # Pre-compute voice clone prompt — use ICL mode for best clone quality
    try:
        prompt_items = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: MODEL.create_voice_clone_prompt(
                ref_audio=ref_path,
                ref_text=ref_text,
            ),
        )
    except Exception as e:
        Path(ref_path).unlink(missing_ok=True)
        logger.error(f"Clone prompt creation error: {e}", exc_info=True)
        raise HTTPException(500, str(e))

    Path(ref_path).unlink(missing_ok=True)

    gen_kwargs = {
        "voice_clone_prompt": prompt_items,
    }

    if is_stream:
        async def stream_clone():
            try:
                audio_np, sr = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: run_generate_voice_clone(text, language, **gen_kwargs),
                )
                chunk_size = 4096 * 4
                audio_bytes = audio_np.tobytes()
                for i in range(0, len(audio_bytes), chunk_size):
                    yield audio_bytes[i : i + chunk_size]
            except Exception as e:
                logger.error(f"Clone streaming error: {e}", exc_info=True)

        return StreamingResponse(
            stream_clone(),
            media_type="application/octet-stream",
            headers={
                "X-Sample-Rate": str(SAMPLE_RATE),
                "X-Audio-Format": "pcm_f32le",
            },
        )

    try:
        audio_np, sr = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: run_generate_voice_clone(text, language, **gen_kwargs),
        )

        if response_format == "pcm":
            return Response(
                content=audio_np.tobytes(),
                media_type="application/octet-stream",
                headers={"X-Sample-Rate": str(SAMPLE_RATE)},
            )

        wav_bytes = numpy_to_wav_bytes(audio_np, sr)
        return Response(content=wav_bytes, media_type="audio/wav")

    except Exception as e:
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
    skip the feature extraction overhead.

    After registration, use voice=<name> (without ref_audio/ref_text) and
    the cached voice_clone_prompt will be used automatically.
    """
    ref_path = await save_upload_to_tempfile(ref_audio)
    try:
        # Build reusable voice clone prompt — use ICL mode for best clone quality
        prompt_items = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: MODEL.create_voice_clone_prompt(
                ref_audio=ref_path,
                ref_text=ref_text,
            ),
        )
        _ref_audio_cache[name] = {"prompt": prompt_items, "text": ref_text}
        logger.info(f"Registered speaker '{name}' via create_voice_clone_prompt")
        return {"ok": True, "name": name, "cached_speakers": list(_ref_audio_cache.keys())}
    except Exception as e:
        logger.error(f"Failed to register speaker '{name}': {e}", exc_info=True)
        raise HTTPException(500, str(e))
    finally:
        import os
        os.unlink(ref_path)


@app.get("/v1/speakers")
async def list_speakers():
    """List cached speakers."""
    cached = list(_ref_audio_cache.keys())
    return {"builtin": [], "cached": cached}


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
    parser = argparse.ArgumentParser(description="Qwen3-TTS Streaming Server (GPU)")
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        help="HuggingFace model ID or local path",
    )
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--secret",
        type=str,
        default=None,
        help="Bearer token for API authentication (optional, no auth if omitted)",
    )
    args = parser.parse_args()

    # Set global secret for auth middleware
    global SECRET
    SECRET = args.secret

    # Environment detection
    env_info = detect_environment()
    print_environment(env_info, args.model)

    load_tts_model(args.model)

    if SECRET:
        print(f"\n[Auth] Secret token enabled. Use:")
        print(f"  Authorization: Bearer {SECRET}")
        print(f"  or ?token={SECRET}")
        print()

    logger.info(f"Starting server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
