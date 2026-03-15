"""
Test client for Qwen3-TTS Streaming Server.

Examples:
  # 1. Non-streaming TTS (saves WAV)
  python test_client.py --mode basic --text "你好，欢迎使用语音合成服务。"

  # 2. Streaming audio output (saves PCM then converts to WAV)
  python test_client.py --mode stream --text "今天天气真好，我们去公园散步吧。"

  # 3. Streaming text input (simulates LLM token-by-token output)
  python test_client.py --mode stream-text --text "这是一个流式文字输入的测试。模型会逐字接收文本，然后实时生成语音。"

  # 4. Voice clone
  python test_client.py --mode clone --text "克隆声音测试" --ref-audio reference.wav --ref-text "参考音频的文字内容"
"""

import argparse
import json
import time
import struct

import httpx
import numpy as np
import soundfile as sf


BASE_URL = "http://localhost:8000"
SAMPLE_RATE = 24000


def test_basic(text: str, voice: str, language: str):
    """Non-streaming TTS, returns WAV file."""
    print(f"[Basic TTS] Text: {text}")
    t0 = time.time()

    resp = httpx.post(
        f"{BASE_URL}/v1/audio/speech",
        json={
            "input": text,
            "voice": voice,
            "language": language,
            "response_format": "wav",
            "stream": False,
        },
        timeout=120.0,
    )
    resp.raise_for_status()

    output_path = "output_basic.wav"
    with open(output_path, "wb") as f:
        f.write(resp.content)

    elapsed = time.time() - t0
    print(f"  Saved to {output_path} ({len(resp.content)} bytes, {elapsed:.2f}s)")


def test_stream(text: str, voice: str, language: str):
    """Streaming audio output — receives PCM float32 chunks progressively."""
    print(f"[Streaming Audio] Text: {text}")
    t0 = time.time()
    first_chunk_time = None
    all_audio = bytearray()

    with httpx.stream(
        "POST",
        f"{BASE_URL}/v1/audio/speech",
        json={
            "input": text,
            "voice": voice,
            "language": language,
            "response_format": "pcm",
            "stream": True,
        },
        timeout=120.0,
    ) as resp:
        resp.raise_for_status()
        for chunk in resp.iter_bytes(chunk_size=4096):
            if first_chunk_time is None:
                first_chunk_time = time.time()
                ttfa = first_chunk_time - t0
                print(f"  Time to first audio: {ttfa:.3f}s")
            all_audio.extend(chunk)

    # Convert PCM float32 to WAV
    audio_np = np.frombuffer(bytes(all_audio), dtype=np.float32)
    output_path = "output_stream.wav"
    sf.write(output_path, audio_np, SAMPLE_RATE)

    elapsed = time.time() - t0
    duration = len(audio_np) / SAMPLE_RATE
    print(f"  Saved to {output_path} (audio duration: {duration:.2f}s, total time: {elapsed:.2f}s)")


def test_stream_text(text: str, voice: str, language: str):
    """
    Streaming text input — simulates LLM streaming output.
    Sends text character by character (or in small chunks) and receives audio back.
    """
    print(f"[Streaming Text Input] Full text: {text}")
    t0 = time.time()
    first_chunk_time = None
    all_audio = bytearray()

    # Simulate streaming text: send characters in small batches
    # In real usage, this would come from an LLM's streaming output
    chunk_size = 3  # characters per chunk
    text_chunks = [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

    def generate_request_body():
        for i, chunk in enumerate(text_chunks):
            is_last = i == len(text_chunks) - 1
            line = json.dumps({"text": chunk, "done": is_last}, ensure_ascii=False) + "\n"
            print(f"  Sending: {chunk!r} (done={is_last})")
            yield line.encode("utf-8")
            time.sleep(0.05)  # Simulate LLM token generation delay

    with httpx.stream(
        "POST",
        f"{BASE_URL}/v1/audio/speech/stream-text",
        params={"voice": voice, "language": language},
        content=generate_request_body(),
        headers={"Content-Type": "application/x-ndjson"},
        timeout=120.0,
    ) as resp:
        resp.raise_for_status()
        for chunk in resp.iter_bytes(chunk_size=4096):
            if first_chunk_time is None:
                first_chunk_time = time.time()
                ttfa = first_chunk_time - t0
                print(f"  Time to first audio: {ttfa:.3f}s")
            all_audio.extend(chunk)

    audio_np = np.frombuffer(bytes(all_audio), dtype=np.float32)
    output_path = "output_stream_text.wav"
    sf.write(output_path, audio_np, SAMPLE_RATE)

    elapsed = time.time() - t0
    duration = len(audio_np) / SAMPLE_RATE
    print(f"  Saved to {output_path} (audio duration: {duration:.2f}s, total time: {elapsed:.2f}s)")


def test_clone(text: str, language: str, ref_audio: str, ref_text: str):
    """Voice clone endpoint with file upload."""
    print(f"[Voice Clone] Text: {text}")
    print(f"  Ref audio: {ref_audio}")
    t0 = time.time()

    with open(ref_audio, "rb") as f:
        resp = httpx.post(
            f"{BASE_URL}/v1/audio/speech/clone",
            data={
                "text": text,
                "language": language,
                "ref_text": ref_text,
                "response_format": "wav",
                "stream": "false",
            },
            files={"ref_audio": (ref_audio, f, "audio/wav")},
            timeout=120.0,
        )
    resp.raise_for_status()

    output_path = "output_clone.wav"
    with open(output_path, "wb") as f:
        f.write(resp.content)

    elapsed = time.time() - t0
    print(f"  Saved to {output_path} ({len(resp.content)} bytes, {elapsed:.2f}s)")


def test_openai_compatible(text: str, voice: str, language: str):
    """
    Test using the standard OpenAI Python client.
    Demonstrates drop-in compatibility.
    """
    print(f"[OpenAI Client] Text: {text}")
    try:
        from openai import OpenAI

        client = OpenAI(base_url=f"{BASE_URL}/v1", api_key="not-needed")

        # Non-streaming
        response = client.audio.speech.create(
            model="tts-1",
            voice=voice,
            input=text,
            extra_body={"language": language},
        )
        response.stream_to_file("output_openai.wav")
        print("  Saved to output_openai.wav")

    except ImportError:
        print("  openai package not installed. Run: pip install openai")
    except Exception as e:
        print(f"  Error: {e}")


def main():
    parser = argparse.ArgumentParser(description="Qwen3-TTS Test Client")
    parser.add_argument(
        "--mode",
        choices=["basic", "stream", "stream-text", "clone", "openai"],
        default="basic",
        help="Test mode",
    )
    parser.add_argument("--text", type=str, default="你好，欢迎使用通义千问语音合成服务。")
    parser.add_argument("--voice", type=str, default="Chelsie")
    parser.add_argument("--language", type=str, default="Chinese")
    parser.add_argument("--ref-audio", type=str, default=None)
    parser.add_argument("--ref-text", type=str, default=None)
    parser.add_argument("--url", type=str, default="http://localhost:8000")
    args = parser.parse_args()

    global BASE_URL
    BASE_URL = args.url

    if args.mode == "basic":
        test_basic(args.text, args.voice, args.language)
    elif args.mode == "stream":
        test_stream(args.text, args.voice, args.language)
    elif args.mode == "stream-text":
        test_stream_text(args.text, args.voice, args.language)
    elif args.mode == "clone":
        if not args.ref_audio or not args.ref_text:
            print("Error: --ref-audio and --ref-text are required for clone mode")
            return
        test_clone(args.text, args.language, args.ref_audio, args.ref_text)
    elif args.mode == "openai":
        test_openai_compatible(args.text, args.voice, args.language)


if __name__ == "__main__":
    main()
