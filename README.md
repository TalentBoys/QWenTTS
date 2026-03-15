# Qwen3-TTS Server

基于 [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) 的 TTS 服务，通过 FastAPI 暴露 OpenAI 兼容的语音合成接口。

提供两个版本：
- **`server.py`** — macOS (Apple Silicon) 版，基于 MLX 框架，使用 `mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16` 模型
- **`server_gpu.py`** — GPU (CUDA) 版，基于 PyTorch + `qwen_tts`，使用 `Qwen/Qwen3-TTS-12Hz-1.7B-Base` 模型

两个版本的 API 接口完全一致，`test_client.py` 无需修改即可同时适配。

## 功能

- **语音合成** — `/v1/audio/speech`，支持流式/非流式音频输出
- **流式文本输入** — `/v1/audio/speech/stream-text`，适配 LLM 逐 token 输出场景
- **声音克隆** — `/v1/audio/speech/clone`，上传参考音频生成相似声音
- **Speaker 管理** — `/v1/speakers`，注册/查询可用发音人

## 快速开始

### macOS (Apple Silicon / MLX)

```bash
python -m venv .venv
source .venv/bin/activate
pip install mlx-audio fastapi uvicorn httpx soundfile numpy

python server.py --model mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16 --port 8000
```

### GPU / WSL2 (CUDA)

需要 NVIDIA GPU 和 CUDA 环境（推荐 RTX 3090 及以上，VRAM >= 16GB）。

```bash
conda create -n qwen3-tts python=3.12 -y
conda activate qwen3-tts
pip install -U qwen-tts
pip install -U flash-attn --no-build-isolation
pip install fastapi uvicorn httpx soundfile numpy

python server_gpu.py --port 8000
```

默认加载 `Qwen/Qwen3-TTS-12Hz-1.7B-Base`，也可通过 `--model` 指定其他模型：

```bash
python server_gpu.py --model Qwen/Qwen3-TTS-12Hz-0.6B-Base --port 8000
```

启动后会自动检测并打印运行环境：

```
[Environment] Running on: WSL2 (Linux on Windows) — CUDA GPU mode
  GPU: NVIDIA GeForce RTX 5090
  VRAM: 32GB
  Model: Qwen/Qwen3-TTS-12Hz-1.7B-Base
```

## 测试

```bash
# 基础 TTS
python test_client.py --mode basic --text "你好，欢迎使用语音合成服务。"

# 流式音频输出
python test_client.py --mode stream --text "今天天气真好。"

# 流式文本输入（模拟 LLM 输出）
python test_client.py --mode stream-text --text "这是一个流式文字输入的测试。"

# 声音克隆
python test_client.py --mode clone --text "克隆测试" --ref-audio ref.wav --ref-text "参考文本"

# 指定远程服务地址
python test_client.py --url http://192.168.1.100:8000 --mode basic --text "远程调用测试"
```

## API

| 接口 | 方法 | 说明 |
|------|------|------|
| `/v1/audio/speech` | POST | 文本转语音，支持流式/非流式 |
| `/v1/audio/speech/stream-text` | POST | 流式文本输入（NDJSON），实时返回音频 |
| `/v1/audio/speech/clone` | POST | 上传参考音频 + 文本进行声音克隆 |
| `/v1/speakers` | GET/POST | 查询/注册发音人 |
| `/v1/models` | GET | 查询当前加载的模型 |
| `/health` | GET | 健康检查 |

接口路径参考了 OpenAI Audio API 的风格（`/v1/audio/speech`），因此也可以用 OpenAI SDK 作为客户端调用，但需要通过 `extra_body` 传递 `language` 等非标准参数。

音频输出格式：PCM float32 LE / WAV PCM16，采样率 24000 Hz。
