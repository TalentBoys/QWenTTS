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
# 系统依赖
sudo apt update && sudo apt upgrade -y
sudo apt install python3-venv sox libsox-fmt-all -y

# Python 虚拟环境
python3 -m venv .venv
source .venv/bin/activate

# 安装依赖
pip install --upgrade pip setuptools wheel
pip install numpy typing_extensions sox
pip install -U qwen-tts -v

# 可选：安装 flash-attn 加速（编译较慢，不装也能跑，会自动回退到 sdpa）
# pip install -U flash-attn --no-build-isolation
```

启动服务：

```bash
python server_gpu.py --port 8000
```

启用 Token 认证（推荐公网部署时使用）：

```bash
python server_gpu.py --port 8000 --secret YOUR_SECRET_TOKEN
```

启动后会打印 Token 使用方式：

```
[Auth] Secret token enabled. Use:
  Authorization: Bearer YOUR_SECRET_TOKEN
  or ?token=YOUR_SECRET_TOKEN
```

不传 `--secret` 则不启用认证，适合本地开发。

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

## 公网部署（WSL2）

如果服务运行在 WSL2 中，需要额外配置让外部访问：

### 1. Windows 端口转发

WSL2 的端口只映射到 Windows 的 `localhost`，局域网其他设备无法直接访问。用 `netsh` 转发（管理员 PowerShell）：

```powershell
netsh interface portproxy add v4tov4 listenport=8000 listenaddress=0.0.0.0 connectport=8000 connectaddress=127.0.0.1
```

> 注意：如果 WSL 重启后 `localhost` 转发失效，需要改 `connectaddress` 为 WSL 的实际 IP（在 WSL 中 `hostname -I` 查看）。

### 2. Windows 防火墙放行

```powershell
New-NetFirewallRule -DisplayName "TTS 8000" -Direction Inbound -LocalPort 8000 -Protocol TCP -Action Allow -Profile Any
```

### 3. HTTPS（推荐）

浏览器在非 HTTPS 环境下会禁止麦克风访问（`getUserMedia`），因此公网使用 Web UI 录音功能必须部署 HTTPS。

推荐使用 Nginx Proxy Manager 或 Caddy 做反向代理，将域名指向 `http://<Windows局域网IP>:8000`，并启用 Let's Encrypt 自动证书。

### 4. Token 认证

公网部署务必启用 `--secret` 参数保护 API：

```bash
python server_gpu.py --port 8000 --secret YOUR_SECRET_TOKEN
```

Web UI 顶部提供 Token 输入框，填入后所有 API 请求自动携带认证头，Token 会保存在浏览器 `localStorage` 中。
