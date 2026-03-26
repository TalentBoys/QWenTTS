#!/bin/bash
set -e

cd "$(dirname "$0")"

# Create venv if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate venv
source venv/bin/activate

# Install dependencies if needed
if ! python -c "import torch" 2>/dev/null; then
    echo "Installing dependencies..."
    pip install torch numpy soundfile uvicorn fastapi python-multipart qwen-tts
fi

echo "Starting Qwen3-TTS GPU Server..."
python server_gpu.py "$@"
