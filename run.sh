#!/bin/bash
# Wrapper script for profanity_filter.py with CUDA/cuDNN support

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/venv"

# Set library paths for CUDA/cuDNN
export LD_LIBRARY_PATH="$VENV_DIR/lib/python3.12/site-packages/nvidia/cudnn/lib:$VENV_DIR/lib/python3.12/site-packages/nvidia/cublas/lib:$LD_LIBRARY_PATH"

# Run the profanity filter
exec "$VENV_DIR/bin/python" "$SCRIPT_DIR/profanity_filter.py" "$@"
