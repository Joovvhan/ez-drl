#!/usr/bin/env bash
# macOS / Linux helper to create the conda env and show PyTorch install hints
set -euo pipefail

ENV_NAME=ez-drl

echo "Creating conda environment '${ENV_NAME}' from environment.yml..."
conda env create -f environment.yml -n ${ENV_NAME}
echo "Activate it with: conda activate ${ENV_NAME}"

cat <<'INSTR'
PyTorch installation (choose one based on your platform / GPU):

# 1) macOS (MPS or CPU):
#    - For Apple Silicon MPS support, use the official selector at https://pytorch.org/get-started/locally
#    - A common CPU fallback:
#      pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 2) Windows / Linux with CUDA (example for CUDA 11.7):
#    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117

# If unsure, open the PyTorch local install selector:
# https://pytorch.org/get-started/locally/
INSTR

echo "Done. After installing PyTorch, run: python examples/train_reinforce_cartpole.py"
