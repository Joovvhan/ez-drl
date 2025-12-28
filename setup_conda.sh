#!/usr/bin/env bash
# macOS / Linux helper to create the conda env and show PyTorch install hints
set -euo pipefail

ENV_NAME=ez-drl

echo "Creating conda environment '${ENV_NAME}' from environment.yml..."
conda env create -f environment.yml -n ${ENV_NAME}

if [ $? -ne 0 ]; then
    echo "Error: Failed to create conda environment"
    exit 1
fi

echo ""
echo "======================================================================"
echo "Conda environment created successfully!"
echo "======================================================================"
echo ""
echo "Activate it with: conda activate ${ENV_NAME}"
echo ""
echo "======================================================================"
echo "PyTorch installation (choose one based on your platform / GPU):"
echo "======================================================================"
echo ""
echo "1) macOS with Apple Silicon (MPS support):"
echo "   pip install torch torchvision torchaudio"
echo ""
echo "2) macOS CPU only:"
echo "   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu"
echo ""
echo "3) Linux with CUDA 11.8 (recommended for NVIDIA GPUs):"
echo "   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
echo ""
echo "4) Linux with CUDA 12.1:"
echo "   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121"
echo ""
echo "5) Linux CPU only:"
echo "   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu"
echo ""
echo "If unsure, open the PyTorch local install selector:"
echo "https://pytorch.org/get-started/locally/"
echo ""
echo "======================================================================"
echo "After installing PyTorch, test the setup with:"
echo "   python train.py --env CartPole-v1 --algo ppo --timesteps 10000"
echo "======================================================================"
