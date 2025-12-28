@echo off
REM Windows batch script to create the conda environment and show PyTorch install hints
setlocal

set ENV_NAME=ez-drl

echo Creating conda environment '%ENV_NAME%' from environment.yml...
conda env create -f environment.yml -n %ENV_NAME%

if %ERRORLEVEL% neq 0 (
    echo Error: Failed to create conda environment
    exit /b 1
)

echo.
echo ======================================================================
echo Conda environment created successfully!
echo ======================================================================
echo.
echo Activate it with: conda activate %ENV_NAME%
echo.
echo ======================================================================
echo PyTorch installation (choose one based on your platform / GPU):
echo ======================================================================
echo.
echo 1) Windows with CUDA 11.8 (recommended for NVIDIA GPUs):
echo    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
echo.
echo 2) Windows with CUDA 12.1:
echo    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
echo.
echo 3) Windows CPU only:
echo    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
echo.
echo If unsure, open the PyTorch local install selector:
echo https://pytorch.org/get-started/locally/
echo.
echo ======================================================================
echo After installing PyTorch, test the setup with:
echo    python train.py --env CartPole-v1 --algo ppo --timesteps 10000
echo ======================================================================

endlocal
