param(
    [string]$EnvName = "ez-drl"
)

Write-Host "Creating conda environment '$EnvName' from environment.yml..."
conda env create -f environment.yml -n $EnvName
Write-Host "Activate it with: conda activate $EnvName"

Write-Host "PyTorch installation notes:`n"
Write-Host "- For Windows with CUDA, pick the CUDA build suggested at https://pytorch.org/get-started/locally"
Write-Host "- Example (CUDA 11.7):`
Write-Host "    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117"
Write-Host "- CPU fallback:`n    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu"

Write-Host "After installing PyTorch, run: python examples\train_reinforce_cartpole.py"
