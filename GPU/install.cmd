@echo off
setlocal EnableDelayedExpansion

if not exist .venv (
    python -m venv .venv
)
call .venv\Scripts\activate.bat

python -m pip install --upgrade pip >nul 2>&1
pip uninstall torch torchvision torchaudio -y >nul 2>&1

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

pip install ^
    transformers==4.46.3 ^
    accelerate==1.13.0 ^
    diffusers==0.31.0 ^
    librosa ^
    moviepy ^
    safetensors==0.7.0 ^
    huggingface-hub==0.26.5 ^
    tokenizers==0.20.3 ^
    numpy==2.2.6

pip install xformers --index-url https://download.pytorch.org/whl/cu128 2>nul

python -c "
import torch, sys
print(f'Python: {sys.version.split()[0]}')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'Archs: {torch.cuda.get_arch_list()}')
    x = torch.randn(100, 100, device='cuda', dtype=torch.float16)
    y = torch.mm(x, x.t())
    print(f'sm_120 matmul test: PASS')
else:
    print('CUDA not available - check drivers')
"

endlocal