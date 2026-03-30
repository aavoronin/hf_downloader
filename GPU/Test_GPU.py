import torch

def test_GPU():
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Supported archs: {torch.cuda.get_arch_list()}")
    print(f"GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")

    # Test sm_120 kernel execution
    if torch.cuda.is_available():
        x = torch.randn(100, 100, device='cuda', dtype=torch.float16)
        y = torch.mm(x, x.t())
        print(f"✅ sm_120 matmul test passed: {y.shape}")