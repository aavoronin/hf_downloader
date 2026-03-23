import os
import re
import time
from huggingface_hub import snapshot_download, login
from pathlib import Path

from my_token import HF_TOKEN




# =============================================================================
# CONFIGURATION
# =============================================================================
MODELS_BASE_DIR = "/mnt/d/AIs/StableDiffusion/models_2"
SD15_DIR = os.path.join(MODELS_BASE_DIR, "sd15")
SDXL_DIR = os.path.join(MODELS_BASE_DIR, "sdxl")

print("=" * 70)
print("STABLE DIFFUSION MODEL DOWNLOADER (EXPANDED)")
print("=" * 70)
print(f"Base Models Directory: {MODELS_BASE_DIR}")
print("=" * 70)


# =============================================================================
# PATH NORMALIZATION (WSL/Windows compatibility)
# =============================================================================
def normalize_model_path(path: str) -> str:
    r"""
    Converts WSL/Linux style path to Windows format if running on Windows.
    """
    if os.name == 'nt':
        wsl_pattern = r'^/mnt/([a-zA-Z])(.*)'
        match = re.match(wsl_pattern, path)
        if match:
            drive_letter = match.group(1).upper()
            path_rest = match.group(2).replace('/', '\\')
            return f"{drive_letter}:{path_rest}"
        return path.replace('/', '\\')
    return path


SD15_DIR = normalize_model_path(SD15_DIR)
SDXL_DIR = normalize_model_path(SDXL_DIR)

# Create all directories upfront
for dir_path in [SD15_DIR, os.path.join(SDXL_DIR, "base"), os.path.join(SDXL_DIR, "refiner")]:
    Path(dir_path).mkdir(parents=True, exist_ok=True)

# =============================================================================
# AUTHENTICATION
# =============================================================================
print("\n🔐 HuggingFace Authentication")
print("-" * 70)

if HF_TOKEN and HF_TOKEN != "hf_YOUR_NEW_TOKEN_HERE":
    try:
        login(token=HF_TOKEN, add_to_git_credential=False)
        print("✓ Successfully authenticated with HuggingFace")
    except Exception as e:
        print(f"⚠ Authentication warning: {e}")
else:
    print("⚠️  WARNING: No valid token configured!")
    print("   Please ensure my_token.py contains a valid HF_TOKEN")


# =============================================================================
# GENERIC MODEL DOWNLOAD HELPER
# =============================================================================
def _download_single_model(repo_id: str, local_dir: str, label: str,
                           ignore_patterns: list = None, create_dir: bool = True,
                           force_redownload: bool = False):
    """
    Download a single model from HuggingFace.

    Args:
        repo_id: HuggingFace repository ID
        local_dir: Local directory to download to
        label: Display label for progress messages
        ignore_patterns: File patterns to ignore
        create_dir: Whether to create target directory first
        force_redownload: If True, re-download even if files exist

    Returns:
        bool: True if download succeeded
    """
    if ignore_patterns is None:
        ignore_patterns = ["*.msgpack", "*.ot"]

    # Check if model already exists
    config_path = Path(local_dir) / "model_index.json"
    if config_path.exists() and not force_redownload:
        size_gb = sum(f.stat().st_size for f in Path(local_dir).rglob('*') if f.is_file()) / 1024 ** 3
        print(f"✓ {label} already exists ({size_gb:.2f} GB) - skipping")
        return True

    try:
        if create_dir:
            Path(local_dir).mkdir(parents=True, exist_ok=True)

        print(f"  Downloading {label} to: {local_dir}")

        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            ignore_patterns=ignore_patterns,
            force_download=force_redownload
        )

        size_gb = sum(f.stat().st_size for f in Path(local_dir).rglob('*') if f.is_file()) / 1024 ** 3
        print(f"✓ {label} downloaded successfully ({size_gb:.2f} GB)")
        return True
    except Exception as e:
        print(f"✗ {label} download failed: {e}")
        return False


# =============================================================================
# MODEL DOWNLOAD FUNCTIONS
# =============================================================================
def download_sd15(force=False):
    """Download Stable Diffusion 1.5 (TEST MODE)"""
    print("\n[1/3] Downloading Stable Diffusion 1.5 (TEST MODE)...")
    print("-" * 70)
    print(f"  Target: {SD15_DIR}")
    print(f"  Repo: runwayml/stable-diffusion-v1-5")
    print(f"  Expected Size: ~4 GB")

    return _download_single_model(
        repo_id="runwayml/stable-diffusion-v1-5",
        local_dir=SD15_DIR,
        label="SD 1.5",
        force_redownload=force
    )


def download_sdxl_base(force=False):
    """Download SDXL Base model"""
    print("\n[2/3] Downloading SDXL Base (PROD MODE)...")
    print("-" * 70)
    base_path = os.path.join(SDXL_DIR, "base")
    print(f"  Target: {base_path}")
    print(f"  Repo: stabilityai/stable-diffusion-xl-base-1.0")
    print(f"  Expected Size: ~7 GB")

    return _download_single_model(
        repo_id="stabilityai/stable-diffusion-xl-base-1.0",
        local_dir=base_path,
        label="SDXL Base",
        force_redownload=force
    )


def download_sdxl_refiner(force=False):
    """Download SDXL Refiner model"""
    print("\n[3/3] Downloading SDXL Refiner (PROD MODE)...")
    print("-" * 70)
    refiner_path = os.path.join(SDXL_DIR, "refiner")
    print(f"  Target: {refiner_path}")
    print(f"  Repo: stabilityai/stable-diffusion-xl-refiner-1.0")
    print(f"  Expected Size: ~7 GB")

    return _download_single_model(
        repo_id="stabilityai/stable-diffusion-xl-refiner-1.0",
        local_dir=refiner_path,
        label="SDXL Refiner",
        force_redownload=force
    )


def download_stable_fusion_models(force=False):
    """Download all models (SD 1.5 + SDXL Base + SDXL Refiner)"""
    print("\n" + "=" * 70)
    print("DOWNLOADING ALL MODELS")
    print("=" * 70)

    start_time = time.time()

    results = {
        "SD 1.5": download_sd15(force),
        "SDXL Base": download_sdxl_base(force),
        "SDXL Refiner": download_sdxl_refiner(force)
    }

    elapsed = time.time() - start_time

    print("\n" + "=" * 70)
    print("DOWNLOAD SUMMARY")
    print("=" * 70)
    for name, success in results.items():
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"{name:20s}: {status}")
    print(f"Total Time: {elapsed / 60:.1f} minutes")
    print("=" * 70)

    all_success = all(results.values())
    if all_success:
        print("\n🎉 All models downloaded successfully!")
        print(f"📁 Models location: {MODELS_BASE_DIR}")
    else:
        print("\n⚠️  Some downloads failed. Check errors above.")
        print("💡 Tip: Run with force=True to retry failed downloads")

    return all_success

