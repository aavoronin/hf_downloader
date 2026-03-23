# =============================================================================
# EXECUTION
# =============================================================================
# Download all models (set force=True to re-download existing models)
from download1 import download_stable_fusion_models

##download_stable_fusion_models(force=False)

# example_usage.py
from hf_downloader import HFModelDownloader
from my_token import HF_TOKEN


def main():
    # Initialize downloader - token handled in __init__
    downloader = HFModelDownloader(token=HF_TOKEN)

    success = downloader.download(
        model_id="facebook/mms-tts-rus",
        target_dir="/mnt/d/AIs/Models/mms-tts-rus"
    )

    if success:
        print(f"\n✅ Model ready at: /mnt/d/AIs/Models/mms-tts-rus")
    else:
        print("\n❌ Download failed - check logs above")


if __name__ == "__main__":
    main()
