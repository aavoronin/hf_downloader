# =============================================================================
# EXECUTION
# =============================================================================
# Download all models (set force=True to re-download existing models)
from HFModelLister import HFModelLister
from download1 import download_stable_fusion_models

##download_stable_fusion_models(force=False)

# example_usage.py
from hf_downloader import HFModelDownloader


def main():
    # Initialize downloader - token handled in __init__
    #downloader = HFModelDownloader()

    #success = downloader.download(
    #    model_id="facebook/mms-tts-rus",
    #    target_dir="/mnt/d/AIs/Models/mms-tts-rus"
    #)

    start_url = "https://huggingface.co/models?pipeline_tag=automatic-speech-recognition&sort=likes&search=rus"
    lister = HFModelLister(start_url)
    lister.fetch_all_pages()
    lister.show_results()


if __name__ == "__main__":
    main()
