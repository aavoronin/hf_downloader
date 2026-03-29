from typing import List

from download.MultipleModelsDownloader import MultipleModelsDownloader


def execute_download():
    # Initialize downloader - token handled in __init__
    # downloader = HFModelDownloader()

    # success = downloader.download(
    #    model_id="facebook/mms-tts-rus",
    #    target_dir="/mnt/d/AIs/Models/mms-tts-rus"
    # )

    """start_urls = [
        #"https://huggingface.co/models?pipeline_tag=automatic-speech-recognition&sort=likes&search=rus",
        "https://huggingface.co/models?pipeline_tag=image-to-text&library=pytorch&sort=trending&search=ocr",
        "https://huggingface.co/models?pipeline_tag=image-to-text&library=pytorch&sort=trending&search=eng"
    ]
    for start_url in start_urls:
        lister = HFModelLister(start_url)
        lister.fetch_all_pages()
        lister.show_results()
    """

    start_urls = [
        # "https://huggingface.co/models?pipeline_tag=image-to-text&library=pytorch&sort=trending&search=ocr",
        # "https://huggingface.co/models?pipeline_tag=image-to-text&library=pytorch&sort=trending&search=eng"
        "https://huggingface.co/models?pipeline_tag=image-to-text&sort=likes&search=ru",
        "https://huggingface.co/models?pipeline_tag=image-to-text&library=pytorch&sort=trending"
    ]

    root_folder = r"D:\AIs\Image-to-Text"
    download_certain_type_of_models(root_folder, start_urls)

    start_urls = ["https://huggingface.co/models?pipeline_tag=automatic-speech-recognition&sort=likes&search=ru",
                  "https://huggingface.co/models?pipeline_tag=automatic-speech-recognition&sort=likes"]

    root_folder = r"D:\AIs\Automatic Speech Recognition"
    download_certain_type_of_models(root_folder, start_urls)

    start_urls = ["https://huggingface.co/models?pipeline_tag=text-to-speech&sort=trending&search=rus",
                  "https://huggingface.co/models?pipeline_tag=text-to-speech&sort=trending"]
    root_folder = r"D:\AIs\Text-to-Speech"
    download_certain_type_of_models(root_folder, start_urls)

    start_urls = ["https://huggingface.co/models?pipeline_tag=text-to-audio&sort=trending&search=rus",
                  "https://huggingface.co/models?pipeline_tag=text-to-audio&sort=trending"]
    root_folder = r"D:\AIs\Text-to-Audio"
    download_certain_type_of_models(root_folder, start_urls)

    start_urls = ["https://huggingface.co/models?pipeline_tag=image-text-to-image&sort=likes"]
    exclude = ["Arunk25/Qwen-Image-Edit-Rapid-AIO-GGUF",
               "liuxin9494/Qwen-Image-Edit-Rapid-AIO-GGUF-mirror"]
    exclude = []
    root_folder = r"D:\AIs\Image-Text-to-Image"

    downloader = MultipleModelsDownloader(start_urls=start_urls, root_folder=root_folder,
                                          exclude = exclude)
    download_certain_type_of_models(root_folder, start_urls, exclude)


def download_certain_type_of_models(root_folder: str, start_urls: list[str], exclude: List[str] = []):
    downloader = MultipleModelsDownloader(start_urls=start_urls, root_folder=root_folder,
                                          exclude = exclude)
    downloader.process_urls()
    downloader.show_results()
    downloader.print_local_models()
    downloader.download_models()
    downloader.print_download_summary()
    downloader.print_folder_structure()

