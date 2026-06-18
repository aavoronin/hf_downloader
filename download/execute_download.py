from typing import List

from download.HFModelLister import HFModelLister
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

    """
    start_urls = [
        "https://huggingface.co/models?sort=likes"

    ]
    root_folder = r"D:\AIs\Info"
    HFModelLister.MAX_PAGES = 20
    for start_url in start_urls:
        for _ in range(1):
            download_models_info(root_folder,
                start_urls=[start_url], first_only = False)
    return
    """

    start_urls = [
        #"https://huggingface.co/models?pipeline_tag=any-to-any&sort=downloads",
        #"https://huggingface.co/models?pipeline_tag=any-to-any&num_parameters=min:6B,max:9B&sort=downloads",
        #"https://huggingface.co/models?pipeline_tag=any-to-any&num_parameters=min:6B,max:9B&sort=likes",
        "https://huggingface.co/models?sort=likes&search=Qwen%2FQwen2.5-Coder-32B-Instruct",
        #"https://huggingface.co/models?sort=likes&search=Qwen%2FQwen2.5-Coder-7B-Instruct",
        #"https://huggingface.co/models?sort=likes&search=Qwen%2FQwen2.5-Coder-14B-Instruct",
        #"https://huggingface.co/models?sort=likes&search=Qwen%2FQwen2.5-Coder-1.5B-Instruct",
        #"https://huggingface.co/models?sort=likes&search=Qwen%2FQwen2.5-Coder-3B-Instruct",
        #"https://huggingface.co/models?sort=likes&search=Qwen%2FQwen2.5-Coder-0.5B-Instruct",
        #"https://huggingface.co/models?sort=likes&search=Bhuvneesh%2Fgemma-4-E4B-it-Q8_0-GGUF",
    ]
    root_folder = r"D:\AIs\Any-to-Any"
    HFModelLister.MAX_PAGES = 1
    for start_url in start_urls:
        for _ in range(5):
            download_certain_type_of_models(root_folder,
                start_urls=[start_url], first_only = True)
    return

    start_urls = [
        "https://huggingface.co/models?pipeline_tag=image-text-to-text&num_parameters=min:6B,max:24B&sort=trending"
    ]
    root_folder = r"D:\AIs\Image-Text-to-Text"
    HFModelLister.MAX_PAGES = 1
    for _ in range(5):
        download_certain_type_of_models(root_folder, start_urls)
    return



    start_urls = [
        #"https://huggingface.co/models?pipeline_tag=text-generation&num_parameters=min:0,max:6B&sort=likes&search=sql",
        #"https://huggingface.co/models?pipeline_tag=text-generation&num_parameters=min:0,max:6B&library=transformers&sort=likes&search=pip-sql",
        #"https://huggingface.co/models?pipeline_tag=text-generation&num_parameters=min:0,max:6B&library=transformers&sort=likes&search=text-to-SQL",
        #"https://huggingface.co/models?pipeline_tag=text-generation&num_parameters=min:0,max:6B&library=pytorch&sort=likes&search=sql"
        #"https://huggingface.co/models?pipeline_tag=text-generation&num_parameters=min:0,max:6B&library=gguf&sort=likes&search=sql-GGUF",
        #"https://huggingface.co/models?pipeline_tag=text-generation&num_parameters=min:0,max:6B&sort=likes&search=code",
        "https://huggingface.co/models?pipeline_tag=text-generation&num_parameters=min:0,max:6B&sort=likes&search=SQL"
    ]
    root_folder = r"D:\AIs\text-to-sql"
    HFModelLister.MAX_PAGES = 10
    download_certain_type_of_models(root_folder, start_urls)
    return


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

def download_models_info(root_folder: str,
                         start_urls: list[str],
                         exclude: List[str] = [],
                         first_only=False):
    downloader = MultipleModelsDownloader(start_urls=start_urls, root_folder=root_folder,
                                          exclude=exclude)
    downloader.process_urls()
    downloader.download_model_pages()


def download_certain_type_of_models(root_folder: str,
                                    start_urls: list[str],
                                    exclude: List[str] = [],
                                    first_only=False):
    downloader = MultipleModelsDownloader(start_urls=start_urls, root_folder=root_folder,
                                          exclude = exclude)
    downloader.process_urls()
    downloader.show_results()
    downloader.print_local_models()
    downloader.download_models(first_only)
    downloader.print_download_summary()
    downloader.print_folder_structure()

