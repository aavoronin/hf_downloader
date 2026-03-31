from ASR.ASRManager import ASR_set_device
from ASR.ASR_main import ASR_main
from GPU.Test_GPU import test_GPU
from OCR.ImageToTextManager import set_device, ImageToTextManager
from OCR.prepare_ocr_test import parse_alice
from download.download_aidio_dataset_russian import download_librispeech_english, export_librispeech_samples
from download.execute_download import execute_download
from download.export_audio_samples import export_audio, do_export_10_samples

if __name__ == "__main__":
    #execute_download()
    # parse_alice()

    #test_GPU()
    #ASR_set_device(use_gpu=True)
    ASR_main()
    #download_audio_dataset_russian()
    #do_export_10_samples()



    r"""
    # Шаг 1: Загрузить (streaming - без полной загрузки)
    dataset = download_librispeech_english(
        dest_dir=r"D:\Data\audio\librispeech_en",
        config="clean",
        split="train.100"  # 100 часов, чистая речь
    )

    # Шаг 2: Экспортировать 10 коротких фраз
    files = export_librispeech_samples(
        output_dir=r"D:\Data\audio_test\librispeech",
        cache_dir=r"D:\Data\audio\librispeech_en",
        num_samples=10,
        max_phrase_words=10,  # ≤10 слов
        max_duration_sec=15.0  # ≤15 секунд
    )

    
    manager = ImageToTextManager(r"D:\AIs\Image-to-Text")
    models = manager.get_available_models()
    for model in models:
        print(f"{model.name}: {model.size_mb:.2f} MB")
    test_results = manager.run_test(
        test_image = r"D:\Data\ocr_test\Alice\Alice_test_image0.png",
        reference_text_file = r"D:\Data\ocr_test\Alice\alice0.txt"
    )
    """