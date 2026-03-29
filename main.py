from ASR.ASR_main import ASR_main
from OCR.ImageToTextManager import set_device, ImageToTextManager
from OCR.prepare_ocr_test import parse_alice
from download.execute_download import execute_download

if __name__ == "__main__":
    #execute_download()
    # parse_alice()

    ASR_main()

    """
    set_device(use_gpu=False)
    manager = ImageToTextManager(r"D:\AIs\Image-to-Text")
    models = manager.get_available_models()
    for model in models:
        print(f"{model.name}: {model.size_mb:.2f} MB")
    test_results = manager.run_test(
        test_image = r"D:\Data\ocr_test\Alice\Alice_test_image0.png",
        reference_text_file = r"D:\Data\ocr_test\Alice\alice0.txt"
    )
    """