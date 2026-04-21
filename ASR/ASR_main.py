import os

from ASR.ASRManager import ASRManager
from ASR.AutomaticSpeechRecognition import ASR_set_device


def ASR_main():
    # 1. Configure Device
    # Set to True if you have CUDA available and want GPU acceleration
    ASR_set_device(use_gpu=True)

    root_folder = r"D:\AIs\Automatic Speech Recognition"

    manager = ASRManager(root_folder)

    print("📦 Available Models:")
    models = manager.list_models()
    for i, model in enumerate(models, 1):
        print(f"{i}. {model.name} - {model.size_human}")

    # 3. Generate List of Files using a single list comprehension
    # Includes sample_0000 to sample_0009 AND librispeech_0000 to librispeech_0009
    base_dir = r"D:\Data\audio_test"

    r"""
    test_cases = [
        {"audio": fr"{base_dir}\{prefix}_{i:04d}.wav",
         "reference": fr"{base_dir}\{prefix}_{i:04d}.txt"}
        for prefix in ['sample', 'librispeech']
        for i in range(10)
    ]"""

    base_dir = r"D:\Data\audio_test3_500_20_20\all"

    # Specific file identifiers extracted from your list
    file_ids = [
        "sample_0059", "sample_0025", "sample_0329", "sample_0405", "sample_0485",
        "sample_0056", "sample_0079", "sample_0174", "sample_0165", "sample_0270",
        "sample_0465", "sample_0097", "sample_0430", "sample_0456", "sample_0114",
        "sample_0014", "sample_0049", "sample_0021", "sample_0192", "sample_0358",
        "librispeech_0130", "librispeech_0202", "librispeech_0140", "librispeech_0142", "librispeech_0143",
        "librispeech_0145", "librispeech_0452", "librispeech_0466", "librispeech_0125", "librispeech_0144",
        "librispeech_0149", "librispeech_0182", "librispeech_0217", "librispeech_0141", "librispeech_0119",
        "librispeech_0126", "librispeech_0127", "librispeech_0139", "librispeech_0191", "librispeech_0327"
    ]

    test_cases = [
        {"audio": fr"{base_dir}\{name}.wav",
         "reference": fr"{base_dir}\{name}.txt"}
        for name in file_ids
    ]

    if not test_cases:
        print("❌ No valid test cases found. Check paths.")
        return

    print(f"✅ Loaded {len(test_cases)} test cases for batch processing.")

    # 4. Initialize Manager
    manager = ASRManager(root_folder=root_folder)

    # 5. Run Batch Test
    # You can optionally pass model_names=["model_name_1", "model_name_2"]
    # to test specific models instead of all available ones.
    manager.set_model_filter_mode("ru_models")
    manager.run_test2([t for t in test_cases if "sample" in t["audio"]])
    manager.set_model_filter_mode("en_models")
    manager.run_test2([t for t in test_cases if "librispeech" in t["audio"]])
    #manager.set_model_filter_mode("all_models")
    #manager.run_test2(test_cases[:2])

    r"""

    for s in [
            #"OSR_us_000_0014_8k",
            "testaudio_16000_test01_20s",
            "sample_0000", "sample_0001", "sample_0002", "sample_0003",
            ]:
        audio_path = fr"D:\Data\audio_test\{s}.wav"
        reference_path = fr"D:\Data\audio_test\{s}.txt"

        manager.run_test(
            audio_path=audio_path,
            reference_path=reference_path
        )

    audio_path = r"D:\Data\audio_test\testaudio_16000_test01_20s.wav"
    stats = manager.apply_all(audio_path)
    print(f"\n✅ Successful models: {stats['successful_models']}")
    
    """
