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

    test_cases = [
        {"audio": fr"{base_dir}\{prefix}_{i:04d}.wav",
         "reference": fr"{base_dir}\{prefix}_{i:04d}.txt"}
        for prefix in ['sample', 'librispeech']
        for i in range(2)
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
    manager.run_test2(test_cases=test_cases)

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
