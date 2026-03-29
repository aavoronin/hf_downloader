from ASR.ASRManager import ASRManager


def ASR_main():
    root_folder = r"D:\AIs\Automatic Speech Recognition"

    manager = ASRManager(root_folder)

    print("📦 Available Models:")
    models = manager.list_models()
    for i, model in enumerate(models[:10], 1):
        print(f"{i}. {model.name} - {model.size_human}")

    audio_path = r"D:\Data\audio_test\OSR_us_000_0014_8k.wav"
    reference_path = r"D:\Data\audio_test\OSR_us_000_0014_8k.txt"

    manager.run_test(
        audio_path=audio_path,
        reference_path=reference_path
    )

    stats = manager.apply_all(audio_path)
    print(f"\n✅ Successful models: {stats['successful_models']}")

