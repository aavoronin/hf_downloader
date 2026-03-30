from ASR.ASRManager import ASRManager


def ASR_main():
    root_folder = r"D:\AIs\Automatic Speech Recognition"

    manager = ASRManager(root_folder)

    print("📦 Available Models:")
    models = manager.list_models()
    for i, model in enumerate(models, 1):
        print(f"{i}. {model.name} - {model.size_human}")

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
