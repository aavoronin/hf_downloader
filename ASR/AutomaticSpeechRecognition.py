import hashlib
import traceback
from pathlib import Path
from typing import Dict, Any, Union, List

#from ASR.ASRModelFactory import GLOBAL_CONFIG

#from ASR.ASRManager import TRANSFORMERS_AVAILABLE, AutoProcessor, AutoModelForSpeechSeq2Seq, pipeline, MOVIEPY_AVAILABLE, LIBROSA_AVAILABLE, librosa
#from ASR.ASRModelFactory import ASRModelFactory, TORCH_AVAILABLE, torch, GLOBAL_CONFIG

# === Import each package independently ===
try:
    import torch
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠ Warning: torch not available")

try:
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

    TRANSFORMERS_AVAILABLE = True
except ImportError as e:
    AutoModelForSpeechSeq2Seq = None
    AutoProcessor = None
    pipeline = None
    TRANSFORMERS_AVAILABLE = False
    print(f"⚠ Warning: transformers import failed: {e}")

try:
    import librosa

    LIBROSA_AVAILABLE = True
except ImportError:
    librosa = None
    LIBROSA_AVAILABLE = False
    print("⚠ Warning: librosa not available")

try:
    import moviepy.editor as mp

    MOVIEPY_AVAILABLE = True
except ImportError:
    moviepy = None
    MOVIEPY_AVAILABLE = False
    print("⚠ Warning: moviepy not available")



class AutomaticSpeechRecognition:

    def __init__(self, model_path: Path, model_name: str):
        print(f"   [ASR] Initializing {model_name}...")
        self.model_path = model_path
        self.model_name = model_name
        self._pipeline = None
        self._device = self._determine_device()


        print(f"   [ASR] Device: {self._device}")

    def _determine_device(self) -> str:
        if not GLOBAL_CONFIG["use_gpu"]:
            return "cpu"
        if torch and torch.cuda.is_available():
            return "cuda"
        if torch and hasattr(torch, 'npu') and torch.npu.is_available():
            return "npu"
        print(f"⚠ GPU requested but not available, using CPU for {self.model_name}")
        return "cpu"

    def _load_pipeline(self):
        if self._pipeline is not None:
            return
        print(f"   [Pipeline] Loading pipeline for {self.model_name}...")
        try:
            print(f"   [Pipeline] Loading processor from {self.model_path}...")
            processor = AutoProcessor.from_pretrained(
                str(self.model_path),
                local_files_only=True,
                trust_remote_code=True  # For Granite/Phi/custom models
            )
            print(f"   [Pipeline] Processor loaded")

            model_dtype = torch.float16 if self._device == "cuda" else torch.float32
            print(f"   [Pipeline] Loading model with dtype={model_dtype}...")

            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                str(self.model_path),
                local_files_only=True,
                torch_dtype=model_dtype,  # ✅ FIXED: was 'dtype'
                low_cpu_mem_usage=True,
                trust_remote_code=True
            ).to(self._device)
            print(f"   [Pipeline] Model loaded and moved to {self._device}")

            # Modern device handling (string format)
            device_arg = self._device if self._device != "cpu" else "cpu"

            generate_kwargs = {
                "language": None,
                "return_timestamps": True
            }

            print(f"   [Pipeline] Creating pipeline...")
            self._pipeline = pipeline(
                "automatic-speech-recognition",
                model=model,
                tokenizer=processor.tokenizer,
                feature_extractor=processor.feature_extractor,
                device=device_arg,
                torch_dtype=model_dtype,  # ✅ FIXED: was 'dtype'
                generate_kwargs=generate_kwargs
            )
            print(f"   [Pipeline] ✓ Pipeline ready")
        except ValueError as e:
            if "does not recognize this architecture" in str(e):
                print(f"   ⚠ Model architecture not supported by transformers")
                raise RuntimeError(f"Unsupported model architecture: {self.model_name}")
            raise
        except Exception as e:
            print(f"   [Pipeline] ❌ Failed to load pipeline")
            print(f"   [Pipeline] Error: {str(e)}")
            print(f"   [Pipeline] Traceback:")
            traceback.print_exc()
            raise RuntimeError(f"Failed to load model {self.model_name}: {str(e)}")

    def _extract_audio(self, file_path: Path) -> str:
        if file_path.suffix.lower() in ['.mp4', '.avi', '.mkv', '.mov', '.webm', '.m4v']:
            if not MOVIEPY_AVAILABLE:
                raise ImportError("moviepy is required for video processing. Install with: pip install moviepy")
            temp_audio = Path(
                file_path.parent) / f".temp_{file_path.stem}_{hashlib.md5(str(file_path).encode()).hexdigest()[:8]}.wav"
            try:
                video = mp.VideoFileClip(str(file_path))
                video.audio.write_audiofile(str(temp_audio), verbose=False, logger=None)
                video.close()
                return str(temp_audio)
            except Exception as e:
                print(f"   [Audio] Failed to extract audio from {file_path}")
                traceback.print_exc()
                raise RuntimeError(f"Failed to extract audio from {file_path}: {str(e)}")
        return str(file_path)

    def _load_audio(self, file_path: str) -> Dict[str, Any]:
        if not LIBROSA_AVAILABLE:
            raise ImportError("librosa is required for audio processing. Install with: pip install librosa")
        print(f"   [Audio] Loading audio: {file_path}")
        audio, sr = librosa.load(file_path, sr=16000)
        print(f"   [Audio] Loaded: {len(audio)} samples @ {sr}Hz")
        return {"raw": audio, "sampling_rate": sr}

    def process(self, input_path: Union[str, Path, List[Union[str, Path]]]) -> str:
        print(f"   [Process] Processing: {input_path}")
        self._load_pipeline()
        paths = [input_path] if not isinstance(input_path, list) else input_path
        results = []
        temp_files = []
        try:
            for path in paths:
                path_obj = Path(path)
                audio_path = self._extract_audio(path_obj)
                if audio_path != str(path_obj):
                    temp_files.append(audio_path)
                audio_data = self._load_audio(audio_path)
                print(f"   [Process] Running inference...")
                result = self._pipeline(audio_data)
                print(f"   [Process] Inference complete")

                if isinstance(result, dict):
                    if 'text' in result:
                        results.append(result['text'].strip())
                    elif 'chunks' in result:
                        chunk_texts = [chunk.get('text', '') for chunk in result.get('chunks', []) if chunk.get('text')]
                        combined = ' '.join(chunk_texts).strip()
                        if combined:
                            results.append(combined)
                elif isinstance(result, str):
                    results.append(result.strip())
        except Exception as e:
            print(f"   [Process] ❌ Processing failed: {str(e)}")
            traceback.print_exc()
            raise
        finally:
            for temp_file in temp_files:
                try:
                    Path(temp_file).unlink(missing_ok=True)
                except OSError:
                    pass
        return " ".join(filter(None, results))


def ASR_set_device(use_gpu: bool):
    """Global switch between CPU and GPU mode."""
    GLOBAL_CONFIG["use_gpu"] = use_gpu
    device = "GPU" if use_gpu else "CPU"
    print(f"🔧 Device set to: {device}")


GLOBAL_CONFIG = {
    "use_gpu": False,
    "max_errors_threshold": 10,
    "config_filename": "asr_manager_config.json",
    "stats_filename": "asr_manager_stats.json"
}
