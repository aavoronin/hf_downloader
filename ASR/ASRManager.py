# ASRManager.py
"""
ASR Model Manager - Fixed import handling
"""

import os
import re
import json
import time
import hashlib
import traceback
from datetime import datetime
from pathlib import Path
from typing import Union, List, Optional, Dict, Any
from dataclasses import dataclass, field, asdict
import shutil

# === FIXED: Import each package independently ===
# Import torch first
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False
    print("⚠ Warning: torch not available")

# Import transformers components separately
try:
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

    TRANSFORMERS_AVAILABLE = True
except ImportError as e:
    AutoModelForSpeechSeq2Seq = None
    AutoProcessor = None
    pipeline = None
    TRANSFORMERS_AVAILABLE = False
    print(f"⚠ Warning: transformers import failed: {e}")

# Import optional dependencies
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

GLOBAL_CONFIG = {
    "use_gpu": False,
    "max_errors_threshold": 10,
    "config_filename": "asr_manager_config.json",
    "stats_filename": "asr_manager_stats.json"
}


@dataclass
class ModelInfo:
    name: str
    folder_name: str
    size_bytes: int
    size_human: str
    path: Path
    files: List[str]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ModelErrorLog:
    model_name: str
    error_count: int = 0
    last_error_date: Optional[str] = None
    last_error_message: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> 'ModelErrorLog':
        return cls(**data)


@dataclass
class ProcessingResult:
    model_name: str
    text: Optional[str]
    success: bool
    time_taken: float
    datetime_completed: str
    error_message: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


class ASRModelFactory:

    def __init__(self, root_folder: str):
        self.root_folder = Path(root_folder)
        self.config_path = self.root_folder / GLOBAL_CONFIG["config_filename"]
        self.stats_path = self.root_folder / GLOBAL_CONFIG["stats_filename"]
        self._error_logs: Dict[str, ModelErrorLog] = {}
        self._load_config()

    def _load_config(self):
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for name, log_data in data.get('error_logs', {}).items():
                        self._error_logs[name] = ModelErrorLog.from_dict(log_data)
            except (json.JSONDecodeError, IOError):
                pass

    def _save_config(self):
        data = {
            'error_logs': {name: log.to_dict() for name, log in self._error_logs.items()}
        }
        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _log_error(self, model_name: str, error_message: str):
        if model_name not in self._error_logs:
            self._error_logs[model_name] = ModelErrorLog(model_name=model_name)
        log = self._error_logs[model_name]
        log.error_count += 1
        log.last_error_date = datetime.now().isoformat()
        log.last_error_message = error_message
        self._save_config()

    def get_error_count(self, model_name: str) -> int:
        return self._error_logs.get(model_name, ModelErrorLog(model_name=model_name)).error_count

    def is_model_faulty(self, model_name: str) -> bool:
        return self.get_error_count(model_name) > GLOBAL_CONFIG["max_errors_threshold"]

    def create(self, model_name: str) -> Optional['AutomaticSpeechRecognition']:
        print(f"\n🔍 Creating model: {model_name}")
        print(f"   Error count: {self.get_error_count(model_name)}/{GLOBAL_CONFIG['max_errors_threshold']}")
        print(f"   Is faulty: {self.is_model_faulty(model_name)}")

        if self.is_model_faulty(model_name):
            print(f"⊘ Skipping faulty model: {model_name} (errors > {GLOBAL_CONFIG['max_errors_threshold']})")
            return None
        model_folder = self._find_model_folder(model_name)
        if not model_folder:
            error_msg = f"Model folder not found: {model_name}"
            print(f"❌ {error_msg}")
            self._log_error(model_name, error_msg)
            return None
        if not self._verify_model_files(model_folder):
            error_msg = "Model files verification failed"
            print(f"❌ {error_msg} - Path: {model_folder}")
            files_found = list(model_folder.rglob('*'))[:10]
            print(f"   Files found: {[f.name for f in files_found]}")
            self._log_error(model_name, error_msg)
            return None
        try:
            print(f"   Attempting to initialize AutomaticSpeechRecognition...")
            model = AutomaticSpeechRecognition(
                model_path=model_folder,
                model_name=model_name,
                factory=self
            )
            print(f"   ✓ Model initialized successfully")
            return model
        except Exception as e:
            error_msg = f"Initialization error: {str(e)}"
            print(f"❌ {error_msg}")
            print(f"   Full traceback:")
            traceback.print_exc()
            self._log_error(model_name, error_msg)
            return None

    def _find_model_folder(self, model_name: str) -> Optional[Path]:
        folder_name = model_name.replace('/', '_')
        print(f"   Searching for folder: {folder_name}")
        for item in self.root_folder.iterdir():
            if item.is_dir() and item.name == folder_name:
                if (item / "model_info.json").exists() or (item / "config.json").exists():
                    print(f"   ✓ Found: {item}")
                    return item
        print(f"   ✗ Not found in {self.root_folder}")
        return None

    def _verify_model_files(self, folder: Path) -> bool:
        files = [f.name.lower() for f in folder.rglob('*') if f.is_file()]
        has_config = any('config.json' in f for f in files)
        has_weights = any(
            f.endswith(('.safetensors', '.bin', '.pt', '.pth', '.nemo', '.onnx')) or
            'pytorch_model' in f for f in files
        )
        print(f"   Config found: {has_config}, Weights found: {has_weights}")
        return has_config and has_weights

    def list_available_models(self) -> List[ModelInfo]:
        models = []
        for item in self.root_folder.iterdir():
            if not item.is_dir():
                continue
            model_info_file = item / "model_info.json"
            config_file = item / "config.json"
            if not model_info_file.exists() and not config_file.exists():
                continue
            total_size = self._calculate_folder_size(item)
            model_name = item.name.replace('_', '/', 1) if '_' in item.name else item.name
            files = [str(f.relative_to(item)) for f in item.rglob('*') if f.is_file()]
            models.append(ModelInfo(
                name=model_name,
                folder_name=item.name,
                size_bytes=total_size,
                size_human=self._format_size(total_size),
                path=item,
                files=files
            ))
        models.sort(key=lambda m: m.size_bytes, reverse=True)
        return models

    def _calculate_folder_size(self, folder: Path) -> int:
        total = 0
        for f in folder.rglob('*'):
            if f.is_file():
                try:
                    total += f.stat().st_size
                except OSError:
                    pass
        return total

    def _format_size(self, size_bytes: int) -> str:
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024
        return f"{size_bytes:.2f} PB"

    def save_statistics(self, stats: Dict[str, Any]):
        existing = {}
        if self.stats_path.exists():
            try:
                with open(self.stats_path, 'r', encoding='utf-8') as f:
                    existing = json.load(f)
            except (json.JSONDecodeError, IOError):
                pass
        if 'processing_history' not in existing:
            existing['processing_history'] = []
        existing['processing_history'].append({
            'timestamp': datetime.now().isoformat(),
            **stats
        })
        with open(self.stats_path, 'w', encoding='utf-8') as f:
            json.dump(existing, f, indent=2, ensure_ascii=False)


class AutomaticSpeechRecognition:

    def __init__(self, model_path: Path, model_name: str, factory: ASRModelFactory):
        print(f"   [ASR] Initializing {model_name}...")
        self.model_path = model_path
        self.model_name = model_name
        self.factory = factory
        self._pipeline = None
        self._device = self._determine_device()

        # === FIXED: Check each package independently ===
        if not TORCH_AVAILABLE:
            raise ImportError("torch is required. Install with: pip install torch")
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers is required. Install with: pip install transformers")

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
            processor = AutoProcessor.from_pretrained(str(self.model_path), local_files_only=True)
            print(f"   [Pipeline] Processor loaded")

            model_dtype = torch.float16 if self._device == "cuda" else torch.float32
            print(f"   [Pipeline] Loading model with dtype={model_dtype}...")

            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                str(self.model_path),
                local_files_only=True,
                dtype=model_dtype,
                low_cpu_mem_usage=True
            ).to(self._device)
            print(f"   [Pipeline] Model loaded and moved to {self._device}")

            device_arg = 0 if self._device in ("cuda", "npu") else -1

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
                dtype=model_dtype,
                generate_kwargs=generate_kwargs
            )
            print(f"   [Pipeline] ✓ Pipeline ready")
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


class ASRManager:

    def __init__(self, root_folder: str):
        self.factory = ASRModelFactory(root_folder)

    def list_models(self) -> List[ModelInfo]:
        return self.factory.list_available_models()

    def get_model(self, model_name: str) -> Optional[AutomaticSpeechRecognition]:
        return self.factory.create(model_name)

    def apply_all(
            self,
            input_paths: Union[str, Path, List[Union[str, Path]]],
            model_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        if model_names is None:
            available = self.factory.list_available_models()
            model_names = [m.name for m in available]
        results = []
        successful_models = []
        for model_name in model_names:
            if self.factory.is_model_faulty(model_name):
                results.append(ProcessingResult(
                    model_name=model_name,
                    text=None,
                    success=False,
                    time_taken=0,
                    datetime_completed=datetime.now().isoformat(),
                    error_message="Model marked as faulty (>10 errors)"
                ))
                continue
            start_time = time.time()
            try:
                model = self.factory.create(model_name)
                if model is None:
                    raise RuntimeError("Failed to create model instance")
                text = model.process(input_paths)
                elapsed = time.time() - start_time
                result = ProcessingResult(
                    model_name=model_name,
                    text=text,
                    success=True,
                    time_taken=elapsed,
                    datetime_completed=datetime.now().isoformat()
                )
                successful_models.append(model_name)
            except Exception as e:
                elapsed = time.time() - start_time
                print(f"❌ Model {model_name} failed:")
                print(f"   Error: {str(e)}")
                traceback.print_exc()
                self.factory._log_error(model_name, str(e))
                result = ProcessingResult(
                    model_name=model_name,
                    text=None,
                    success=False,
                    time_taken=elapsed,
                    datetime_completed=datetime.now().isoformat(),
                    error_message=str(e)
                )
            results.append(result)
        stats = {
            'input_paths': [str(p) for p in (input_paths if isinstance(input_paths, list) else [input_paths])],
            'models_tested': len(model_names),
            'successful_models': successful_models,
            'results': [r.to_dict() for r in results]
        }
        self.factory.save_statistics(stats)
        return stats

    def run_test(
            self,
            audio_path: str,
            reference_path: str,
            model_names: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        try:
            with open(reference_path, 'r', encoding='utf-8') as f:
                reference_text = f.read().strip()
                reference_text = reference_text.replace("\r", " ").replace("\n", " ")
                for _ in range(10):
                    reference_text = reference_text.replace("  ", " ")
        except FileNotFoundError:
            print(f"❌ Reference file not found: {reference_path}")
            return []
        print(f"\n🧪 Running ASR Benchmark Test")
        print(f"📁 Audio: {audio_path}")
        print(f"📄 Reference: {reference_path}")
        print(f"📏 Reference length: {len(reference_text)} chars\n")
        if model_names is None:
            available = self.factory.list_available_models()
            model_names = [m.name for m in available]
        print(f"🔄 Testing {len(model_names)} models...\n")
        results = []
        for model_name in model_names:
            if self.factory.is_model_faulty(model_name):
                print(f"⊘ {model_name}: SKIPPED (faulty)")
                continue
            start_time = time.time()
            try:
                print(f"\n{'=' * 60}")
                print(f" Testing: {model_name}")
                print(f"{'=' * 60}")
                model = self.factory.create(model_name)
                if model is None:
                    print(f"✗ {model_name}: FAILED to initialize")
                    continue
                predicted_text = model.process(audio_path)
                print(f"\n📝 Predicted: {predicted_text}")
                elapsed = time.time() - start_time
                similarity = self._normalized_levenshtein_similarity(
                    reference_text.lower(), predicted_text.lower())
                status = "✓" if similarity > 0.5 else "⚠"
                print(f"\n{status} {model_name}")
                print(f"  Similarity: {similarity:.4f} | Time: {elapsed:.2f}s")
                results.append({
                    'model_name': model_name,
                    'similarity': similarity,
                    'success': True,
                    'time_taken': elapsed
                })
            except Exception as e:
                elapsed = time.time() - start_time
                print(f"\n✗ {model_name}: ERROR")
                print(f"  Error: {str(e)}")
                print(f"  Traceback:")
                traceback.print_exc()
                self.factory._log_error(model_name, str(e))
                results.append({
                    'model_name': model_name,
                    'similarity': 0.0,
                    'success': False,
                    'time_taken': elapsed,
                    'error': str(e)
                })
        print(f"\n📊 Test Summary")
        print(f"{'Model Name':<50} {'Similarity':>10} {'Status':>10} {'Time (s)':>10}")
        print("-" * 82)
        for r in sorted(results, key=lambda x: x['similarity'], reverse=True):
            status = "SUCCESS" if r['success'] else "FAILED"
            print(f"{r['model_name']:<50} {r['similarity']:>10.4f} {status:>10} {r['time_taken']:>10.2f}")
        return results

    @staticmethod
    def _normalized_levenshtein_similarity(s1: str, s2: str) -> float:
        if not s1 and not s2:
            return 1.0
        if not s1 or not s2:
            return 0.0

        def levenshtein_distance(a: str, b: str) -> int:
            if len(a) < len(b):
                a, b = b, a
            previous_row = list(range(len(b) + 1))
            for i, c1 in enumerate(a):
                current_row = [i + 1]
                for j, c2 in enumerate(b):
                    insertions = previous_row[j + 1] + 1
                    deletions = current_row[j] + 1
                    substitutions = previous_row[j] + (c1 != c2)
                    current_row.append(min(insertions, deletions, substitutions))
                previous_row = current_row
            return previous_row[-1]

        distance = levenshtein_distance(s1.lower(), s2.lower())
        max_len = max(len(s1), len(s2))
        return 1.0 - (distance / max_len)


def set_device(use_gpu: bool):
    """Global switch between CPU and GPU mode."""
    GLOBAL_CONFIG["use_gpu"] = use_gpu
    device = "GPU" if use_gpu else "CPU"
    print(f"🔧 Device set to: {device}")# ASRManager.py
"""
ASR Model Manager - Fixed import handling
"""

import os
import re
import json
import time
import hashlib
import traceback
from datetime import datetime
from pathlib import Path
from typing import Union, List, Optional, Dict, Any
from dataclasses import dataclass, field, asdict
import shutil

# === FIXED: Import each package independently ===
# Import torch first
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False
    print("⚠ Warning: torch not available")

# Import transformers components separately
try:
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError as e:
    AutoModelForSpeechSeq2Seq = None
    AutoProcessor = None
    pipeline = None
    TRANSFORMERS_AVAILABLE = False
    print(f"⚠ Warning: transformers import failed: {e}")

# Import optional dependencies
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

GLOBAL_CONFIG = {
    "use_gpu": False,
    "max_errors_threshold": 10,
    "config_filename": "asr_manager_config.json",
    "stats_filename": "asr_manager_stats.json"
}


@dataclass
class ModelInfo:
    name: str
    folder_name: str
    size_bytes: int
    size_human: str
    path: Path
    files: List[str]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ModelErrorLog:
    model_name: str
    error_count: int = 0
    last_error_date: Optional[str] = None
    last_error_message: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> 'ModelErrorLog':
        return cls(**data)


@dataclass
class ProcessingResult:
    model_name: str
    text: Optional[str]
    success: bool
    time_taken: float
    datetime_completed: str
    error_message: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


class ASRModelFactory:

    def __init__(self, root_folder: str):
        self.root_folder = Path(root_folder)
        self.config_path = self.root_folder / GLOBAL_CONFIG["config_filename"]
        self.stats_path = self.root_folder / GLOBAL_CONFIG["stats_filename"]
        self._error_logs: Dict[str, ModelErrorLog] = {}
        self._load_config()

    def _load_config(self):
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for name, log_data in data.get('error_logs', {}).items():
                        self._error_logs[name] = ModelErrorLog.from_dict(log_data)
            except (json.JSONDecodeError, IOError):
                pass

    def _save_config(self):
        data = {
            'error_logs': {name: log.to_dict() for name, log in self._error_logs.items()}
        }
        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _log_error(self, model_name: str, error_message: str):
        if model_name not in self._error_logs:
            self._error_logs[model_name] = ModelErrorLog(model_name=model_name)
        log = self._error_logs[model_name]
        log.error_count += 1
        log.last_error_date = datetime.now().isoformat()
        log.last_error_message = error_message
        self._save_config()

    def get_error_count(self, model_name: str) -> int:
        return self._error_logs.get(model_name, ModelErrorLog(model_name=model_name)).error_count

    def is_model_faulty(self, model_name: str) -> bool:
        return self.get_error_count(model_name) > GLOBAL_CONFIG["max_errors_threshold"]

    def create(self, model_name: str) -> Optional['AutomaticSpeechRecognition']:
        print(f"\n🔍 Creating model: {model_name}")
        print(f"   Error count: {self.get_error_count(model_name)}/{GLOBAL_CONFIG['max_errors_threshold']}")
        print(f"   Is faulty: {self.is_model_faulty(model_name)}")

        if self.is_model_faulty(model_name):
            print(f"⊘ Skipping faulty model: {model_name} (errors > {GLOBAL_CONFIG['max_errors_threshold']})")
            return None
        model_folder = self._find_model_folder(model_name)
        if not model_folder:
            error_msg = f"Model folder not found: {model_name}"
            print(f"❌ {error_msg}")
            self._log_error(model_name, error_msg)
            return None
        if not self._verify_model_files(model_folder):
            error_msg = "Model files verification failed"
            print(f"❌ {error_msg} - Path: {model_folder}")
            files_found = list(model_folder.rglob('*'))[:10]
            print(f"   Files found: {[f.name for f in files_found]}")
            self._log_error(model_name, error_msg)
            return None
        try:
            print(f"   Attempting to initialize AutomaticSpeechRecognition...")
            model = AutomaticSpeechRecognition(
                model_path=model_folder,
                model_name=model_name,
                factory=self
            )
            print(f"   ✓ Model initialized successfully")
            return model
        except Exception as e:
            error_msg = f"Initialization error: {str(e)}"
            print(f"❌ {error_msg}")
            print(f"   Full traceback:")
            traceback.print_exc()
            self._log_error(model_name, error_msg)
            return None

    def _find_model_folder(self, model_name: str) -> Optional[Path]:
        folder_name = model_name.replace('/', '_')
        print(f"   Searching for folder: {folder_name}")
        for item in self.root_folder.iterdir():
            if item.is_dir() and item.name == folder_name:
                if (item / "model_info.json").exists() or (item / "config.json").exists():
                    print(f"   ✓ Found: {item}")
                    return item
        print(f"   ✗ Not found in {self.root_folder}")
        return None

    def _verify_model_files(self, folder: Path) -> bool:
        files = [f.name.lower() for f in folder.rglob('*') if f.is_file()]
        has_config = any('config.json' in f for f in files)
        has_weights = any(
            f.endswith(('.safetensors', '.bin', '.pt', '.pth', '.nemo', '.onnx')) or
            'pytorch_model' in f for f in files
        )
        print(f"   Config found: {has_config}, Weights found: {has_weights}")
        return has_config and has_weights

    def list_available_models(self) -> List[ModelInfo]:
        models = []
        for item in self.root_folder.iterdir():
            if not item.is_dir():
                continue
            model_info_file = item / "model_info.json"
            config_file = item / "config.json"
            if not model_info_file.exists() and not config_file.exists():
                continue
            total_size = self._calculate_folder_size(item)
            model_name = item.name.replace('_', '/', 1) if '_' in item.name else item.name
            files = [str(f.relative_to(item)) for f in item.rglob('*') if f.is_file()]
            models.append(ModelInfo(
                name=model_name,
                folder_name=item.name,
                size_bytes=total_size,
                size_human=self._format_size(total_size),
                path=item,
                files=files
            ))
        models.sort(key=lambda m: m.size_bytes, reverse=True)
        return models

    def _calculate_folder_size(self, folder: Path) -> int:
        total = 0
        for f in folder.rglob('*'):
            if f.is_file():
                try:
                    total += f.stat().st_size
                except OSError:
                    pass
        return total

    def _format_size(self, size_bytes: int) -> str:
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024
        return f"{size_bytes:.2f} PB"

    def save_statistics(self, stats: Dict[str, Any]):
        existing = {}
        if self.stats_path.exists():
            try:
                with open(self.stats_path, 'r', encoding='utf-8') as f:
                    existing = json.load(f)
            except (json.JSONDecodeError, IOError):
                pass
        if 'processing_history' not in existing:
            existing['processing_history'] = []
        existing['processing_history'].append({
            'timestamp': datetime.now().isoformat(),
            **stats
        })
        with open(self.stats_path, 'w', encoding='utf-8') as f:
            json.dump(existing, f, indent=2, ensure_ascii=False)


class AutomaticSpeechRecognition:

    def __init__(self, model_path: Path, model_name: str, factory: ASRModelFactory):
        print(f"   [ASR] Initializing {model_name}...")
        self.model_path = model_path
        self.model_name = model_name
        self.factory = factory
        self._pipeline = None
        self._device = self._determine_device()

        # === FIXED: Check each package independently ===
        if not TORCH_AVAILABLE:
            raise ImportError("torch is required. Install with: pip install torch")
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers is required. Install with: pip install transformers")

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
            processor = AutoProcessor.from_pretrained(str(self.model_path), local_files_only=True)
            print(f"   [Pipeline] Processor loaded")

            model_dtype = torch.float16 if self._device == "cuda" else torch.float32
            print(f"   [Pipeline] Loading model with dtype={model_dtype}...")

            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                str(self.model_path),
                local_files_only=True,
                dtype=model_dtype,
                low_cpu_mem_usage=True
            ).to(self._device)
            print(f"   [Pipeline] Model loaded and moved to {self._device}")

            device_arg = 0 if self._device in ("cuda", "npu") else -1

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
                dtype=model_dtype,
                generate_kwargs=generate_kwargs
            )
            print(f"   [Pipeline] ✓ Pipeline ready")
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


class ASRManager:

    def __init__(self, root_folder: str):
        self.factory = ASRModelFactory(root_folder)

    def list_models(self) -> List[ModelInfo]:
        return self.factory.list_available_models()

    def get_model(self, model_name: str) -> Optional[AutomaticSpeechRecognition]:
        return self.factory.create(model_name)

    def apply_all(
            self,
            input_paths: Union[str, Path, List[Union[str, Path]]],
            model_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        if model_names is None:
            available = self.factory.list_available_models()
            model_names = [m.name for m in available]
        results = []
        successful_models = []
        for model_name in model_names:
            if self.factory.is_model_faulty(model_name):
                results.append(ProcessingResult(
                    model_name=model_name,
                    text=None,
                    success=False,
                    time_taken=0,
                    datetime_completed=datetime.now().isoformat(),
                    error_message="Model marked as faulty (>10 errors)"
                ))
                continue
            start_time = time.time()
            try:
                model = self.factory.create(model_name)
                if model is None:
                    raise RuntimeError("Failed to create model instance")
                text = model.process(input_paths)
                elapsed = time.time() - start_time
                result = ProcessingResult(
                    model_name=model_name,
                    text=text,
                    success=True,
                    time_taken=elapsed,
                    datetime_completed=datetime.now().isoformat()
                )
                successful_models.append(model_name)
            except Exception as e:
                elapsed = time.time() - start_time
                print(f"❌ Model {model_name} failed:")
                print(f"   Error: {str(e)}")
                traceback.print_exc()
                self.factory._log_error(model_name, str(e))
                result = ProcessingResult(
                    model_name=model_name,
                    text=None,
                    success=False,
                    time_taken=elapsed,
                    datetime_completed=datetime.now().isoformat(),
                    error_message=str(e)
                )
            results.append(result)
        stats = {
            'input_paths': [str(p) for p in (input_paths if isinstance(input_paths, list) else [input_paths])],
            'models_tested': len(model_names),
            'successful_models': successful_models,
            'results': [r.to_dict() for r in results]
        }
        self.factory.save_statistics(stats)
        return stats

    def run_test(
            self,
            audio_path: str,
            reference_path: str,
            model_names: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        try:
            with open(reference_path, 'r', encoding='utf-8') as f:
                reference_text = f.read().strip()
                reference_text = reference_text.replace("\r", " ").replace("\n", " ")
                for _ in range(10):
                    reference_text = reference_text.replace("  ", " ")
        except FileNotFoundError:
            print(f"❌ Reference file not found: {reference_path}")
            return []
        print(f"\n🧪 Running ASR Benchmark Test")
        print(f"📁 Audio: {audio_path}")
        print(f"📄 Reference: {reference_path}")
        print(f"📏 Reference length: {len(reference_text)} chars\n")
        if model_names is None:
            available = self.factory.list_available_models()
            model_names = [m.name for m in available]
        print(f"🔄 Testing {len(model_names)} models...\n")
        results = []
        for model_name in model_names:
            if self.factory.is_model_faulty(model_name):
                print(f"⊘ {model_name}: SKIPPED (faulty)")
                continue
            start_time = time.time()
            try:
                print(f"\n{'=' * 60}")
                print(f" Testing: {model_name}")
                print(f"{'=' * 60}")
                model = self.factory.create(model_name)
                if model is None:
                    print(f"✗ {model_name}: FAILED to initialize")
                    continue
                predicted_text = model.process(audio_path)
                print(f"\n📝 Predicted: {predicted_text}")
                elapsed = time.time() - start_time
                similarity = self._normalized_levenshtein_similarity(
                    reference_text.lower(), predicted_text.lower())
                status = "✓" if similarity > 0.5 else "⚠"
                print(f"\n{status} {model_name}")
                print(f"  Similarity: {similarity:.4f} | Time: {elapsed:.2f}s")
                results.append({
                    'model_name': model_name,
                    'similarity': similarity,
                    'success': True,
                    'time_taken': elapsed
                })
            except Exception as e:
                elapsed = time.time() - start_time
                print(f"\n✗ {model_name}: ERROR")
                print(f"  Error: {str(e)}")
                print(f"  Traceback:")
                traceback.print_exc()
                self.factory._log_error(model_name, str(e))
                results.append({
                    'model_name': model_name,
                    'similarity': 0.0,
                    'success': False,
                    'time_taken': elapsed,
                    'error': str(e)
                })
        print(f"\n📊 Test Summary")
        print(f"{'Model Name':<50} {'Similarity':>10} {'Status':>10} {'Time (s)':>10}")
        print("-" * 82)
        for r in sorted(results, key=lambda x: x['similarity'], reverse=True):
            status = "SUCCESS" if r['success'] else "FAILED"
            print(f"{r['model_name']:<50} {r['similarity']:>10.4f} {status:>10} {r['time_taken']:>10.2f}")
        return results

    @staticmethod
    def _normalized_levenshtein_similarity(s1: str, s2: str) -> float:
        if not s1 and not s2:
            return 1.0
        if not s1 or not s2:
            return 0.0

        def levenshtein_distance(a: str, b: str) -> int:
            if len(a) < len(b):
                a, b = b, a
            previous_row = list(range(len(b) + 1))
            for i, c1 in enumerate(a):
                current_row = [i + 1]
                for j, c2 in enumerate(b):
                    insertions = previous_row[j + 1] + 1
                    deletions = current_row[j] + 1
                    substitutions = previous_row[j] + (c1 != c2)
                    current_row.append(min(insertions, deletions, substitutions))
                previous_row = current_row
            return previous_row[-1]

        distance = levenshtein_distance(s1.lower(), s2.lower())
        max_len = max(len(s1), len(s2))
        return 1.0 - (distance / max_len)


def set_device(use_gpu: bool):
    """Global switch between CPU and GPU mode."""
    GLOBAL_CONFIG["use_gpu"] = use_gpu
    device = "GPU" if use_gpu else "CPU"
    print(f"🔧 Device set to: {device}")