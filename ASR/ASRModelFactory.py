import json
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, List, Any

from ASR.AutomaticSpeechRecognition import AutomaticSpeechRecognition, GLOBAL_CONFIG
from ASR.ModelErrorLog import ModelErrorLog
from ASR.ModelInfo import ModelInfo
#from ASR.AutomaticSpeechRecognition import AutomaticSpeechRecognition


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
                model_name=model_name
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

        # Skip pure ONNX models (not supported by AutoModelForSpeechSeq2Seq)
        if any(f.endswith('.onnx') for f in files) and not any(
                f.endswith(('.safetensors', '.bin', '.pt', '.pth')) or 'pytorch_model' in f for f in files
        ):
            print(f"   ⚠ Skipping ONNX-only model (not PyTorch compatible)")
            return False

        has_config = any('config.json' in f for f in files)
        has_weights = any(
            f.endswith(('.safetensors', '.bin', '.pt', '.pth')) or
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
            except Exception as e:
                print(f"⚠ Warning: Failed to read stats file {self.stats_path}: {e}")
                existing = {}

        if 'processing_history' not in existing:
            existing['processing_history'] = []

        existing['processing_history'].append({
            'timestamp': datetime.now().isoformat(),
            **stats
        })

        try:
            with open(self.stats_path, 'w', encoding='utf-8') as f:
                json.dump(existing, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"❌ Error: Failed to save statistics to {self.stats_path}: {e}")


TORCH_AVAILABLE = True
torch = None

