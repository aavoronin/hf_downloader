import os
import json
import time
import signal
import warnings
from datetime import datetime
from pathlib import Path
from typing import List, Union, Dict, Any, Optional
from dataclasses import dataclass, asdict
import torch
from PIL import Image
import difflib

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*deprecated.*")
warnings.filterwarnings("ignore", message=".*Tied weights.*")
warnings.filterwarnings("ignore", message=".*regex pattern.*")
warnings.filterwarnings("ignore", message=".*SentencePiece.*")

GLOBAL_CONFIG = {
    "use_gpu": False,
    "max_errors_before_ignore": 10,
    "config_file": "model_errors.json"
}


@dataclass
class ModelInfo:
    name: str
    size_mb: float
    path: str


@dataclass
class ModelResult:
    model_name: str
    text: str
    success: bool
    time_taken: float
    datetime_completed: str
    error_message: Optional[str] = None


@dataclass
class ModelErrorInfo:
    model_name: str
    error_count: int
    last_error_date: str
    is_faulty: bool = False


class TextToImage:
    MODEL_TYPE_MAP = {
        "trocr": "VisionEncoderDecoder",
        "blip": "BlipForConditionalGeneration",
        "blip-2": "Blip2ForConditionalGeneration",
        "git": "GitForCausalLM",
        "donut": "VisionEncoderDecoder",
        "pix2struct": "Pix2StructForConditionalGeneration",
        "vit-gpt2": "VisionEncoderDecoder",
        "mblip": "Blip2ForConditionalGeneration",
    }

    def __init__(self, model_name: str, model_path: str, use_gpu: bool = False):
        self.model_name = model_name
        self.model_path = model_path
        self.use_gpu = use_gpu
        self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        self.model = None
        self.processor = None
        self._initialized = False
        self._init_error = None

    def _detect_model_type(self, config) -> str:
        model_type = getattr(config, 'model_type', '').lower()
        arch = getattr(config, 'architectures', [])
        if arch:
            arch_name = arch[0].lower() if isinstance(arch, list) else str(arch).lower()
            if 'trocr' in arch_name:
                return 'trocr'
            elif 'blip2' in arch_name or 'blip-2' in arch_name:
                return 'blip-2'
            elif 'blip' in arch_name:
                return 'blip'
            elif 'git' in arch_name:
                return 'git'
            elif 'donut' in arch_name:
                return 'donut'
            elif 'pix2struct' in arch_name:
                return 'pix2struct'
            elif 'mblip' in arch_name:
                return 'mblip'
            elif 'vit' in arch_name and 'gpt2' in arch_name:
                return 'vit-gpt2'
        if model_type:
            for key in self.MODEL_TYPE_MAP:
                if key in model_type: return key
        return 'generic'

    def _load_model_with_correct_class(self, model_type: str):
        from transformers import (
            AutoConfig, AutoModel,
            VisionEncoderDecoderModel, BlipForConditionalGeneration,
            Blip2ForConditionalGeneration, GitForCausalLM,
            Pix2StructForConditionalGeneration
        )
        load_kwargs = {
            "cache_dir": self.model_path,
            "local_files_only": False,
            "torch_dtype": torch.float32,
            "low_cpu_mem_usage": True,
        }
        if model_type in ('trocr', 'donut', 'vit-gpt2'):
            return VisionEncoderDecoderModel.from_pretrained(self.model_path, **load_kwargs)
        elif model_type == 'blip':
            return BlipForConditionalGeneration.from_pretrained(self.model_path, **load_kwargs)
        elif model_type in ('blip-2', 'mblip'):
            return Blip2ForConditionalGeneration.from_pretrained(self.model_path, **load_kwargs)
        elif model_type == 'git':
            return GitForCausalLM.from_pretrained(self.model_path, **load_kwargs)
        elif model_type == 'pix2struct':
            return Pix2StructForConditionalGeneration.from_pretrained(self.model_path, **load_kwargs)
        else:
            try:
                from transformers import AutoModelForVision2Seq
                return AutoModelForVision2Seq.from_pretrained(self.model_path, **load_kwargs)
            except (ImportError, ValueError, OSError):
                return AutoModel.from_pretrained(self.model_path, **load_kwargs)

    def initialize(self) -> bool:
        try:
            from transformers import AutoProcessor, AutoConfig
            load_kwargs = {"cache_dir": self.model_path, "local_files_only": False}
            config = AutoConfig.from_pretrained(self.model_path, **load_kwargs)
            model_type = self._detect_model_type(config)
            try:
                self.processor = AutoProcessor.from_pretrained(self.model_path, **load_kwargs)
            except:
                from transformers import AutoTokenizer, AutoImageProcessor
                try:
                    self.processor = {
                        'tokenizer': AutoTokenizer.from_pretrained(self.model_path, **load_kwargs),
                        'image_processor': AutoImageProcessor.from_pretrained(self.model_path, **load_kwargs)
                    }
                except:
                    self.processor = None
            self.model = self._load_model_with_correct_class(model_type)
            self.model.to(self.device)
            self.model.eval()
            self._initialized = True
            return True
        except Exception as e:
            self._init_error = str(e)
            self._initialized = False
            return False

    def _prepare_inputs(self, image: Image.Image):
        if self.processor is None:
            raise RuntimeError("Processor not initialized")
        if isinstance(self.processor, dict):
            inputs = self.processor['image_processor'](images=image, return_tensors="pt")
            return {k: v.to(self.device) for k, v in inputs.items()}
        try:
            inputs = self.processor(images=image, return_tensors="pt")
            return {k: v.to(self.device) for k, v in inputs.items()}
        except TypeError:
            inputs = self.processor(image, return_tensors="pt")
            return {k: v.to(self.device) for k, v in inputs.items()}

    def _decode_output(self, generated_ids) -> str:
        if self.processor is None:
            return ""
        try:
            if isinstance(self.processor, dict):
                return self.processor['tokenizer'].batch_decode(generated_ids, skip_special_tokens=True)[0]
            return self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        except:
            if hasattr(self.processor, 'tokenizer'):
                return self.processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            return str(generated_ids)

    def process_image(self, image_path: Union[str, Path]) -> str:
        if not self._initialized:
            if not self.initialize():
                raise RuntimeError(f"Model {self.model_name} failed to initialize: {self._init_error}")
        try:
            image = Image.open(image_path).convert("RGB")
            inputs = self._prepare_inputs(image)

            if 'trocr' in self.model_name.lower():
                generate_kwargs = {
                    "max_length": 256,
                    "num_beams": 4,
                    "early_stopping": True,
                    "no_repeat_ngram_size": 3,
                    "do_sample": False,
                    "pad_token_id": self.model.config.pad_token_id if hasattr(self.model.config, 'pad_token_id') else 0,
                }
            elif 'donut' in self.model_name.lower():
                generate_kwargs = {
                    "max_length": 512,
                    "num_beams": 4,
                    "early_stopping": True,
                    "do_sample": False,
                    "pad_token_id": 0,
                }
            elif 'pix2struct' in self.model_name.lower():
                inputs = self.processor(
                    images=image,
                    text="What is the text in this image?",
                    return_tensors="pt"
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                generate_kwargs = {
                    "max_length": 256,
                    "num_beams": 3,
                    "early_stopping": True,
                }
            elif 'blip' in self.model_name.lower() or 'git' in self.model_name.lower():
                generate_kwargs = {
                    "max_length": 50,
                    "num_beams": 3,
                    "do_sample": False,
                    "early_stopping": True,
                }
            else:
                generate_kwargs = {
                    "max_length": 128,
                    "num_beams": 2,
                    "do_sample": False,
                    "early_stopping": True,
                }

            if hasattr(self.model.config, 'eos_token_id'):
                generate_kwargs["eos_token_id"] = self.model.config.eos_token_id
            if hasattr(self.model.config, 'pad_token_id'):
                generate_kwargs["pad_token_id"] = self.model.config.pad_token_id

            with torch.no_grad():
                generated_ids = self.model.generate(**inputs, **generate_kwargs)

            text = self._decode_output(generated_ids)
            return text.strip()
        except Exception as e:
            raise RuntimeError(f"Error processing image with {self.model_name}: {str(e)}")

    def process_images(self, image_paths: List[Union[str, Path]]) -> str:
        texts = []
        for img_path in image_paths:
            try:
                text = self.process_image(img_path)
                texts.append(text)
            except Exception as e:
                texts.append(f"[Error processing {img_path}: {str(e)}]")
        return "\n".join(texts)

    def __call__(self, images: Union[str, Path, List[Union[str, Path]]]) -> str:
        if isinstance(images, (str, Path)):
            return self.process_image(images)
        elif isinstance(images, list):
            return self.process_images(images)
        else:
            raise TypeError("Images must be a path or list of paths")


class ModelErrorTracker:
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self.errors: Dict[str, ModelErrorInfo] = {}
        self._load()

    def _load(self):
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for model_name, info in data.items():
                        self.errors[model_name] = ModelErrorInfo(**info)
            except Exception as e:
                print(f"Warning: Could not load error config: {e}")

    def _save(self):
        try:
            data = {name: asdict(info) for name, info in self.errors.items()}
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Warning: Could not save error config: {e}")

    def log_error(self, model_name: str):
        if model_name not in self.errors:
            self.errors[model_name] = ModelErrorInfo(model_name, 0, "", False)
        self.errors[model_name].error_count += 1
        self.errors[model_name].last_error_date = datetime.now().isoformat()
        if self.errors[model_name].error_count >= GLOBAL_CONFIG["max_errors_before_ignore"]:
            self.errors[model_name].is_faulty = True
        self._save()

    def is_faulty(self, model_name: str) -> bool:
        return self.errors.get(model_name, ModelErrorInfo("", 0, "", False)).is_faulty


class ModelFactory:
    def __init__(self, base_folder: str, error_tracker: ModelErrorTracker):
        self.base_folder = Path(base_folder)
        self.error_tracker = error_tracker
        self._model_cache: Dict[str, TextToImage] = {}

    def create_model(self, model_name: str) -> Optional[TextToImage]:
        if self.error_tracker.is_faulty(model_name):
            return None
        if model_name in self._model_cache:
            return self._model_cache[model_name]
        model_path = self.base_folder / model_name
        if not model_path.exists():
            return None
        model = TextToImage(model_name, str(model_path), GLOBAL_CONFIG["use_gpu"])
        if not model.initialize():
            self.error_tracker.log_error(model_name)
            return None
        self._model_cache[model_name] = model
        return model


class ImageToTextManager:
    def __init__(self, base_folder: str = r"D:\AIs\Image-to-Text"):
        self.base_folder = Path(base_folder)
        self.error_tracker = ModelErrorTracker(
            os.path.join(self.base_folder, GLOBAL_CONFIG["config_file"])
        )
        self.factory = ModelFactory(self.base_folder, self.error_tracker)
        self._model_list: List[ModelInfo] = []
        self._scan_models()

    def _scan_models(self):
        self._model_list = []
        if not self.base_folder.exists():
            return
        for item in self.base_folder.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                total_size = sum(f.stat().st_size for f in item.rglob('*') if f.is_file())
                self._model_list.append(ModelInfo(
                    name=item.name,
                    size_mb=round(total_size / (1024 * 1024), 2),
                    path=str(item)
                ))
        self._model_list.sort(key=lambda x: x.name)

    def get_available_models(self) -> List[ModelInfo]:
        return self._model_list.copy()

    def get_model(self, model_name: str) -> Optional[TextToImage]:
        return self.factory.create_model(model_name)

    def apply_all(self, images: Union[str, Path, List[Union[str, Path]]]) -> Dict[str, Any]:
        results = []
        statistics = {
            "successful_models": [],
            "failed_models": [],
            "ignored_models": [],
            "total_models": len(self._model_list),
            "timestamp": datetime.now().isoformat()
        }
        image_list = [images] if isinstance(images, (str, Path)) else images
        for model_info in self._model_list:
            model_name = model_info.name
            if self.error_tracker.is_faulty(model_name):
                statistics["ignored_models"].append({
                    "model_name": model_name,
                    "reason": f"Exceeded {GLOBAL_CONFIG['max_errors_before_ignore']} errors"
                })
                continue
            start_time = time.time()
            result = ModelResult(
                model_name=model_name,
                text="",
                success=False,
                time_taken=0,
                datetime_completed=datetime.now().isoformat()
            )
            try:
                model = self.get_model(model_name)
                if model is None:
                    result.error_message = "Failed to initialize model"
                    self.error_tracker.log_error(model_name)
                else:
                    if len(image_list) == 1:
                        result.text = model(image_list[0])
                    else:
                        result.text = model(image_list)
                    result.success = True
                    statistics["successful_models"].append(model_name)
            except Exception as e:
                result.success = False
                result.error_message = str(e)
                self.error_tracker.log_error(model_name)
                statistics["failed_models"].append({
                    "model_name": model_name,
                    "error": str(e)
                })
            result.time_taken = round(time.time() - start_time, 3)
            result.datetime_completed = datetime.now().isoformat()
            results.append(asdict(result))
        statistics["results"] = results
        return statistics

    def run_test(
            self,
            test_image: str,
            reference_text_file: str
    ) -> Dict[str, Any]:
        try:
            with open(reference_text_file, 'r', encoding='utf-8') as f:
                reference_text = f.read().strip()
        except Exception as e:
            print(f"Error loading reference text: {e}")
            return {"error": str(e)}
        print(f"\n{'=' * 80}")
        print(f"TEST RESULTS")
        print(f"Reference: {reference_text_file}")
        print(f"Image: {test_image}")
        print(f"{'=' * 80}\n")
        test_results = []
        for model_info in self._model_list:
            model_name = model_info.name
            if self.error_tracker.is_faulty(model_name):
                print(f"[{model_name}] SKIPPED (Faulty - >10 errors)")
                continue
            try:
                start_time = time.time()
                model = self.get_model(model_name)
                if model is None:
                    print(f"[{model_name}] FAILED to initialize")
                    test_results.append({
                        "model_name": model_name,
                        "levenshtein_ratio": 0.0,
                        "success": False,
                        "time_taken": 0,
                        "error": "Initialization failed"
                    })
                    continue
                predicted_text = model(test_image)
                time_taken = round(time.time() - start_time, 3)
                ratio = difflib.SequenceMatcher(None, reference_text, predicted_text).ratio()
                status = "SUCCESS" if ratio > 0.5 else "PARTIAL"
                print(f"[{model_name}] Ratio: {ratio:.4f} | {status} | Time: {time_taken}s")
                test_results.append({
                    "model_name": model_name,
                    "levenshtein_ratio": round(ratio, 4),
                    "success": ratio > 0.5,
                    "time_taken": time_taken,
                    "predicted_text": predicted_text[:100] + "..." if len(predicted_text) > 100 else predicted_text
                })
            except Exception as e:
                time_taken = round(time.time() - start_time, 3) if 'start_time' in locals() else 0
                print(f"[{model_name}] FAILED | Error: {str(e)[:100]}")
                self.error_tracker.log_error(model_name)
                test_results.append({
                    "model_name": model_name,
                    "levenshtein_ratio": 0.0,
                    "success": False,
                    "time_taken": time_taken,
                    "error": str(e)[:100]
                })
        print(f"\n{'=' * 80}")
        print("SUMMARY - ALL MODELS")
        print(f"{'=' * 80}")
        successful = sum(1 for r in test_results if r["success"])
        print(f"Total Models Tested: {len(test_results)}")
        print(f"Successful (>50% match): {successful}")
        print(f"Failed: {len(test_results) - successful}")
        test_results.sort(key=lambda x: x["levenshtein_ratio"], reverse=True)
        print(f"\n{'Rank':<5} {'Status':<8} {'Ratio':<8} {'Time':<10} Model")
        print(f"{'-' * 5} {'-' * 8} {'-' * 8} {'-' * 10} {'-' * 70}")
        for i, r in enumerate(test_results, 1):
            status = "✓" if r["success"] else "✗"
            ratio_str = f"{r['levenshtein_ratio']:.4f}"
            time_str = f"{r['time_taken']:.3f}s"
            print(f"{i:<5} {status:<8} {ratio_str:<8} {time_str:<10} {r['model_name']}")
        return {
            "test_results": test_results,
            "reference_text": reference_text,
            "successful_count": successful,
            "total_count": len(test_results)
        }

    def _normalized_levenshtein_similarity(self, s1: str, s2: str) -> float:
        try:
            import Levenshtein
            distance = Levenshtein.distance(s1, s2)
            max_len = max(len(s1), len(s2))
            return 1.0 - (distance / max_len) if max_len > 0 else 1.0
        except ImportError:
            return difflib.SequenceMatcher(None, s1, s2).ratio()


def set_device(use_gpu: bool = False):
    GLOBAL_CONFIG["use_gpu"] = use_gpu
    if use_gpu:
        if torch.cuda.is_available():
            print("GPU mode enabled")
        else:
            print("GPU requested but not available. Falling back to CPU.")
            GLOBAL_CONFIG["use_gpu"] = False
    else:
        print("CPU mode enabled")