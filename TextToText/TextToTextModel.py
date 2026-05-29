import traceback
from pathlib import Path
from typing import Union, Dict, Any, Callable
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, AutoConfig

# Try to import llama_cpp for GGUF support
try:
    from llama_cpp import Llama

    LLAMA_CPP_AVAILABLE = True
except ImportError:
    Llama = None
    LLAMA_CPP_AVAILABLE = False


# =============================================================================
# CUSTOM MODEL REGISTRY
# =============================================================================
# Register custom initialization and parsing logic for specific models here.
# To add a new model, simply append an entry with a unique identifier string.
# =============================================================================

def _init_default_causal(model_path: str, device: str) -> Dict[str, Any]:
    """Standard initialization for most causal LMs."""
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, local_files_only=True, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    ).to(device)
    return {"model": model, "tokenizer": tokenizer}


def _parse_pip_sql_output(text: str) -> str:
    """Extract SQL content between <sql> and </sql> tags."""
    if "<sql>" in text and "</sql>" in text:
        return text.split('<sql>')[1].split('</sql>')[0].strip()
    return text.strip()


def _init_pip_sql(model_path: str, device: str) -> Dict[str, Any]:
    """Custom initialization for pip-sql models."""
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, local_files_only=True, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    ).to(device)
    return {"model": model, "tokenizer": tokenizer}


def _parse_qwen_sql_output(text: str) -> str:
    """Extract SQL from Qwen model output - return text after prompt."""
    sql_keywords = ["SELECT", "INSERT", "UPDATE", "DELETE", "CREATE", "DROP", "ALTER", "WITH"]
    upper_text = text.upper()
    for kw in sql_keywords:
        idx = upper_text.find(kw)
        if idx != -1:
            return text[idx:].strip()
    return text.strip()


def _init_qwen_sql(model_path: str, device: str) -> Dict[str, Any]:
    """Custom initialization for Qwen Text-to-SQL models (standard PyTorch)."""
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, local_files_only=True, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    ).to(device)
    return {"model": model, "tokenizer": tokenizer}


def _init_qwen_gguf(model_path: str, device: str) -> Dict[str, Any]:
    """Custom initialization for Qwen GGUF models using llama-cpp-python."""
    if not LLAMA_CPP_AVAILABLE:
        raise ImportError("llama-cpp-python is required for GGUF models. Install with: pip install llama-cpp-python")

    # Find the actual .gguf file in the folder
    gguf_files = list(Path(model_path).glob("*.gguf"))
    if not gguf_files:
        raise FileNotFoundError(f"No .gguf file found in {model_path}")

    gguf_path = str(gguf_files[0])
    print(f"   [GGUF] Loading {gguf_path} via llama-cpp-python...")

    # Determine GPU layers: -1 for all on GPU if CUDA available, 0 for CPU
    n_gpu_layers = -1 if device == "cuda" else 0

    # Increase context if needed (default: 4096, max: 262144 for this model)
    n_ctx = 262144  # or 16384, 32768, etc. - but higher = more VRAM

    llm = Llama(
        model_path=gguf_path,
        n_ctx=n_ctx,  # ← Increase this value
        n_gpu_layers=n_gpu_layers,
        verbose=False
    )
    return {"llm": llm, "is_gguf": True}

def _parse_gguf_output(text: str) -> str:
    """Parse GGUF model output - return stripped text."""
    return text.strip()


# Registry mapping identifier strings to their handlers
CUSTOM_MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "pip-sql-1.3b": {
        "description": "PipableAI/pip-sql for SQL generation with tag-based parsing",
        "init_fn": _init_pip_sql,
        "parse_fn": _parse_pip_sql_output,
        "default_max_tokens": 200
    },
    "Qwen-3-4b-Text_to_SQL-GGUF": {
        "description": "Qwen-3 Text-to-SQL model (GGUF format, uses llama-cpp-python)",
        "init_fn": _init_qwen_gguf,
        "parse_fn": _parse_gguf_output,
        "default_max_tokens": 256,
        "use_gguf": True
    },
    "Qwen-2.5-3b-Text_to_SQL": {
        "description": "Qwen-2.5 Text-to-SQL model",
        "init_fn": _init_qwen_sql,
        "parse_fn": _parse_qwen_sql_output,
        "default_max_tokens": 512
    },
}


# =============================================================================

class TextToTextModel:
    def __init__(self, model_path: Path, model_name: str):
        print(f"   [T2T] Initializing {model_name}...")
        self.model_path = model_path
        self.model_name = model_name
        self._pipeline = None
        self._custom_config = None
        self._custom_objects = None
        self._device = self._determine_device()
        self._identify_model()
        print(f"   [T2T] Device: {self._device}")
        if self._custom_config:
            print(f"   [T2T] Custom handler: {self._custom_config['description']}")

    def _identify_model(self):
        """Check model name/path against the custom registry."""
        name_lower = self.model_name.lower()
        path_str = str(self.model_path).lower()
        for key, config in CUSTOM_MODEL_REGISTRY.items():
            if key.lower() in name_lower or key.lower() in path_str:
                self._custom_config = config
                return

    def _determine_device(self) -> str:
        if torch.cuda.is_available():
            return "cuda"
        print(f"⚠ GPU unavailable, using CPU for {self.model_name}")
        return "cpu"

    def _load_pipeline(self):
        if self._pipeline is not None or self._custom_objects is not None:
            return
        print(f"   [Pipeline] Loading for {self.model_name}...")
        try:
            if self._custom_config:
                print("   [Pipeline] Using custom initialization...")
                self._custom_objects = self._custom_config["init_fn"](
                    str(self.model_path), self._device
                )
                print("   [Pipeline] ✓ Custom model & tokenizer ready")
                return

            # Standard causal LM pipeline initialization
            config = AutoConfig.from_pretrained(
                str(self.model_path), local_files_only=True, trust_remote_code=True
            )
            if not hasattr(config, 'architectures') or not config.architectures:
                raise ValueError("Model lacks architecture information")
            arch = config.architectures[0].lower()
            unsupported = ['vision', 'image', 'diffusion', 'llada', 'lumina']
            if any(u in arch for u in unsupported):
                raise ValueError(f"Architecture '{arch}' unsupported for text-generation")

            tokenizer = AutoTokenizer.from_pretrained(
                str(self.model_path), local_files_only=True, trust_remote_code=True
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            model_dtype = torch.float16 if self._device == "cuda" else torch.float32
            model = AutoModelForCausalLM.from_pretrained(
                str(self.model_path), local_files_only=True,
                torch_dtype=model_dtype, low_cpu_mem_usage=True,
                trust_remote_code=True
            ).to(self._device)
            device_arg = 0 if self._device == "cuda" else -1
            self._pipeline = pipeline(
                "text-generation", model=model, tokenizer=tokenizer,
                device=device_arg
            )
            print("   [Pipeline] ✓ Pipeline ready")
        except ImportError as e:
            if "cannot import name" in str(e) or "initialization" in str(e):
                print(f"   ⚠ Transformers compatibility error: {e}")
                raise RuntimeError(f"Model {self.model_name} incompatible with transformers version")
            raise
        except ValueError as e:
            if "not supported" in str(e).lower() or "architecture" in str(e).lower():
                print(f"   ⚠ {e}")
                raise
            raise
        except Exception as e:
            print("   [Pipeline] ❌ Load failed")
            traceback.print_exc()
            raise RuntimeError(f"Failed to load model {self.model_name}: {str(e)}")

    def process(self, prompt: str, max_new_tokens: int = 512) -> str:
        print("   [Process] Processing request...")
        self._load_pipeline()
        try:
            if self._custom_config:
                max_t = self._custom_config.get("default_max_tokens", max_new_tokens)

                # Handle GGUF models via llama-cpp-python
                if self._custom_config.get("use_gguf"):
                    if not LLAMA_CPP_AVAILABLE:
                        raise RuntimeError("llama-cpp-python not available for GGUF model")
                    llm = self._custom_objects["llm"]
                    out = llm(
                        prompt,
                        max_tokens=max_t,
                        temperature=0.2,
                        top_p=0.9,
                        echo=False
                    )
                    generated_text = out["choices"][0]["text"]
                    return self._custom_config["parse_fn"](generated_text)

                # Handle standard custom models
                tokenizer = self._custom_objects["tokenizer"]
                model = self._custom_objects["model"]
                inputs = tokenizer(prompt, return_tensors="pt").to(self._device)
                outputs = model.generate(**inputs, max_new_tokens=max_t, do_sample=False)
                generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                return self._custom_config["parse_fn"](generated_text)

            outputs = self._pipeline(prompt, max_new_tokens=max_new_tokens, do_sample=False)
            if isinstance(outputs, list):
                return outputs[0].get("generated_text", "").strip()
            elif isinstance(outputs, dict):
                return outputs.get("generated_text", "").strip()
            return str(outputs).strip()
        except Exception as e:
            print(f"   [Process] ❌ Processing failed: {str(e)}")
            traceback.print_exc()
            raise