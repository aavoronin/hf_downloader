import traceback
from pathlib import Path
from typing import Union, Dict, Any, Callable
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, AutoConfig

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

# Registry mapping identifier strings to their handlers
CUSTOM_MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "pip-sql-1.3b": {
        "description": "PipableAI/pip-sql for SQL generation with tag-based parsing",
        "init_fn": _init_pip_sql,
        "parse_fn": _parse_pip_sql_output,
        "default_max_tokens": 200
    },
    # Example: Add future models here easily
    # "sqlcoder-7b": {
    #     "description": "Defog/sqlcoder for SQL generation",
    #     "init_fn": _init_default_causal,
    #     "parse_fn": lambda t: t.split("```sql")[-1].split("```")[0].strip() if "```sql" in t else t.strip(),
    #     "default_max_tokens": 512
    # }
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
            if key in name_lower or key in path_str:
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