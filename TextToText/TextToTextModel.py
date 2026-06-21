import traceback
from pathlib import Path
from typing import Union, Dict, Any, Callable, List, Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, AutoConfig

# Optional import for prem-research/premsql library
try:
    from premsql.agents import BaseLineAgent
    from premsql.generators import Text2SQLGeneratorHF
    from premsql.agents.tools import SimpleMatplotlibTool
    from premsql.executors import SQLiteExecutor

    PREMSQL_AVAILABLE = True
except ImportError:
    PREMSQL_AVAILABLE = False
    BaseLineAgent = None
    Text2SQLGeneratorHF = None
    SimpleMatplotlibTool = None
    SQLiteExecutor = None

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
    sql_keywords = ["SELECT", "INSERT", "UPDATE", "DELETE", "CREATE",
                    "DROP", "ALTER", "WITH"]
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
        raise ImportError("llama-cpp-python is required for GGUF models. "
                          "Install with: pip install llama-cpp-python")
    # Find the actual .gguf file in the folder
    gguf_files = list(Path(model_path).glob("*.gguf"))
    if not gguf_files:
        raise FileNotFoundError(f"No .gguf file found in {model_path}")
    gguf_path = str(gguf_files[0])
    print(f"   [GGUF] Loading {gguf_path} via llama-cpp-python...")
    # Determine GPU layers: -1 for all on GPU if CUDA available, 0 for CPU
    n_gpu_layers = -1 if device == "cuda" else 0
    # Increase context if needed (default: 4096, max: 262144 for this model)
    n_ctx = 262144
    llm = Llama(
        model_path=gguf_path,
        n_ctx=n_ctx,
        n_gpu_layers=n_gpu_layers,
        verbose=False
    )
    return {"llm": llm, "is_gguf": True}


def _parse_gguf_output(text: str) -> str:
    """Parse GGUF model output - return stripped text."""
    return text.strip()


def _init_prem_sql(model_path: str, device: str) -> Dict[str, Any]:
    """Custom initialization for prem-research/prem-1B-SQL using premsql."""
    if not PREMSQL_AVAILABLE:
        raise ImportError("premsql library is required for prem-1B-SQL. "
                          "Install with: pip install premsql")
    # Initialize the Text2SQL generator
    text2sql_model = Text2SQLGeneratorHF(
        model_or_name_or_path=model_path,
        experiment_name="text2text_runner",
        device=device,
        type="inference"
    )
    # Initialize analyzer/plotter model (using same path for simplicity)
    analyser_model = Text2SQLGeneratorHF(
        model_or_name_or_path=model_path,
        experiment_name="text2text_runner",
        device=device,
        type="inference"
    )
    # Create agent with SQLite executor
    agent = BaseLineAgent(
        session_name="text2text_session",
        db_connection_uri="sqlite:///:memory:",
        specialized_model1=text2sql_model,
        specialized_model2=analyser_model,
        plot_tool=SimpleMatplotlibTool(),
        executor=SQLiteExecutor()
    )
    return {"agent": agent, "is_prem": True}


def _parse_prem_sql_output(text: str) -> str:
    """Parse prem-sql output - agent returns dataframe, extract SQL if present."""
    # prem agent may return dataframe or dict; try to extract SQL string
    if hasattr(text, 'show_dataframe'):
        return str(text)
    return str(text).strip()


def _parse_antelope_output(text: str) -> str:
    """Parse Antelope model output - extract SQL after ### SQL: prefix."""
    if "### SQL:" in text:
        # Get content after ### SQL: and take first line
        sql_part = text.split("### SQL:")[-1].strip()
        return sql_part.split('\n')[0].strip()
    return text.strip()


def _init_antelope_sql(model_path: str, device: str) -> Dict[str, Any]:
    """Custom initialization for AuricErgeson/Antelope-textTosql."""
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
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True
    )
    if device == "cuda" and not hasattr(model, 'device'):
        model = model.to(device)
    return {"model": model, "tokenizer": tokenizer}


def _parse_gemma_sql_output(text: str) -> str:
    """Parse Gemma-style SQL output - extract model answer after turn tags."""
    # Split on end_of_turn and take first two parts
    parts = text.split('<end_of_turn>')[:2]
    ans = ''.join(parts)
    # Extract model answer after "model" keyword
    if "model" in ans:
        model_answer = ans.split("model")[1].strip()
        return model_answer
    return text.strip()


def _init_gemma_sql(model_path: str, device: str) -> Dict[str, Any]:
    """Custom initialization for suriya7/Gemma2B-Finetuned-Sql-Generator."""
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


def _init_gemma_gguf(model_path: str, device: str) -> Dict[str, Any]:
    """Custom initialization for Gemma GGUF models using llama-cpp-python."""
    if not LLAMA_CPP_AVAILABLE:
        raise ImportError("llama-cpp-python is required for GGUF models. "
                          "Install with: pip install llama-cpp-python")
    # Find the actual .gguf file in the folder
    gguf_files = list(Path(model_path).glob("*.gguf"))
    if not gguf_files:
        raise FileNotFoundError(f"No .gguf file found in {model_path}")
    gguf_path = str(gguf_files[0])
    print(f"   [GGUF] Loading {gguf_path} via llama-cpp-python...")
    # Determine GPU layers: -1 for all on GPU if CUDA available, 0 for CPU
    n_gpu_layers = -1 if device == "cuda" else 0
    # Use reasonable context for Gemma models
    n_ctx = 4096
    llm = Llama(
        model_path=gguf_path,
        n_ctx=n_ctx,
        n_gpu_layers=n_gpu_layers,
        verbose=False
    )
    return {"llm": llm, "is_gguf": True}


def _init_bagel_gguf(model_path: str, device: str) -> Dict[str, Any]:
    """Custom initialization for calcuis/bagel-gguf using llama-cpp-python."""
    if not LLAMA_CPP_AVAILABLE:
        raise ImportError("llama-cpp-python is required for GGUF models. "
                          "Install with: pip install llama-cpp-python")
    # Find the actual .gguf file in the folder
    gguf_files = list(Path(model_path).glob("*.gguf"))
    if not gguf_files:
        raise FileNotFoundError(f"No .gguf file found in {model_path}")
    gguf_path = str(gguf_files[0])
    print(f"   [GGUF] Loading {gguf_path} via llama-cpp-python...")
    # Determine GPU layers: -1 for all on GPU if CUDA available, 0 for CPU
    n_gpu_layers = -1 if device == "cuda" else 0
    # Use large context for Bagel models
    n_ctx = 8192
    llm = Llama(
        model_path=gguf_path,
        n_ctx=n_ctx,
        n_gpu_layers=n_gpu_layers,
        verbose=False
    )
    return {"llm": llm, "is_gguf": True}


def _init_gemma4_gguf(model_path: str, device: str) -> Dict[str, Any]:
    """Custom initialization for Bhuvneesh/gemma-4-E4B-it-Q8_0-GGUF."""
    if not LLAMA_CPP_AVAILABLE:
        raise ImportError("llama-cpp-python is required for GGUF models. "
                          "Install with: pip install llama-cpp-python")
    # Find the actual .gguf file in the folder
    gguf_files = list(Path(model_path).glob("*.gguf"))
    if not gguf_files:
        raise FileNotFoundError(f"No .gguf file found in {model_path}")
    gguf_path = str(gguf_files[0])
    print(f"   [GGUF] Loading {gguf_path} via llama-cpp-python...")
    # Determine GPU layers: -1 for all on GPU if CUDA available, 0 for CPU
    n_gpu_layers = -1 if device == "cuda" else 0
    # Context window size. 131072 matches the model's training capacity.
    n_ctx = 131072
    llm = Llama(
        model_path=gguf_path,
        n_ctx=n_ctx,
        n_gpu_layers=n_gpu_layers,
        verbose=False
    )
    return {"llm": llm, "is_gguf": True}


def _parse_gemma4_output(text: str) -> str:
    """Parse Gemma-4 model output, extracting code blocks if present."""
    if "```sql" in text:
        return text.split("```sql")[1].split("```")[0].strip()
    if "```" in text:
        return text.split("```")[1].split("```")[0].strip()
    return text.strip()


# =============================================================================
# QWEN2.5-CODER CUSTOM HANDLERS
# =============================================================================
def _init_qwen25_coder(model_path: str, device: str,
                       context_length: int = 32768) -> Dict[str, Any]:
    """
    Custom initialization for Qwen2.5-Coder models.
    Handles chat template formatting and proper token management.
    Args:
        model_path: Path to model files
        device: Target device ('cuda' or 'cpu')
        context_length: Max context length (32768 for small models, 131072 for 7B/14B)
    Returns:
        Dict with model, tokenizer, and config
    """
    print(f"   [Qwen2.5-Coder] Loading with context_length={context_length}...")
    # Load tokenizer with chat template support
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True,
        model_max_length=context_length
    )
    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Determine dtype based on device
    dtype = torch.float16 if device == "cuda" else torch.float32
    # Load model with appropriate settings for Qwen2.5 architecture
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        # Enable YaRN for extended context on 7B/14B models if needed
        rope_scaling={"factor": 4.0, "original_max_position_embeddings": 32768, "type": "yarn"}
        if context_length > 32768 else None
    ).to(device)
    return {
        "model": model,
        "tokenizer": tokenizer,
        "context_length": context_length,
        "is_qwen25_coder": True
    }


def _parse_qwen25_coder_output(text: str, original_prompt: str = "") -> str:
    """
    Parse Qwen2.5-Coder model output.
    Extracts only the assistant's response from the chat-formatted output.
    Args:
        text: Full generated text from model
        original_prompt: Original user prompt for reference
    Returns:
        Cleaned response text
    """
    # Qwen2.5-Coder uses chat template, output may contain:
    # - System message prefix
    # - User message prefix
    # - Assistant response (what we want)
    # Common Qwen chat markers
    assistant_markers = ["<|im_start|>assistant", "assistant\n", "Assistant:"]
    # Try to find assistant response start
    for marker in assistant_markers:
        if marker in text:
            # Extract everything after the marker
            parts = text.split(marker, 1)
            if len(parts) > 1:
                response = parts[1].strip()
                # Remove trailing im_end token if present
                if "<|im_end|> " in response:
                    response = response.split("<|im_end|>")[0].strip()
                return response
    # Fallback: return text as-is, stripped
    return text.strip()


def _process_qwen25_coder_prompt(prompt: str, tokenizer,
                                 system_message: Optional[str] = None) -> str:
    """
    Format prompt using Qwen2.5-Coder chat template.
    Args:
        prompt: User's instruction/prompt
        tokenizer: Qwen tokenizer with apply_chat_template support
        system_message: Optional system message (default: Qwen standard)
    Returns:
        Formatted text ready for tokenization
    """
    if system_message is None:
        system_message = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt}
    ]
    # Apply chat template with generation prompt marker
    formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    return formatted


# =============================================================================
# Registry mapping identifier strings to their handlers
# =============================================================================
CUSTOM_MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "pip-sql-1.3b": {
        "description": "PipableAI/pip-sql for SQL generation with tag-based parsing",
        "init_fn": _init_pip_sql,
        "parse_fn": _parse_pip_sql_output,
        "default_max_tokens": 1024 * 4
    },
    "pip-sql-1.3b-GGUF": {
        "description": "QuantFactory/pip-sql-1.3b-GGUF with tag-based parsing",
        "init_fn": _init_pip_sql,
        "parse_fn": _parse_pip_sql_output,
        "default_max_tokens": 1024 * 4
    },
    "prem-1B-SQL": {
        "description": "prem-research/prem-1B-SQL using premsql agent framework",
        "init_fn": _init_prem_sql,
        "parse_fn": _parse_prem_sql_output,
        "default_max_tokens": 1024 * 4,
        "use_prem": True
    },
    "Qwen-3-4b-Text_to_SQL-GGUF": {
        "description": "Qwen-3 Text-to-SQL model (GGUF format, uses llama-cpp-python)",
        "init_fn": _init_qwen_gguf,
        "parse_fn": _parse_gguf_output,
        "default_max_tokens": 1024 * 4,
        "use_gguf": True
    },
    "Qwen-2.5-3b-Text_to_SQL": {
        "description": "Qwen-2.5 Text-to-SQL model",
        "init_fn": _init_qwen_sql,
        "parse_fn": _parse_qwen_sql_output,
        "default_max_tokens": 1024 * 4
    },
    "Antelope-textTosql": {
        "description": "AuricErgeson/Antelope-textTosql with custom prompt format",
        "init_fn": _init_antelope_sql,
        "parse_fn": _parse_antelope_output,
        "default_max_tokens": 1024 * 4
    },
    "Gemma2B-Finetuned-Sql-Generator": {
        "description": "suriya7/Gemma2B-Finetuned-Sql-Generator with Gemma chat template",
        "init_fn": _init_gemma_sql,
        "parse_fn": _parse_gemma_sql_output,
        "default_max_tokens": 1024 * 4
    },
    "Gemma2B-Finetuned-Sql-Generator-GGUF": {
        "description": "Gemma2B-Finetuned-Sql-Generator (GGUF format, uses llama-cpp-python)",
        "init_fn": _init_gemma_gguf,
        "parse_fn": _parse_gemma_sql_output,
        "default_max_tokens": 1024 * 4,
        "use_gguf": True
    },
    "bagel-gguf": {
        "description": "calcuis/bagel-gguf model (GGUF format, uses llama-cpp-python)",
        "init_fn": _init_bagel_gguf,
        "parse_fn": _parse_gguf_output,
        "default_max_tokens": 1024 * 4,
        "use_gguf": True
    },
    "gemma-4-E4B-it-Q8_0-GGUF": {
        "description": "Bhuvneesh/gemma-4-E4B-it-Q8_0-GGUF (Instruction-tuned, uses chat completion)",
        "init_fn": _init_gemma4_gguf,
        "parse_fn": _parse_gemma4_output,
        "default_max_tokens": 131072,
        "max_input_tokens": 32768,
        "use_gguf": True,
        "use_chat_completion": True
    },
    # =====================================================================
    # QWEN2.5-CODER MODELS (0.5B, 1.5B, 3B - 32K context)
    # =====================================================================
    "Qwen2.5-Coder-0.5B-Instruct": {
        "description": "Qwen/Qwen2.5-Coder-0.5B-Instruct - Code-specific LLM with chat template",
        "init_fn": lambda path, dev: _init_qwen25_coder(path, dev, context_length=32768),
        "parse_fn": _parse_qwen25_coder_output,
        "default_max_tokens": 4096,
        "max_input_tokens": 32768,
        "is_qwen25_coder": True,
        "system_message": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
    },
    "Qwen2.5-Coder-1.5B-Instruct": {
        "description": "Qwen/Qwen2.5-Coder-1.5B-Instruct - Code-specific LLM with chat template",
        "init_fn": lambda path, dev: _init_qwen25_coder(path, dev, context_length=32768),
        "parse_fn": _parse_qwen25_coder_output,
        "default_max_tokens": 4096,
        "max_input_tokens": 32768,
        "is_qwen25_coder": True,
        "system_message": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
    },
    "Qwen2.5-Coder-3B-Instruct": {
        "description": "Qwen/Qwen2.5-Coder-3B-Instruct - Code-specific LLM with chat template",
        "init_fn": lambda path, dev: _init_qwen25_coder(path, dev, context_length=32768),
        "parse_fn": _parse_qwen25_coder_output,
        "default_max_tokens": 4096,
        "max_input_tokens": 32768,
        "is_qwen25_coder": True,
        "system_message": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
    },
    # =====================================================================
    # QWEN2.5-CODER MODELS (7B, 14B - 128K context with YaRN support)
    # =====================================================================
    "Qwen2.5-Coder-7B-Instruct": {
        "description": "Qwen/Qwen2.5-Coder-7B-Instruct - Code LLM with 128K context (YaRN)",
        "init_fn": lambda path, dev: _init_qwen25_coder(path, dev, context_length=131072),
        "parse_fn": _parse_qwen25_coder_output,
        "default_max_tokens": 8192,
        "max_input_tokens": 131072,
        "is_qwen25_coder": True,
        "system_message": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
        "use_yarn": True
    },
    "Qwen2.5-Coder-14B-Instruct": {
        "description": "Qwen/Qwen2.5-Coder-14B-Instruct - Code LLM with 128K context (YaRN)",
        "init_fn": lambda path, dev: _init_qwen25_coder(path, dev, context_length=131072),
        "parse_fn": _parse_qwen25_coder_output,
        "default_max_tokens": 8192,
        "max_input_tokens": 131072,
        "is_qwen25_coder": True,
        "system_message": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
        "use_yarn": True
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

#    def _determine_device(self) -> str:
#        if torch.cuda.is_available():
#            return "cuda"
#        print(f"⚠ GPU unavailable, using CPU for {self.model_name}")
#        return "cpu"

    def _determine_device(self) -> str:
        if not torch.cuda.is_available():
            print(f"⚠ GPU unavailable, using CPU for {self.model_name}")
            return "cpu"

        # Find and select NVIDIA RTX 5070 Ti specifically
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            if "RTX 5070 Ti" in gpu_name or "RTX" in gpu_name:
                print(f"✓ Found NVIDIA GPU: {gpu_name}")
                torch.cuda.set_device(i)
                return f"cuda:{i}"

        # Fallback to first CUDA device if RTX not found
        print(f"⚠ RTX GPU not found, using default CUDA device")
        return "cuda"

    def _load_pipeline(self):
        if self._pipeline is not None or self._custom_objects is not None:
            return

        # ==========================================
        # RTX 5070 Ti GPU Selection Logic
        # ==========================================
        # CHANGE THIS: Set to 0 if 5070 Ti is the first card in `nvidia-smi`,
        # or 1 if it is the second card.
        TARGET_GPU_INDEX = 1

        # Determine the specific device string for PyTorch (e.g., "cuda:1")
        target_device = f"cuda:{TARGET_GPU_INDEX}" if self._device == "cuda" else self._device
        device_arg = TARGET_GPU_INDEX if self._device == "cuda" else -1

        if self._device == "cuda":
            # Force PyTorch to set this GPU as the default for any implicit allocations
            torch.cuda.set_device(TARGET_GPU_INDEX)
        # ==========================================

        print(f"   [Pipeline] Loading for {self.model_name} on {target_device}...")
        try:
            if self._custom_config:
                print("   [Pipeline] Using custom initialization...")
                # Note: We pass self._device ("cuda") here because custom loaders
                # (like llama-cpp) handle specific GPU indexing internally via tensor_split.
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

            # Load the model explicitly onto the targeted GPU (e.g., "cuda:1")
            model = AutoModelForCausalLM.from_pretrained(
                str(self.model_path), local_files_only=True,
                torch_dtype=model_dtype, low_cpu_mem_usage=True,
                trust_remote_code=True
            ).to(target_device)

            # Pass the specific integer index to the pipeline
            self._pipeline = pipeline(
                "text-generation", model=model, tokenizer=tokenizer,
                device=device_arg
            )
            print(f"   [Pipeline] ✓ Pipeline ready on {target_device}")

        except ImportError as e:
            if "cannot import name" in str(e) or "initialization" in str(e):
                print(f"   ⚠ Transformers compatibility error: {e}")
                raise RuntimeError(f"Model {self.model_name} incompatible with "
                                   f"transformers version")
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

    def _calculate_available_tokens(self, prompt: str, max_new_tokens: int) -> tuple:
        """
        Calculate token budget respecting model context limits.
        Returns:
            tuple: (adjusted_max_new_tokens, prompt_tokens_count)
        """
        if not self._custom_config or not self._custom_objects:
            return max_new_tokens, 0
        tokenizer = self._custom_objects.get("tokenizer")
        if not tokenizer:
            return max_new_tokens, 0

        context_length = self._custom_config.get("max_input_tokens", 32768)
        # Count prompt tokens
        prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
        prompt_token_count = len(prompt_tokens)
        # Reserve space for response
        reserved = self._custom_config.get("default_max_tokens", max_new_tokens)
        available = context_length - prompt_token_count - 100  # safety margin
        # Use minimum of requested, reserved, or available
        adjusted_max = min(max_new_tokens, reserved, max(available, 256))
        return adjusted_max, prompt_token_count

    def process(self, prompt: str, max_new_tokens: int = 1024 * 4) -> str:
        print("   [Process] Processing request...")
        self._load_pipeline()
        try:
            if self._custom_config:
                # Handle Qwen2.5-Coder models with chat template
                if self._custom_config.get("is_qwen25_coder"):
                    tokenizer = self._custom_objects["tokenizer"]
                    model = self._custom_objects["model"]
                    # Format prompt with chat template
                    system_msg = self._custom_config.get(
                        "system_message",
                        "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
                    )
                    formatted_prompt = _process_qwen25_coder_prompt(
                        prompt, tokenizer, system_msg
                    )
                    # Calculate token budget
                    adjusted_max, prompt_len = self._calculate_available_tokens(
                        formatted_prompt, max_new_tokens
                    )
                    print(f"   [Process] Prompt tokens: {prompt_len}, Max new: {adjusted_max}")
                    # Tokenize and move to device
                    inputs = tokenizer(
                        formatted_prompt,
                        return_tensors="pt",
                        truncation=True,
                        max_length=prompt_len + adjusted_max
                    ).to(self._device)
                    # Generate with proper settings for Qwen2.5
                    # Explicitly unset sampling parameters to avoid transformers warnings
                    with torch.no_grad():
                        generated_ids = model.generate(
                            **inputs,
                            max_new_tokens=adjusted_max,
                            do_sample=False,
                            pad_token_id=tokenizer.pad_token_id,
                            eos_token_id=tokenizer.eos_token_id,
                            top_k=None,
                            top_p=None,
                            temperature=None
                        )
                    # Extract only the newly generated tokens (Qwen pattern)
                    input_len = inputs.input_ids.shape[1]
                    generated_ids = generated_ids[:, input_len:]
                    # Decode and parse
                    raw_text = tokenizer.batch_decode(
                        generated_ids, skip_special_tokens=True
                    )[0]
                    return _parse_qwen25_coder_output(raw_text, prompt)

                # Handle prem-sql models via premsql agent
                if self._custom_config.get("use_prem"):
                    if not PREMSQL_AVAILABLE:
                        raise RuntimeError("premsql not available for prem model")
                    agent = self._custom_objects["agent"]
                    # prem agent expects query string, returns response object
                    response = agent(prompt)
                    if hasattr(response, 'show_dataframe'):
                        return str(response.show_dataframe())
                    return str(response).strip()

                # Handle GGUF models via llama-cpp-python
                if self._custom_config.get("use_gguf"):
                    if not LLAMA_CPP_AVAILABLE:
                        raise RuntimeError("llama-cpp-python not available for GGUF model")
                    llm = self._custom_objects["llm"]

                    if self._custom_config.get("use_chat_completion"):
                        messages = [{"role": "user", "content": prompt}]
                        response = llm.create_chat_completion(
                            messages=messages,
                            max_tokens=max_new_tokens,
                            temperature=0.7,
                            top_p=0.9,
                            stream=False
                        )
                        generated_text = response["choices"][0]["message"]["content"]
                    else:
                        out = llm(
                            prompt,
                            max_tokens=max_new_tokens,
                            temperature=0.2,
                            top_p=0.9,
                            echo=False
                        )
                        generated_text = out["choices"][0]["text"]

                    return self._custom_config["parse_fn"](generated_text)

                # Handle standard custom models
                tokenizer = self._custom_objects["tokenizer"]
                model = self._custom_objects["model"]
                # Calculate token budget for standard custom models
                adjusted_max, _ = self._calculate_available_tokens(prompt, max_new_tokens)
                inputs = tokenizer(prompt, return_tensors="pt").to(self._device)
                # Explicitly unset sampling parameters to avoid transformers warnings
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=adjusted_max,
                    do_sample=False,
                    top_k=None,
                    top_p=None,
                    temperature=None
                )
                generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                return self._custom_config["parse_fn"](generated_text)

            # Standard pipeline processing
            # Explicitly unset sampling parameters to avoid transformers warnings
            outputs = self._pipeline(
                prompt,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                top_k=None,
                top_p=None,
                temperature=None
            )
            if isinstance(outputs, list):
                return outputs[0].get("generated_text", "").strip()
            elif isinstance(outputs, dict):
                return outputs.get("generated_text", "").strip()
            return str(outputs).strip()
        except Exception as e:
            print(f"   [Process] ❌ Processing failed: {str(e)}")
            traceback.print_exc()
            raise