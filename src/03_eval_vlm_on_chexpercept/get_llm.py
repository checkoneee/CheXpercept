"""
LLM Setup Module for CheXpercept Evaluation

This module provides functions to initialize various LLM clients:
- OpenAI API (GPT-4o, GPT-4V, etc.)
- Anthropic API (Claude)
- Google Gemini API
- Open-source models via vLLM
"""

import os
from typing import Tuple, Optional, Any
from openai import OpenAI, AzureOpenAI
from google import genai
import yaml

from model_configs import get_model_config, list_models, OPENSOURCE_MODEL_CONFIGS, get_safe_tensor_parallel_size

api_keys = yaml.load(open('api_info/api_keys.yaml', 'r'), Loader=yaml.SafeLoader)

# Set environment variables for Google APIs
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = api_keys.get('gemini', {}).get('credentials_path')
#os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "True"


# ---------------------------------------------------------------------------
# RadVLM (transformers backend) helpers
# ---------------------------------------------------------------------------

def get_radvlm_model(model_path: str):
    """Load RadVLM weights from *model_path* (local PhysioNet download).

    Requirements:
        pip install transformers torch torchvision

    Download weights from:
        https://physionet.org/content/radvlm-model/1.0.0/
    """
    import torch
    from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration

    print(f"Loading RadVLM from: {model_path}")

    hf_model = LlavaOnevisionForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        #low_cpu_mem_usage=True,
    ).to("cuda")

    processor = AutoProcessor.from_pretrained(model_path)

    print("RadVLM loaded successfully.")

    return hf_model, processor

def get_hulu_med_model(config: dict):
    """Load Hulu-Med-32B.
    """

    from transformers import AutoModelForCausalLM, AutoProcessor
    import torch

    print(f"Loading Hulu-Med from: {config.get('model_name')}")

    # Load model and processor
    model = AutoModelForCausalLM.from_pretrained(
        config.get('model_name'),
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",
    )

    processor = AutoProcessor.from_pretrained(
        config.get('model_name'),
        trust_remote_code=True
    )

    processor.tokenizer.model_max_length = 32768

    return model, processor

def get_opensource_llm(model: str, tensor_parallel_size: int = None, gpu_memory_utilization: float = 0.85):
    """
    Initialize open-source VLM.

    Routes to the appropriate backend based on model config:
    - backend == "transformers"  →  HuggingFace transformers (e.g. RadVLM)
    - default                    →  vLLM

    Args:
        model: Model alias defined in model_configs.py
               (e.g., "qwen3.5-moe", "glm-4v", "medgemma", "radvlm")
        tensor_parallel_size: Number of GPUs (vLLM only; ignored for transformers backend)
        gpu_memory_utilization: GPU memory utilization (vLLM only)

    Returns:
        Tuple of (model_instance, sampling_params)
    """
    config = get_model_config(model)

    # ----- transformers backend (e.g. RadVLM) -----
    if config.get("backend") == "transformers":
        model_path = config.get("model_path")
        if not model_path:
            raise ValueError(
                f"Model '{model}' uses the transformers backend but 'model_path' is not set "
                f"in model_configs.py. Please set it to the local weights directory."
            )

        return get_radvlm_model(model_path)

    if model == "hulu-med":
        return get_hulu_med_model(config)

    # ----- vLLM backend (default) -----
    from vllm import LLM, SamplingParams  # lazy import: only required for vLLM-backed models

    # Auto-detect number of GPUs from CUDA_VISIBLE_DEVICES
    if tensor_parallel_size is None:
        cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '0')
        if cuda_visible_devices:
            num_gpus = len([x.strip() for x in cuda_visible_devices.split(',') if x.strip()])
            tensor_parallel_size = num_gpus
            print(f"Auto-detected {num_gpus} GPU(s) from CUDA_VISIBLE_DEVICES: {cuda_visible_devices}")
        else:
            tensor_parallel_size = 1
            print("CUDA_VISIBLE_DEVICES not set, using 1 GPU")

    # Adjust tensor_parallel_size to satisfy the model's attention-head divisibility constraint
    tensor_parallel_size = get_safe_tensor_parallel_size(model, tensor_parallel_size)
    
    # Build LLM kwargs
    llm_kwargs = {
        "model": config["model_name"],
        "tensor_parallel_size": tensor_parallel_size,
        "gpu_memory_utilization": config.get("gpu_utilization", 0.85),
    }

    # Add optional vLLM parameters from config
    for key in ("max_model_len", "limit_mm_per_prompt", "trust_remote_code",
                "enforce_eager", "max_num_seqs", "dtype"):
        if key in config:
            llm_kwargs[key] = config[key]

    # Allow arbitrary extra kwargs declared in config
    for key, val in config.get("extra_vllm_kwargs", {}).items():
        llm_kwargs[key] = val
    
    llm = LLM(**llm_kwargs)
    
    # Default sampling parameters
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=8192,
        top_p=1.0,
    )
    
    return llm, sampling_params


def get_openai_client(model: str = "gpt-4o", api_key: Optional[str] = None, base_url: Optional[str] = None) -> Tuple[OpenAI, str]:
    """
    Initialize OpenAI API client
    
    Args:
        model: Model name (e.g., "gpt-4o", "gpt-4-turbo", "gpt-4-vision-preview")
        api_key: OpenAI API key (if None, uses environment variable)
        base_url: Custom base URL for OpenAI-compatible APIs
    
    Returns:
        Tuple of (OpenAI client, model name)
    """
    if api_key is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if api_key is None:
            raise ValueError("OpenAI API key not provided and OPENAI_API_KEY environment variable not set")
    
    client = OpenAI(api_key=api_key, base_url=base_url)
    
    return client, model


def get_azure_openai_client(deployment_name: str = "gpt-4o") -> Tuple[AzureOpenAI, str]:
    """
    Initialize Azure OpenAI API client
    
    Args:
        deployment_name: Azure deployment name (e.g., "gpt-4o", "gpt-5.2")
    
    Returns:
        Tuple of (AzureOpenAI client, deployment name)
    """

    if deployment_name == "gpt-5.4-nano" or deployment_name == "gpt-5.4":
        api_version = api_keys.get('azure', {}).get('api_version')
        subscription_key = api_keys.get('azure', {}).get('api_key')
        endpoint = api_keys.get('azure', {}).get('endpoint')
    else:
        raise ValueError(f"Invalid deployment name: {deployment_name}")

    client = AzureOpenAI(
        api_version=api_version,
        azure_endpoint=endpoint,
        api_key=subscription_key,
    )
    
    return client, deployment_name


def get_gemini_client(model: str = "gemini-2.0-flash-exp") -> Tuple[genai.Client, str]:
    """
    Initialize Google Gemini API client
    
    Args:
        model: Model name (e.g., "gemini-2.0-flash-exp", "gemini-1.5-pro")
    
    Returns:
        Tuple of (Gemini client, model name)
    """
    client = genai.Client(
        vertexai=True,
        location='global',
        project=api_keys.get('gemini', {}).get('project'),
    )
    
    return client, model

class DummyLLM:
    """Dummy LLM for debugging without loading actual model"""
    def __init__(self, model_name: str):
        self.model_name = model_name
        print(f"[DEBUG] DummyLLM initialized for {model_name}")
    
    def chat(self, messages, sampling_params=None):
        """Return dummy response"""
        print(f"[DEBUG] DummyLLM.chat called with {len(messages)} messages")
        
        # Create dummy output structure similar to vLLM
        class DummyOutput:
            def __init__(self):
                self.text = "1"  # Dummy answer
        
        class DummyResult:
            def __init__(self):
                self.outputs = [DummyOutput()]
        
        return [DummyResult()]

class DummySamplingParams:
    """Dummy sampling params for debugging"""
    def __init__(self):
        self.temperature = 0.0
        self.max_tokens = 1024
        self.top_p = 1.0
        print("[DEBUG] DummySamplingParams initialized")

def get_dummy_llm(model: str) -> Tuple[DummyLLM, DummySamplingParams]:
    """
    Get dummy LLM for debugging (no actual model loading)
    
    Args:
        model: Model name (for display only)
    
    Returns:
        Tuple of (DummyLLM, DummySamplingParams)
    """
    print(f"[DEBUG] Using DummyLLM instead of loading {model}")
    dummy_llm = DummyLLM(model)
    dummy_sampling_params = DummySamplingParams()
    return dummy_llm, dummy_sampling_params

def get_llm_client(
    provider: str,
    model: str,
    debug: bool = False,
    **kwargs
) -> Tuple[Any, str, str]:
    """
    Universal LLM client getter
    
    Args:
        provider: Provider name ("openai", "azure", "gemini", "opensource", "dummy")
        model: Model name
        debug: If True, use dummy LLM for debugging (default: False)
        **kwargs: Additional arguments for specific providers
    
    Returns:
        Tuple of (client, model_name, provider_type)
    
    Example:
        >>> client, model, provider = get_llm_client("openai", "gpt-4o")
        >>> client, model, provider = get_llm_client("opensource", "qwen2-vl")
        >>> client, model, provider = get_llm_client("opensource", "qwen2-vl", debug=True)  # Use dummy
    """
    provider = provider.lower()
    
    # Debug mode: use dummy LLM
    if debug or provider == "dummy":
        print("[DEBUG MODE] Using DummyLLM for debugging")
        client, sampling_params = get_dummy_llm(model)
        return (client, sampling_params), model, "opensource"  # Pretend to be opensource
    
    if provider == "openai":
        client, model = get_openai_client(model, **kwargs)
        return (client, model), model, "openai"
    
    elif provider == "azure":
        client, model = get_azure_openai_client(model)
        return (client, model), model, "azure"
    
    elif provider == "gemini":
        client, model = get_gemini_client(model)
        return (client, model), model, "gemini"
    
    elif provider == "opensource":
        client, sampling_params = get_opensource_llm(model, **kwargs)
        return (client, sampling_params), model, "opensource"
    
    else:
        raise ValueError(f"Invalid provider: {provider}. Available: openai, azure, gemini, opensource, dummy")


# Convenience function for backward compatibility
def setup_llm(provider: str, model: str, **kwargs):
    """
    Setup LLM client (alias for get_llm_client)
    """
    return get_llm_client(provider, model, **kwargs)