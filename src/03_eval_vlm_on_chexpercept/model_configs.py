"""
Centralized model configuration registry for open-source VLMs.

Each entry key is the short model alias used in CLI / eval scripts.
Required fields  : model_name
Optional fields  : max_model_len, limit_mm_per_prompt, trust_remote_code,
                   enforce_eager, max_num_seqs, dtype, extra_vllm_kwargs
Meta fields      : domain ("general" | "medical"), description
"""

from typing import Dict, Any

# ---------------------------------------------------------------------------
# Open-source VLM configs
# ---------------------------------------------------------------------------
# key -> vLLM-compatible config dict

OPENSOURCE_MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {

    # -----------------------------------------------------------------------
    # General-domain models
    # -----------------------------------------------------------------------
    "qwen3.5-moe": {
        "model_name": "Qwen/Qwen3.5-122B-A10B",
        "domain": "general",
        "description": "Qwen3.5 MoE 122B total / 10B active",
        "max_model_len": 65536,
        "limit_mm_per_prompt": {"image": 7},
        "reasoning_parser": "qwen3",
    },
    "qwen3.6-27b": {
        "model_name": "Qwen/Qwen3.6-27B",
        "domain": "general",
        "description": "Qwen3.6 27B",
        "max_model_len": 65536,
        "limit_mm_per_prompt": {"image": 7},
        "reasoning_parser": "qwen3",
    },
    "qwen3.6": {
        "model_name": "Qwen/Qwen3.6-35B-A3B",
        "domain": "general",
        "description": "Qwen3.6 35B",
        "max_model_len": 65536,
        "limit_mm_per_prompt": {"image": 7},
        "reasoning_parser": "qwen3",
    },
    "qwen3.5": {
        "model_name": "Qwen/Qwen3.5-27B",
        "domain": "general",
        "description": "Qwen3.5 27B",
        "max_model_len": 65536,
        "limit_mm_per_prompt": {"image": 7},
        "reasoning_parser": "qwen3",
    },
    "glm-4v": {
        "model_name": "zai-org/GLM-4.6V",
        "domain": "general",
        "description": "GLM-4.6V vision-language model",
        "max_model_len": 65536,
        "trust_remote_code": True,
        "limit_mm_per_prompt": {"image": 7},
        "gpu_num": 4,
    },
    "internvl3.5": {
        "model_name": "OpenGVLab/InternVL3_5-38B",
        "domain": "general",
        "description": "InternVL3_5 38B",
        "max_model_len": 40960,
        "trust_remote_code": True,
        "limit_mm_per_prompt": {"image": 7},
    },
    "gemma3": {
        "model_name": "google/gemma-3-27b-it",
        "domain": "general",
        "description": "Gemma3 27B",
        "max_model_len": 65536,
        "trust_remote_code": True,
        "limit_mm_per_prompt": {"image": 7},
    },
    "gemma4": {
        "model_name": "google/gemma-4-31B-it",
        "domain": "general",
        "description": "Gemma4 31B",
        "max_model_len": 65536,
        "trust_remote_code": True,
        "limit_mm_per_prompt": {"image": 7},
    },

    # -----------------------------------------------------------------------
    # Medical-domain models
    # -----------------------------------------------------------------------
    "medgemma": {
        "model_name": "google/medgemma-27b-it",
        "domain": "medical",
        "description": "MedGemma 27B Instruct",
        "max_model_len": 65536,
    },
    "medgemma1.5":{
        "model_name": "google/medgemma-1.5-4b-it",
        "domain": "medical",
        "description": "MedGemma 1.5 4B",
        "max_model_len": 65536,
        "trust_remote_code": True,
        "limit_mm_per_prompt": {"image": 7},
    },
    "huatuo": {
        "model_name": "FreedomIntelligence/HuatuoGPT-Vision-7B-Qwen2.5VL",
        "domain": "medical",
        "description": "HuatuoGPT-Vision 7B (Qwen2.5-VL backbone)",
        "max_model_len": 65536,
        "limit_mm_per_prompt": {"image": 7},
        "gpu_num": 4, 
    },
    "lingshu": {
        "model_name": "lingshu-medical-mllm/Lingshu-32B",
        "domain": "medical",
        "description": "Lingshu-32B",
        "trust_remote_code": True,
        "max_model_len": 65536,
        "limit_mm_per_prompt": {"image": 7},
    },
    "hulu-med": { 
        "model_name": "ZJU-AI4H/Hulu-Med-32B",
        "domain": "medical",
        "description": "Hulu-Med-32B",
        "trust_remote_code": True,
        "max_model_len": 32768,
        "limit_mm_per_prompt": {"image": 7},
    }
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def list_models(domain: str = None) -> Dict[str, str]:
    """Return {alias: model_name} optionally filtered by domain."""
    return {
        alias: cfg["model_name"]
        for alias, cfg in OPENSOURCE_MODEL_CONFIGS.items()
        if domain is None or cfg.get("domain") == domain
    }


def get_model_config(alias: str) -> Dict[str, Any]:
    """Return config dict for *alias*, raising ValueError if unknown."""
    if alias not in OPENSOURCE_MODEL_CONFIGS:
        available = list(OPENSOURCE_MODEL_CONFIGS.keys())
        raise ValueError(
            f"Unknown model alias '{alias}'. "
            f"Available: {available}"
        )
    return OPENSOURCE_MODEL_CONFIGS[alias]

def get_safe_tensor_parallel_size(alias: str, requested: int) -> int:
    """Return the largest valid tensor_parallel_size <= *requested*.

    vLLM requires tensor_parallel_size to evenly divide num_attention_heads.
    If the model config does not specify num_attention_heads, *requested* is
    returned unchanged (caller is responsible).

    Args:
        alias:     Model alias defined in OPENSOURCE_MODEL_CONFIGS.
        requested: Desired number of GPUs (e.g. from CUDA_VISIBLE_DEVICES).

    Returns:
        Adjusted tensor_parallel_size that satisfies the divisibility constraint.
    """
    cfg = get_model_config(alias)
    gpu_num = cfg.get("gpu_num")
    if gpu_num is None:
        return requested

    # Largest divisor of heads that is <= requested
    best = max(n for n in range(1, requested + 1) if gpu_num % n == 0)

    if best != requested:
        print(
            f"[GPU] '{alias}' has {gpu_num} GPUs → "
            f"tensor_parallel_size adjusted {requested} → {best} "
            f"(valid: {sorted(n for n in range(1, gpu_num + 1) if gpu_num % n == 0)})"
        )
    return best
