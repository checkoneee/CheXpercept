"""
Estimate API cost for evaluating CheXpercept benchmark.

Simulates the multi-turn evaluation flow WITHOUT making any API calls.
Counts input tokens from text + images and assumes a fixed output token count.
All input tokens billed at full price (no caching assumed).

Usage:
    python estimate_cost.py
    python estimate_cost.py --model gpt-5.4-nano --output-tokens 100
    python estimate_cost.py --model gemini-3.1-flash-lite-preview
"""

import os
import sys
import json
import argparse
from typing import Dict, List

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
if project_root not in sys.path:
    sys.path.append(project_root)

from utils.config import open_config

# ---------------------------------------------------------------------------
# Pricing (copied to avoid importing inference_vlm which pulls in torch/vllm)
# ---------------------------------------------------------------------------

MODEL_PRICING = {
    # Gemini (https://ai.google.dev/gemini-api/docs/pricing)
    "gemini-3.1-flash-lite-preview": {"input": 0.25,  "cached_input": 0.025,  "output": 1.50,  "image_tokens": 1089, "default_output_tokens": 1000, "cache_min_tokens": 8000},
    "gemini-3.1-pro-preview":        {"input": 2.00,  "cached_input": 0.20,   "output": 12.00, "image_tokens": 1089, "default_output_tokens": 1000, "cache_min_tokens": 8000},
    # OpenAI (https://developers.openai.com/api/docs/pricing)
    "gpt-5.4-nano": {"input": 0.20,  "cached_input": 0.02,   "output": 1.25,  "image_tokens": 800, "default_output_tokens": 1000, "cache_min_tokens": 1024},
    "gpt-5.4":      {"input": 2.50,  "cached_input": 0.25,   "output": 15.00, "image_tokens": 800, "default_output_tokens": 1000, "cache_min_tokens": 1024},
}
# All prices in USD per 1M tokens. Gemini prices are for prompts <= 200k tokens.
# Image tokens assume 1024x1024 input images.

def _lookup_pricing(model_id: str) -> dict:
    model_lower = model_id.lower()
    best, best_len = {}, 0
    for key, val in MODEL_PRICING.items():
        if key in model_lower and len(key) > best_len:
            best, best_len = val, len(key)
    return best


# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------
# Text: tiktoken (exact for GPT, approximate for Gemini — no offline tokenizer)
# Images: per-model token counts from MODEL_PRICING["image_tokens"]

import tiktoken

_TOKENIZER = tiktoken.encoding_for_model("gpt-4o")


def _image_tokens_for_model(model_id: str) -> int:
    pricing = _lookup_pricing(model_id)
    return pricing.get("image_tokens", 1024)


def _text_tokens(text: str) -> int:
    if not text:
        return 0
    return len(_TOKENIZER.encode(text))


# ---------------------------------------------------------------------------
# QA path helper
# ---------------------------------------------------------------------------

def return_qa_path(qa: Dict) -> str:
    if qa.get("detection_qa", {}).get("answer") != "Yes":
        return "lesion_free"
    return (
        "revision_free"
        if qa.get("contour_qa", {}).get("no_deformation")
        else "revision_required"
    )


# ---------------------------------------------------------------------------
# Per-QA cost estimation
# ---------------------------------------------------------------------------

SYSTEM_MESSAGE = (
    "You are an expert radiologist. Analyze the given chest X-ray image and the question. "
    "Provide the final answer in the following format: 'Answer: [Option Number]'. "
    "Only the option number(s) should be included in the final answer part. "
    "(e.g., 'Answer: 1', 'Answer: 2', 'Answer: 1,3')."
)


def estimate_single_qa(
    key_id: str,
    qa: Dict,
    chexpercept_path: str,
    model_id: str,
    output_tokens_per_call: int,
) -> Dict:
    """Estimate tokens and cost for a single QA entry without calling any API.

    Each API call includes the full conversation (system + history + new content),
    all billed at the full input price (no caching assumed).
    """
    qa_path = return_qa_path(qa)
    sys_tok = _text_tokens(SYSTEM_MESSAGE)

    # Track prefix (cacheable) and new tokens separately per call.
    # Prefix = system + chat history from prior turns (repeated across calls).
    # New = current question + images (unique to this call).
    # GPT caches the prefix when it exceeds 1024 tokens.
    calls_prefix_text = []    # prefix text tokens per call
    calls_prefix_images = []  # prefix image count per call
    calls_new_text = []       # new text tokens per call
    calls_new_images = []     # new image count per call
    history_text_tokens = 0
    history_image_count = 0

    def _add_call(question: str, num_images: int):
        nonlocal history_text_tokens, history_image_count
        new_text = _text_tokens(question)
        calls_prefix_text.append(sys_tok + history_text_tokens)
        calls_prefix_images.append(history_image_count)
        calls_new_text.append(new_text)
        calls_new_images.append(num_images)
        history_text_tokens += new_text + output_tokens_per_call
        history_image_count += num_images

    # Stage 1: Detection QA (all paths)
    detection_qa = qa.get("detection_qa", {})
    if detection_qa:
        img_path = os.path.join(chexpercept_path, key_id, "detection_qa", "xray.png")
        num_imgs = 1 if os.path.exists(img_path) else 0
        _add_call(detection_qa.get("question", ""), num_imgs)

    if qa_path == "lesion_free":
        return _make_result(key_id, qa_path,
                            calls_prefix_text, calls_prefix_images,
                            calls_new_text, calls_new_images,
                            output_tokens_per_call, model_id)

    # Stage 2: Contour evaluation QA
    contour_qa = qa.get("contour_qa", {})
    contour_eval_qa = contour_qa.get("contour_eval_qa", {})
    if contour_eval_qa:
        contour_root = os.path.join(chexpercept_path, key_id, "contour_qa")
        rel_path = contour_eval_qa.get("relative_path", "")
        num_imgs = 0
        if rel_path:
            img_path = os.path.join(contour_root, rel_path)
            if os.path.exists(img_path):
                num_imgs = 1
        _add_call(contour_eval_qa.get("question", ""), num_imgs)

    # Stage 3: Contour revision QA (revision_required only)
    if qa_path == "revision_required":
        expansion_qa = contour_qa.get("contour_revision_qa_expansion", {})
        initial_context = contour_qa.get("initial_context", "")
        revision_folder = os.path.join(chexpercept_path, key_id, "contour_qa", "contour_revision_qa")
        if expansion_qa:
            img_path = os.path.join(revision_folder, "xray_with_mask_and_points.png")
            num_imgs = 1 if os.path.exists(img_path) else 0
            question = f"{initial_context}\n\n{expansion_qa.get('question', '')}"
            _add_call(question, num_imgs)

        contraction_qa = contour_qa.get("contour_revision_qa_contraction", {})
        if contraction_qa:
            _add_call(contraction_qa.get("question", ""), 0)

        revision_result_qa = contour_qa.get("contour_revision_qa_revision_result", {})
        if revision_result_qa:
            num_imgs = 0
            for opt in revision_result_qa.get("answer_options", []):
                rp = opt.get("relative_path")
                if rp and os.path.exists(os.path.join(revision_folder, rp)):
                    num_imgs += 1
            _add_call(revision_result_qa.get("question", ""), num_imgs)

    # Stage 4: Attribute extraction QA
    attribute_extraction_qa = qa.get("attribute_extraction_qa") or {}
    for qa_type in ("distribution", "location", "severity/measurement", "comparison"):
        if qa_type in attribute_extraction_qa:
            _add_call(attribute_extraction_qa[qa_type].get("question", ""), 0)

    return _make_result(key_id, qa_path,
                        calls_prefix_text, calls_prefix_images,
                        calls_new_text, calls_new_images,
                        output_tokens_per_call, model_id)


def _make_result(
    key_id: str, qa_path: str,
    calls_prefix_text: List[int], calls_prefix_images: List[int],
    calls_new_text: List[int], calls_new_images: List[int],
    output_tokens_per_call: int, model_id: str,
) -> Dict:
    num_calls = len(calls_prefix_text)
    total_output = num_calls * output_tokens_per_call

    img_tok = _image_tokens_for_model(model_id)
    pricing = _lookup_pricing(model_id)
    input_price = pricing.get("input", 0)
    cached_price = pricing.get("cached_input", 0)
    output_price = pricing.get("output", 0)

    total_text = sum(calls_prefix_text) + sum(calls_new_text)
    total_images = sum(calls_prefix_images) + sum(calls_new_images)
    total_input = total_text + total_images * img_tok

    # No-cache cost: all input at full price
    cost = (total_input * input_price + total_output * output_price) / 1_000_000

    # With-cache cost: prefix tokens cached if model supports it and prefix >= 1024 tokens
    cost_cached = 0.0
    if cached_price:
        for i in range(num_calls):
            prefix_tok = calls_prefix_text[i] + calls_prefix_images[i] * img_tok
            new_tok = calls_new_text[i] + calls_new_images[i] * img_tok
            if prefix_tok >= pricing.get("cache_min_tokens", 0):
                call_cost = (prefix_tok * cached_price + new_tok * input_price) / 1_000_000
            else:
                call_cost = ((prefix_tok + new_tok) * input_price) / 1_000_000
            cost_cached += call_cost
        cost_cached += (total_output * output_price) / 1_000_000

    return {
        "key_id": key_id,
        "qa_path": qa_path,
        "num_calls": num_calls,
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "cost": cost,
        "cost_cached": cost_cached,
        # Separated counts for per-model recalculation
        "prefix_text_tokens": sum(calls_prefix_text),
        "prefix_image_count": sum(calls_prefix_images),
        "new_text_tokens": sum(calls_new_text),
        "new_image_count": sum(calls_new_images),
        # Per-call lists for accurate cache_min_tokens recalculation
        "calls_prefix_text": calls_prefix_text,
        "calls_prefix_images": calls_prefix_images,
        "calls_new_text": calls_new_text,
        "calls_new_images": calls_new_images,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(
    config: Dict,
    model_id: str,
    output_tokens_per_call: int,
    limit: int = None,
    per_path_limit: int = None,
):
    qa_results_path = config["qa_generation"]["qa_results_path"]
    chexpercept_path = config["qa_generation"]["chexpercept_path"]

    with open(qa_results_path) as f:
        qa_results = json.load(f)
    print(f"Loaded {len(qa_results)} QA entries")

    pricing = _lookup_pricing(model_id)
    if not pricing:
        print(f"Warning: No pricing found for '{model_id}'. Cost will be $0.")
        print(f"Available models: {list(MODEL_PRICING.keys())}")

    items = list(qa_results.items())
    if per_path_limit:
        buckets: Dict[str, List] = {"lesion_free": [], "revision_free": [], "revision_required": []}
        for key_id, qa in items:
            buckets.setdefault(return_qa_path(qa), []).append((key_id, qa))
        items = []
        for path_name, bucket in buckets.items():
            items.extend(bucket[:per_path_limit])
            print(f"  [{path_name}] selected {min(len(bucket), per_path_limit)}/{len(bucket)}")
    elif limit:
        items = items[:limit]

    totals = {
        "input_tokens": 0, "output_tokens": 0,
        "cost": 0.0, "cost_cached": 0.0, "num_calls": 0,
        # Separated counts for per-model recalculation
        "prefix_text_tokens": 0, "prefix_image_count": 0,
        "new_text_tokens": 0, "new_image_count": 0,
        # Per-call lists for accurate cache_min_tokens recalculation
        "all_calls_prefix_text": [], "all_calls_prefix_images": [],
        "all_calls_new_text": [], "all_calls_new_images": [],
    }
    path_counts = {}

    for key_id, qa in items:
        result = estimate_single_qa(key_id, qa, chexpercept_path, model_id, output_tokens_per_call)
        totals["input_tokens"] += result["total_input_tokens"]
        totals["output_tokens"] += result["total_output_tokens"]
        totals["cost"] += result["cost"]
        totals["cost_cached"] += result["cost_cached"]
        totals["num_calls"] += result["num_calls"]
        totals["prefix_text_tokens"] += result["prefix_text_tokens"]
        totals["prefix_image_count"] += result["prefix_image_count"]
        totals["new_text_tokens"] += result["new_text_tokens"]
        totals["new_image_count"] += result["new_image_count"]
        totals["all_calls_prefix_text"].extend(result["calls_prefix_text"])
        totals["all_calls_prefix_images"].extend(result["calls_prefix_images"])
        totals["all_calls_new_text"].extend(result["calls_new_text"])
        totals["all_calls_new_images"].extend(result["calls_new_images"])
        path_counts[result["qa_path"]] = path_counts.get(result["qa_path"], 0) + 1

    img_tok = _image_tokens_for_model(model_id)
    total_text = totals["prefix_text_tokens"] + totals["new_text_tokens"]
    total_images = totals["prefix_image_count"] + totals["new_image_count"]

    print(f"\n{'=' * 65}")
    print(f"  Cost Estimation: {model_id}")
    print(f"  (output tokens assumed: {output_tokens_per_call} per call)")
    print(f"{'=' * 65}")
    print(f"  QA entries          : {len(items):,}")
    for path, count in sorted(path_counts.items()):
        print(f"    {path:<20}: {count:,}")
    print(f"  Total API calls     : {totals['num_calls']:,}")
    print(f"  Input tokens        : {totals['input_tokens']:,}")
    print(f"    Text tokens       : {total_text:,}")
    print(f"    Image tokens      : {total_images * img_tok:,} ({total_images:,} images × {img_tok:,} tok)")
    print(f"  Output tokens       : {totals['output_tokens']:,}")
    if pricing:
        print(f"  Input price         : ${pricing['input']:.3f} / 1M tokens")
        if pricing.get("cached_input"):
            print(f"  Cached input price  : ${pricing['cached_input']:.3f} / 1M tokens (prefix >= {pricing.get('cache_min_tokens', 0)} tokens)")
        print(f"  Output price        : ${pricing['output']:.2f} / 1M tokens")
    print(f"  -----------------------------------------------")
    print(f"  Estimated cost      : ${totals['cost']:.4f}")
    if totals["cost_cached"]:
        savings = totals["cost"] - totals["cost_cached"]
        print(f"  With caching        : ${totals['cost_cached']:.4f} (saves ${savings:.4f})")
    print(f"{'=' * 65}")

    # Per-model comparison table (recalculates image tokens and output tokens per model)
    if not limit:
        print(f"\n  Cost comparison across all models:")
        print(f"  {'Model':<32} {'Img Tok':>8} {'Out Tok':>8} {'Cost':>12} {'Cached':>12}")
        print(f"  {'-' * 74}")
        m_text = totals["prefix_text_tokens"] + totals["new_text_tokens"]
        m_imgs = totals["prefix_image_count"] + totals["new_image_count"]
        for name, price in MODEL_PRICING.items():
            input_p = price["input"]
            cached_p = price.get("cached_input", 0)
            output_p = price["output"]
            img_tok = price["image_tokens"]
            m_output_tok = price.get("default_output_tokens", 1000)

            # Recompute token totals with this model's image/output token counts
            m_input = m_text + m_imgs * img_tok
            m_output = totals["num_calls"] * m_output_tok
            output_cost = m_output * output_p / 1_000_000

            cost = m_input * input_p / 1_000_000 + output_cost

            # With-cache: approximate using aggregate prefix/new split
            # (per-call threshold already applied in per-QA results above)
            cached_str = "N/A"
            if cached_p:
                prefix_tok = totals["prefix_text_tokens"] + totals["prefix_image_count"] * img_tok
                new_tok = totals["new_text_tokens"] + totals["new_image_count"] * img_tok
                cost_cached = (prefix_tok * cached_p + new_tok * input_p) / 1_000_000 + output_cost
                cached_str = f"${cost_cached:>10.4f}"

            print(f"  {name:<32} {img_tok:>7,} {m_output_tok:>7,} ${cost:>11.4f} {cached_str:>12}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Estimate API cost for CheXpercept evaluation")
    parser.add_argument(
        "--config", type=str,
        default=os.path.join(project_root, "cfg/config_nas.yaml"),
    )
    parser.add_argument(
        "--model", type=str, default="gpt-5.4-nano",
        help=f"Model ID for pricing. Available: {list(MODEL_PRICING.keys())}",
    )
    parser.add_argument(
        "--output-tokens", type=int, default=None,
        help="Assumed output tokens per API call (default: per-model from MODEL_PRICING)",
    )
    parser.add_argument("--limit", type=int, help="Limit QA count")
    parser.add_argument(
        "--per_path_limit", type=int,
        help="Sample up to N QAs per qa_path. Overrides --limit.",
    )
    args = parser.parse_args()

    output_tokens = args.output_tokens
    if output_tokens is None:
        pricing = _lookup_pricing(args.model)
        output_tokens = pricing.get("default_output_tokens", 500)

    config = open_config(args.config)
    main(config, model_id=args.model, output_tokens_per_call=output_tokens, limit=args.limit, per_path_limit=args.per_path_limit)
