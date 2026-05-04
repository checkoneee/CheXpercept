import os
import yaml

# Set HF_* env vars before any `import torch` so vLLM/transformers see them.
# hf_token comes from api_info/api_keys.yaml (secret); hf_home/hf_hub_cache
# come from cfg/config.yaml (path config).
_repo_root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
with open(os.path.join(_repo_root, "api_info/api_keys.yaml")) as _f:
    _api_keys = yaml.safe_load(_f)
os.environ["HF_TOKEN"] = _api_keys["hf_token"]
_cfg_path = os.path.join(_repo_root, "cfg/config.yaml")
_hf_cfg = (yaml.safe_load(open(_cfg_path)) if os.path.exists(_cfg_path) else {}).get("huggingface", {})
if _hf_cfg.get("hf_home"):
    os.environ["HF_HOME"] = _hf_cfg["hf_home"]
if _hf_cfg.get("hf_hub_cache"):
    os.environ["HF_HUB_CACHE"] = _hf_cfg["hf_hub_cache"]

import sys
import json
import argparse
from typing import Any, Dict, List, Optional, Tuple
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
if project_root not in sys.path:
    sys.path.append(project_root)

from get_llm import get_llm_client
from inference_vlm import inference_vllm, inference_vllms, parse_response, summarize_token_usage
from oracle import (
    ensure_chat_history_idx,
    patch_chat_history_response,
    format_oracle_option_response,
    return_oracle_answer,
)
from scoring import evaluate_model_output

SYSTEM_MESSAGE = (
    "You are an expert radiologist. Analyze the given chest X-ray image and the question. "
    "Provide the final answer in the following format: 'Answer: [Option Number]'. "
    "Only the option number(s) should be included in the final answer part. "
    "(e.g., 'Answer: 1', 'Answer: 2', 'Answer: 1,3')."
)


# ---------------------------------------------------------------------------
# Route helper
# ---------------------------------------------------------------------------

def return_qa_path(qa: Dict) -> str:
    """Determine the evaluation route for a QA entry.

    Returns
    -------
    'lesion_free'       – lesion absent → detection stage only
    'revision_free'     – lesion present, mask acceptable
    'revision_required' – lesion present, mask needs revision
    """
    if qa.get("detection_qa", {}).get("answer") != "Yes":
        return "lesion_free"
    return (
        "revision_free"
        if qa.get("contour_qa", {}).get("no_deformation")
        else "revision_required"
    )


# ---------------------------------------------------------------------------
# Inference wrapper
# ---------------------------------------------------------------------------

def _infer(
    model_id: str,
    llm_model: Any,
    sampling_params: Any,
    question: str,
    imgs: List[str],
    system_message: str,
    chat_history: List,
    token_log: Optional[List] = None,
    enable_thinking: bool = True,
) -> Tuple[str, List]:
    """Run inference and return (parsed_response, updated_chat_history)."""
    fn = inference_vllms(model_id)
    if not enable_thinking and any(k in model_id.lower() for k in ('qwen3.5', 'qwen3.6')) and fn is inference_vllm:
        raw, chat_history, token_info = fn(
            llm_model, sampling_params, question, imgs, system_message, chat_history,
            chat_kwargs={"chat_template_kwargs": {"enable_thinking": False}},
        )
    else:
        raw, chat_history, token_info = fn(
            llm_model, sampling_params, question, imgs, system_message, chat_history,
        )
    if token_info is not None and token_log is not None:
        token_log.append(token_info)
    return parse_response(raw, model_id), chat_history


# ---------------------------------------------------------------------------
# Stage evaluators
# ---------------------------------------------------------------------------

def evaluate_detection_qa(
    qa: Dict,
    key_id: str,
    chexpercept_path: str,
    llm_model: Any,
    sampling_params: Any,
    system_message: str,
    chat_history: List,
    model_id: str,
    token_log: Optional[List] = None,
    enable_thinking: bool = True,
) -> Tuple[Optional[str], List]:
    """Stage 1 – Lesion presence detection."""
    detection_qa = qa.get("detection_qa", {})
    if not detection_qa:
        return None, chat_history

    img_path = os.path.join(chexpercept_path, key_id, "detection_qa", "xray.png")
    if not os.path.exists(img_path):
        print(f"Warning: Image not found: {img_path}")
        return None, chat_history

    return _infer(
        model_id, llm_model, sampling_params,
        detection_qa["question"], [img_path], system_message, chat_history,
        token_log=token_log, enable_thinking=enable_thinking,
    )


def evaluate_contour_evaluation_qa(
    qa: Dict,
    key_id: str,
    chexpercept_path: str,
    llm_model: Any,
    sampling_params: Any,
    system_message: str,
    chat_history: List,
    results: Dict,
    oracle_setting: str,
    model_id: str,
    token_log: Optional[List] = None,
    enable_thinking: bool = True,
) -> Tuple[Optional[str], List]:
    """Stage 2 – Does the contour mask need revision?"""
    contour_qa = qa.get("contour_qa", {})
    contour_eval_qa = contour_qa.get("contour_eval_qa", {})
    if not contour_qa or not contour_eval_qa:
        return None, chat_history

    contour_root = os.path.join(chexpercept_path, key_id, "contour_qa")
    if not os.path.exists(contour_root):
        print(f"Warning: Folder not found: {contour_root}")
        return None, chat_history

    oracle = return_oracle_answer(qa, results, step="contour_evaluation_qa")
    question = contour_eval_qa.get("question", "")

    if oracle_setting == "explicit" and oracle:
        question = oracle + "\n\n" + question
    elif oracle_setting == "implicit" and oracle:
        gt_resp = format_oracle_option_response(
            results.get("detection_qa", {}).get("ground_truth")
        )
        if gt_resp:
            patch_chat_history_response(
                chat_history,
                ensure_chat_history_idx(results).get("detection_qa"),
                gt_resp,
            )

    imgs = []
    rel_path = contour_eval_qa.get("relative_path", "")
    if rel_path:
        img_path = os.path.join(contour_root, rel_path)
        if os.path.exists(img_path):
            imgs.append(img_path)

    response, chat_history = _infer(
        model_id, llm_model, sampling_params, question, imgs, system_message, chat_history,
        token_log=token_log, enable_thinking=enable_thinking,
    )
    ensure_chat_history_idx(results)["contour_evaluation_qa"] = len(chat_history) - 1
    return response, chat_history


def evaluate_contour_revision_qa(
    qa: Dict,
    key_id: str,
    chexpercept_path: str,
    llm_model: Any,
    sampling_params: Any,
    system_message: str,
    chat_history: List,
    results: Dict,
    oracle_setting: str,
    model_id: str,
    token_log: Optional[List] = None,
    enable_thinking: bool = True,
) -> Tuple[Dict, List]:
    """Stage 3 – Revision QA (expansion, contraction, revision_result)."""
    contour_qa = qa.get("contour_qa", {})
    if not contour_qa:
        return {}, chat_history

    contour_root = os.path.join(chexpercept_path, key_id, "contour_qa")
    if not os.path.exists(contour_root):
        print(f"Warning: Folder not found: {contour_root}")
        return {}, chat_history

    revision_folder = os.path.join(contour_root, "contour_revision_qa")
    responses = {}

    results["contour_revision_qa"]["ground_truth"] = {
        key: contour_qa.get(f"contour_revision_qa_{key}", {}).get("answer_index")
        for key in ("expansion", "contraction", "revision_result")
    }

    # Build initial context with optional oracle
    initial_context = contour_qa.get("initial_context", "")
    oracle = return_oracle_answer(qa, results, step="contour_revision_qa")

    if oracle_setting == "explicit" and oracle:
        initial_context = oracle + "\n\n" + initial_context
    elif oracle_setting == "implicit" and oracle:
        gt_resp = format_oracle_option_response(
            results.get("contour_evaluation_qa", {}).get("ground_truth")
        )
        if gt_resp:
            patch_chat_history_response(
                chat_history,
                ensure_chat_history_idx(results).get("contour_evaluation_qa"),
                gt_resp,
            )

    # 1. Expansion QA
    expansion_qa = contour_qa.get("contour_revision_qa_expansion", {})
    img_path = os.path.join(revision_folder, "xray_with_mask_and_points.png")
    if expansion_qa and os.path.exists(img_path):
        responses["expansion"], chat_history = _infer(
            model_id, llm_model, sampling_params,
            f"{initial_context}\n\n{expansion_qa['question']}",
            [img_path], system_message, chat_history,
            token_log=token_log, enable_thinking=enable_thinking,
        )
        ensure_chat_history_idx(results)["contour_revision_qa_expansion"] = len(chat_history) - 1

    # 2. Contraction QA
    contraction_qa = contour_qa.get("contour_revision_qa_contraction", {})
    if contraction_qa:
        responses["contraction"], chat_history = _infer(
            model_id, llm_model, sampling_params,
            contraction_qa["question"], [], system_message, chat_history,
            token_log=token_log, enable_thinking=enable_thinking,
        )
        ensure_chat_history_idx(results)["contour_revision_qa_contraction"] = len(chat_history) - 1

    # Score expansion/contraction before proceeding to revision_result
    results["contour_revision_qa"]["responses"] = responses
    evaluate_model_output(results, step="contour_revision_qa_revision")

    # 3. Revision Result QA
    revision_result_qa = contour_qa.get("contour_revision_qa_revision_result", {})
    if revision_result_qa:
        oracle_rr = return_oracle_answer(qa, results, step="contour_revision_qa_revision_result")
        question = revision_result_qa["question"]

        if oracle_setting == "explicit" and oracle_rr:
            question = oracle_rr + "\n\n" + question
        elif oracle_setting == "implicit" and oracle_rr:
            idx_map = ensure_chat_history_idx(results)
            gt = results["contour_revision_qa"]["ground_truth"]
            rev = results["contour_revision_qa"]
            for sub_key in ("expansion", "contraction"):
                gt_resp = format_oracle_option_response(gt.get(sub_key))
                if gt_resp and rev.get(sub_key, {}).get("correct") is False:
                    patch_chat_history_response(
                        chat_history,
                        idx_map.get(f"contour_revision_qa_{sub_key}"),
                        gt_resp,
                    )

        imgs = [
            os.path.join(revision_folder, opt["relative_path"])
            for opt in revision_result_qa.get("answer_options", [])
            if opt.get("relative_path")
            and os.path.exists(os.path.join(revision_folder, opt["relative_path"]))
        ]
        if imgs:
            responses["revision_result"], chat_history = _infer(
                model_id, llm_model, sampling_params,
                question, imgs, system_message, chat_history,
                token_log=token_log, enable_thinking=enable_thinking,
            )
            ensure_chat_history_idx(results)["contour_revision_qa_revision_result"] = len(chat_history) - 1

    evaluate_model_output(results, step="contour_revision_qa_revision_result")
    return responses, chat_history


def evaluate_attribute_extraction_qa(
    qa: Dict,
    key_id: str,
    chexpercept_path: str,
    llm_model: Any,
    sampling_params: Any,
    system_message: str,
    chat_history: List,
    results: Dict,
    oracle_setting: str,
    model_id: str,
    token_log: Optional[List] = None,
    enable_thinking: bool = True,
) -> Tuple[Dict, List]:
    """Stage 4 – Attribute extraction (distribution, location, severity, comparison)."""
    attribute_extraction_qa = qa.get("attribute_extraction_qa", {})
    if not attribute_extraction_qa:
        return {}, chat_history

    responses = {}
    oracle = return_oracle_answer(qa, results, step="attribute_extraction_qa")

    for qa_type in ("distribution", "location", "severity/measurement", "comparison"):
        if qa_type not in attribute_extraction_qa:
            continue

        question = attribute_extraction_qa[qa_type]["question"]

        # Oracle injection only on the first turn (distribution)
        if qa_type == "distribution":
            if oracle_setting == "explicit" and oracle:
                question = oracle + "\n\n" + question
            elif oracle_setting == "implicit" and oracle:
                idx_map = ensure_chat_history_idx(results)
                qa_path = results.get("qa_path")
                if qa_path == "revision_free":
                    gt_resp = format_oracle_option_response(
                        results.get("contour_evaluation_qa", {}).get("ground_truth")
                    )
                    if gt_resp:
                        patch_chat_history_response(
                            chat_history, idx_map.get("contour_evaluation_qa"), gt_resp
                        )
                elif qa_path == "revision_required":
                    gt_resp = format_oracle_option_response(
                        results.get("contour_revision_qa", {})
                        .get("ground_truth", {})
                        .get("revision_result")
                    )
                    if gt_resp:
                        patch_chat_history_response(
                            chat_history,
                            idx_map.get("contour_revision_qa_revision_result"),
                            gt_resp,
                        )

        responses[qa_type], chat_history = _infer(
            model_id, llm_model, sampling_params,
            question, [], system_message, chat_history,
            token_log=token_log, enable_thinking=enable_thinking,
        )
        ensure_chat_history_idx(results)[f"attribute_extraction_qa_{qa_type}"] = len(chat_history) - 1

    return responses, chat_history


# ---------------------------------------------------------------------------
# Top-level single-QA evaluator
# ---------------------------------------------------------------------------

def _finalize_results(results: Dict, chat_history: List) -> Dict:
    results["chat_history"] = chat_history
    return results


def evaluate_single_qa(
    model_name: str,
    key_id: str,
    qa: Dict,
    chexpercept_path: str,
    llm_model: Any,
    sampling_params: Any,
    oracle_setting: str = "explicit",
    enable_thinking: bool = True,
) -> Dict:
    """Run all evaluation stages for a single QA entry.

    Args:
        model_name:       Model alias (used to select inference function).
        key_id:           Dataset entry identifier.
        qa:               Full QA dictionary for this entry.
        chexpercept_path: Root directory of the CheXpercept dataset.
        llm_model:        Loaded model instance.
        sampling_params:  Inference sampling parameters.
        oracle_setting:   'explicit' | 'implicit' | 'none'

    Returns:
        Dictionary with per-stage responses, ground truth, and correctness flags.
    """
    qa_path = return_qa_path(qa)
    results: Dict[str, Any] = {
        "key_id": key_id,
        "qa_path": qa_path,
        "detection_qa": {},
        "contour_evaluation_qa": {},
        "contour_revision_qa": {},
        "attribute_extraction_qa": {},
    }

    token_log: List = []
    shared = dict(
        chexpercept_path=chexpercept_path,
        llm_model=llm_model,
        sampling_params=sampling_params,
        system_message=SYSTEM_MESSAGE,
        model_id=model_name,
        token_log=token_log,
        enable_thinking=enable_thinking,
    )
    chat_history: List = []

    # Stage 1 – Lesion presence detection (all routes)
    detection_response, chat_history = evaluate_detection_qa(
        qa, key_id, chat_history=chat_history, **shared
    )
    if detection_response:
        results["detection_qa"]["response"] = detection_response
        results["detection_qa"]["ground_truth"] = qa.get("detection_qa", {}).get("answer_index")
        evaluate_model_output(results, step="detection_qa")
        ensure_chat_history_idx(results)["detection_qa"] = len(chat_history) - 1

    if qa_path == "lesion_free":
        if token_log:
            results["token_usage"] = summarize_token_usage(token_log, model_name)
        return _finalize_results(results, chat_history)

    # Stage 2 – Contour evaluation (routes A, B)
    contour_eval_response, chat_history = evaluate_contour_evaluation_qa(
        qa, key_id, chat_history=chat_history, results=results,
        oracle_setting=oracle_setting, **shared
    )
    if contour_eval_response:
        results["contour_evaluation_qa"]["responses"] = contour_eval_response
        results["contour_evaluation_qa"]["ground_truth"] = (
            qa.get("contour_qa", {}).get("contour_eval_qa", {}).get("answer_index")
        )
        evaluate_model_output(results, step="contour_evaluation_qa")

    # Stage 3 – Contour revision (route A only)
    if qa_path == "revision_required":
        _, chat_history = evaluate_contour_revision_qa(
            qa, key_id, chat_history=chat_history, results=results,
            oracle_setting=oracle_setting, **shared
        )

    # Stage 4 – Attribute extraction (routes A, B)
    attr_responses, chat_history = evaluate_attribute_extraction_qa(
        qa, key_id, chat_history=chat_history, results=results,
        oracle_setting=oracle_setting, **shared
    )
    if attr_responses:
        results["attribute_extraction_qa"]["responses"] = attr_responses
        results["attribute_extraction_qa"]["ground_truth"] = {
            qa_type: qa.get("attribute_extraction_qa", {}).get(qa_type, {}).get("answer_index")
            for qa_type in attr_responses
        }
        evaluate_model_output(results, step="attribute_extraction_qa")

    if token_log:
        results["token_usage"] = summarize_token_usage(token_log, model_name)

    return _finalize_results(results, chat_history)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _save_results(all_results: Dict, out_file: str) -> None:
    """Atomically save results to JSON (write to temp then rename)."""
    tmp_file = out_file + ".tmp"
    with open(tmp_file, "w") as f:
        json.dump(all_results, f, indent=2)
    os.replace(tmp_file, out_file)


def main(
    benchmark_dir: str,
    output_path_root: str,
    provider: str = "opensource",
    model_name: str = "medgemma",
    limit: Optional[int] = None,
    per_path_limit: Optional[int] = None,
    debug: bool = False,
    oracle_setting: str = "explicit",
    enable_thinking: bool = True,
    resume: bool = False,
    save_every: int = 10,
) -> None:
    """Run VLM evaluation on the CheXpercept benchmark.

    Args:
        benchmark_dir: Root directory of the downloaded CheXpercept benchmark.
                       Must contain ``chexpercept.json`` and a ``chexpercept/``
                       sub-directory with per-key_id image folders.
        output_path_root: Where to write per-model evaluation results.
    """
    qa_results_path = os.path.join(benchmark_dir, "chexpercept.json")
    chexpercept_path = os.path.join(benchmark_dir, "chexpercept")

    print(f"Loading QA results from: {qa_results_path}")
    with open(qa_results_path) as f:
        qa_results = json.load(f)
    print(f"Loaded {len(qa_results)} QA sets")

    prefix = "[DEBUG MODE] " if debug else ""
    print(f"{prefix}Initializing {provider}/{model_name}...")
    client_tuple, model_name_returned, provider_returned = get_llm_client(
        provider, model_name, debug=debug
    )
    print(f"{prefix}Model ready.")

    llm_model, sampling_params = (
        client_tuple if isinstance(client_tuple, tuple) else (client_tuple, None)
    )

    model_dir = f"{provider_returned}_{model_name}"
    if not enable_thinking:
        model_dir += "_no_thinking"
    output_path = os.path.join(
        output_path_root,
        model_dir,
        f"oracle_{oracle_setting}",
    )
    os.makedirs(output_path, exist_ok=True)

    items = list(qa_results.items())

    if per_path_limit:
        buckets: Dict[str, List] = {"lesion_free": [], "revision_free": [], "revision_required": []}
        for key_id, qa in items:
            buckets.setdefault(return_qa_path(qa), []).append((key_id, qa))
        items = []
        for path_name, bucket in buckets.items():
            items.extend(bucket[:per_path_limit])
            print(f"  [{path_name}] selected {min(len(bucket), per_path_limit)}/{len(bucket)}")
        print(f"Per-path limit: {per_path_limit} → total {len(items)} QA sets")
    elif limit:
        items = items[:limit]
        print(f"Limiting evaluation to {limit} QA sets")

    out_file = os.path.join(output_path, "all_results.json")

    # Resume: load existing results and skip already-completed entries
    all_results = {}
    if resume and os.path.exists(out_file):
        with open(out_file) as f:
            all_results = json.load(f)
        print(f"Resumed from checkpoint: {len(all_results)} entries already completed")

    new_count = 0
    for key_id, qa in tqdm(items, desc="Evaluating"):
        if key_id in all_results:
            continue
        try:
            all_results[key_id] = evaluate_single_qa(
                model_name, key_id, qa, chexpercept_path,
                llm_model, sampling_params,
                oracle_setting=oracle_setting,
                enable_thinking=enable_thinking,
            )
            new_count += 1
            if new_count % save_every == 0:
                _save_results(all_results, out_file)
                print(f"  [checkpoint] Saved {len(all_results)} entries to {out_file}")
        except Exception as e:
            import traceback
            print(f"Error evaluating {key_id}: {e}")
            traceback.print_exc()

    _save_results(all_results, out_file)
    print(f"\nDone! Results saved to {out_file} ({len(all_results)} entries)")

    # Aggregate and print token usage / cost summary
    total_cost = 0.0
    total_input = 0
    total_cached_input = 0
    total_output = 0
    total_thinking = 0
    total_calls = 0
    for r in all_results.values():
        usage = r.get("token_usage", {})
        total_cost += usage.get("estimated_cost_usd", 0)
        total_input += usage.get("input_tokens", 0)
        total_cached_input += usage.get("cached_input_tokens", 0)
        total_output += usage.get("output_tokens", 0)
        total_thinking += usage.get("thinking_tokens", 0)
        total_calls += usage.get("num_calls", 0)

    if total_calls > 0:
        print(f"\n{'='*60}")
        print(f"  Token Usage Summary  ({provider}/{model_name})")
        print(f"{'='*60}")
        print(f"  QA entries evaluated : {len(all_results):,}")
        print(f"  Total API calls      : {total_calls:,}")
        print(f"  Input tokens         : {total_input:,}")
        if total_cached_input:
            print(f"    (cached input)     : {total_cached_input:,}")
        print(f"  Output tokens        : {total_output:,}")
        if total_thinking:
            print(f"    (thinking tokens)  : {total_thinking:,}")
        print(f"  Estimated cost (USD) : ${total_cost:.4f}")
        print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate VLMs on CheXpercept.\n\n"
            "Place the downloaded CheXpercept benchmark at "
            "<project_root>/data/chexpercept/ (default), so the layout is:\n"
            "  data/chexpercept/\n"
            "    ├── chexpercept.json     # QA pairs\n"
            "    └── chexpercept/         # per-key_id image folders\n"
            "Override the location with --benchmark-dir."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--benchmark-dir", type=str,
        default=os.path.join(project_root, "data/chexpercept"),
        help="Root of the downloaded benchmark (must contain chexpercept.json + chexpercept/).",
    )
    parser.add_argument(
        "--output-path", type=str,
        default=os.path.join(current_dir, "outputs"),
        help="Where to write per-model evaluation results.",
    )
    parser.add_argument(
        "--provider", type=str, default="opensource",
        choices=["opensource", "openai", "azure", "gemini", "dummy"],
    )
    parser.add_argument("--model", type=str, default="medgemma1.5")
    parser.add_argument("--limit", type=int, help="Limit total QA count (for testing)")
    parser.add_argument(
        "--per_path_limit", type=int,
        help="Sample up to N QAs per qa_path (lesion_free, revision_free, revision_required). Overrides --limit.",
    )
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--oracle_setting", type=str, default="implicit",
        choices=["explicit", "implicit", "none"],
        help=(
            "'explicit': prepend GT text before the next question; "
            "'implicit': silently replace prior assistant response with GT option; "
            "'none': no oracle injection."
        ),
    )
    parser.add_argument(
        "--no-thinking", action="store_true",
        help="Disable thinking mode for Qwen3.5 models",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from existing checkpoint (skip already-completed entries)",
    )
    parser.add_argument(
        "--save-every", type=int, default=10,
        help="Save checkpoint every N entries (default: 10)",
    )
    args = parser.parse_args()

    expected_json = os.path.join(args.benchmark_dir, "chexpercept.json")
    expected_imgs = os.path.join(args.benchmark_dir, "chexpercept")
    if not (os.path.isfile(expected_json) and os.path.isdir(expected_imgs)):
        raise FileNotFoundError(
            f"Benchmark not found at: {args.benchmark_dir}\n"
            f"Expected layout:\n"
            f"  {args.benchmark_dir}/\n"
            f"    chexpercept.json\n"
            f"    chexpercept/<key_id>/...\n"
            f"Place the downloaded CheXpercept benchmark there, or pass "
            f"--benchmark-dir <path>."
        )

    main(
        benchmark_dir=args.benchmark_dir,
        output_path_root=args.output_path,
        provider=args.provider,
        model_name=args.model,
        limit=args.limit,
        per_path_limit=args.per_path_limit,
        debug=args.debug,
        oracle_setting=args.oracle_setting,
        enable_thinking=not args.no_thinking,
        resume=args.resume,
        save_every=args.save_every,
    )
