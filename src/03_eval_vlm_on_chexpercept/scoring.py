"""
Response parsing and step-level evaluation for the CheXpercept pipeline.
"""
import re
from typing import Dict, List


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def extract_int(text: str) -> int:
    """Return the first integer found in *text*, or -1 if none.

    Recognizes negative sign so that the sentinel "-1" emitted by
    ``normalize_response`` survives the round-trip and stays as -1
    (not 1) — preventing non-numeric responses from being silently
    scored as correct when ground truth happens to equal 1.
    """
    nums = re.findall(r"-?\d+", text)
    return int(nums[0]) if nums else -1


def normalize_response(text: str) -> str:
    """Convert a free-form model response to a comma-separated number string.

    Priority order:
      1. Numbers after 'answer is / answer are / answer:' phrases.
      2. Last non-empty line containing only digits, commas, or spaces.
      3. Fallback: all numbers found anywhere in the text.
    """
    if not text:
        return str(-1)
    text = text.strip()
    # Strip thinking-token artifacts whose digits would otherwise leak into
    # the "any number anywhere" fallback (e.g. gemma's "<unused94>thought").
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<unused\d+>", "", text)

    m = re.search(
        r"answers?\s*(?:is|are)\s*[:\-]?\s*([\d][\d,\s]*)", text, re.IGNORECASE
    )
    if m:
        nums = re.findall(r"\d+", m.group(1))
        if nums:
            return ",".join(nums)

    for line in reversed([ln.strip() for ln in text.split("\n") if ln.strip()]):
        if re.fullmatch(r"[\d,\s]+", line):
            nums = re.findall(r"\d+", line)
            if nums:
                return ",".join(nums)

    nums = re.findall(r"\d+", text)
    return ",".join(nums) if nums else str(-1)


def _parse_nums(response: str) -> List[int]:
    """Normalize *response* and return a list of parsed integers."""
    return [extract_int(r) for r in normalize_response(response).split(",")]


# ---------------------------------------------------------------------------
# Step-level evaluation
# ---------------------------------------------------------------------------

def evaluate_model_output(results: Dict, step: str) -> None:
    """Update *results* in-place with correctness flags for *step*."""

    if step == "detection_qa":
        resp = _parse_nums(results["detection_qa"].get("response", ""))
        gt = results["detection_qa"].get("ground_truth")
        results["detection_qa"]["correct"] = resp[0] == gt

    elif step == "contour_evaluation_qa":
        resp = _parse_nums(results["contour_evaluation_qa"].get("responses", ""))
        gt = results["contour_evaluation_qa"].get("ground_truth")
        results["contour_evaluation_qa"]["correct"] = resp[0] == gt

    elif step == "contour_revision_qa_revision":
        rev = results["contour_revision_qa"]
        responses = rev.get("responses", {})
        gt = rev.get("ground_truth", {})
        rev["correct"] = True
        for key in ("expansion", "contraction"):
            rev.setdefault(key, {})
            resp = _parse_nums(responses.get(key, ""))
            correct = set(resp) == set(gt.get(key) or [])
            rev[key]["correct"] = correct
            if not correct:
                rev["correct"] = False

    elif step == "contour_revision_qa_revision_result":
        rev = results["contour_revision_qa"]
        resp = _parse_nums(rev.get("responses", {}).get("revision_result", ""))
        gt = rev.get("ground_truth", {}).get("revision_result")
        rev.setdefault("revision_result", {})
        correct = resp[0] == gt
        rev["revision_result"]["correct"] = correct
        if not correct:
            rev["correct"] = False

    else:  # attribute_extraction_qa
        attr = results["attribute_extraction_qa"]
        responses = attr.get("responses", {})
        gt = attr.get("ground_truth", {})
        attr["correct"] = True

        for key in ("distribution", "severity/measurement", "comparison"):
            attr.setdefault(key, {})
            resp = _parse_nums(responses.get(key, ""))
            correct = int(resp[0]) == gt.get(key)
            attr[key]["correct"] = correct
            if not correct:
                attr["correct"] = False

        attr.setdefault("location", {})
        resp = _parse_nums(responses.get("location", ""))
        correct = set(resp) == set(gt.get("location") or [])
        attr["location"]["correct"] = correct
        if not correct:
            attr["correct"] = False
