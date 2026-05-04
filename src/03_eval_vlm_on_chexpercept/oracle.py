"""
Oracle injection utilities for the CheXpercept evaluation pipeline.

Oracle modes
------------
explicit : Prepend a ground-truth hint to the next question prompt.
implicit : Silently replace the prior model response in chat_history with the
           ground-truth option string so downstream turns reason from correct context.
none     : No oracle injection.
"""
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Chat-history bookkeeping
# ---------------------------------------------------------------------------

def ensure_chat_history_idx(results: Dict) -> Dict[str, int]:
    """Return (and lazily create) results['_chat_history_idx']."""
    if not isinstance(results.get("_chat_history_idx"), dict):
        results["_chat_history_idx"] = {}
    return results["_chat_history_idx"]


def patch_chat_history_response(
    chat_history: List, idx: Optional[int], oracle_response: str
) -> None:
    """Replace the model response at chat_history[idx] with oracle_response.

    The original raw response is preserved at index 3 so visualization tools
    can display both the raw and oracle-patched outputs side by side.
    """
    if not isinstance(idx, int) or not (0 <= idx < len(chat_history)):
        return
    turn = chat_history[idx]
    if not isinstance(turn, list) or len(turn) < 3:
        return
    if len(turn) == 3:
        turn.append(turn[2])  # stash original before overwriting
    turn[2] = oracle_response


def format_oracle_option_response(answer_index: Any) -> Optional[str]:
    """Format a ground-truth answer index as 'Answer: N[,M,…]'."""
    if answer_index is None:
        return None
    if isinstance(answer_index, list):
        nums = sorted(int(x) for x in answer_index if x is not None)
        return ("Answer: " + ",".join(str(n) for n in nums)) if nums else None
    try:
        return f"Answer: {int(answer_index)}"
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Oracle text generators
# ---------------------------------------------------------------------------

def _join_point_colors(colors: List[str]) -> str:
    """Format a list of point color names into natural language."""
    if len(colors) == 1:
        return f"{colors[0]} point"
    if len(colors) == 2:
        return f"{colors[0]} and {colors[1]} points"
    return ", ".join(colors[:-1]) + f", and {colors[-1]} points"


def _expansion_contraction_oracle_lines(contour_qa: Dict, keys: List[str]) -> List[str]:
    lines = []
    for key in keys:
        qa = contour_qa.get(key, {})
        op = key.split("_")[-1]  # 'expansion' or 'contraction'
        if qa.get("answer") == "None":
            lines.append(
                f"Ground Truth Provided: The lesion mask is definitively not needed "
                f"to be revised using {op} operation."
            )
        else:
            colors = [
                qa["answer_options"][i - 1]["color"]
                for i in qa.get("answer_index", [])
            ]
            lines.append(
                f"Ground Truth Provided: The lesion mask is definitively needed "
                f"to be revised using {op} operation at {_join_point_colors(colors)}."
            )
    return lines


def generate_oracle_text_for_revision_result(contour_qa: Dict, results: Dict) -> str:
    """Oracle text injected before revision_result QA.

    Informs the model which expansion/contraction operations were correct.
    """
    lines = _expansion_contraction_oracle_lines(
        contour_qa,
        ["contour_revision_qa_expansion", "contour_revision_qa_contraction"],
    )
    return (
        "\n".join(lines)
        + "\nThese are the absolute facts. Disregard your previous answer and "
        "evaluate the following questions strictly based on these facts."
    )


def generate_oracle_text_for_revision_required(contour_qa: Dict, results: Dict) -> str:
    """Oracle text injected before attribute_extraction when qa_path='revision_required'.

    Informs the model which revised mask was the correct one.
    """
    qa = contour_qa.get("contour_revision_qa_revision_result", {})
    idx = qa.get("answer_index", 0)
    ordinal = {1: "1st", 2: "2nd", 3: "3rd"}.get(idx, f"{idx}th")
    return (
        f"Ground Truth Provided: The lesion mask reflecting the modifications "
        f"is the {ordinal} mask.\n"
        "This is an absolute fact. Disregard your previous answer and evaluate "
        "the following questions strictly based on this fact."
    )


# ---------------------------------------------------------------------------
# Main oracle dispatcher
# ---------------------------------------------------------------------------

def return_oracle_answer(qa: Dict, results: Dict, step: str) -> Optional[str]:
    """Return oracle prompt text when a prior step was incorrect; else None."""

    if step == "contour_evaluation_qa":
        if results.get("detection_qa", {}).get("correct", False):
            return None
        return (
            "Ground Truth Provided: The lesion is definitively present in this image. "
            "This is an absolute fact. Disregard your previous answer and evaluate the "
            "following questions strictly based on the confirmed presence of the lesion."
        )

    if step == "contour_revision_qa":
        if results.get("contour_evaluation_qa", {}).get("correct", False):
            return None
        return (
            "Ground Truth Provided: The lesion mask is definitively needed to be revised "
            "in this image. This is an absolute fact. Disregard your previous answer and "
            "evaluate the following questions strictly based on the confirmed need of revision."
        )

    if step == "contour_revision_qa_revision_result":
        if results.get("contour_revision_qa", {}).get("correct", False):
            return None
        return generate_oracle_text_for_revision_result(qa.get("contour_qa", {}), results)

    if step == "attribute_extraction_qa":
        qa_path = results.get("qa_path")
        if qa_path == "revision_free":
            if results.get("contour_evaluation_qa", {}).get("correct", False):
                return None
            return (
                "Ground Truth Provided: The lesion mask is definitively not needed to be "
                "revised in this image. This is an absolute fact. Disregard your previous "
                "answer and evaluate the following questions strictly based on the confirmed "
                "no need of revision of the lesion mask."
            )
        if qa_path == "revision_required":
            if results.get("contour_revision_qa", {}).get("revision_result", {}).get("correct", False):
                return None
            return generate_oracle_text_for_revision_required(qa.get("contour_qa", {}), results)

    return None
