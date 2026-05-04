import re
import time
import torch
import base64
from PIL import Image as PILImage
from google.genai import types
from google.genai import errors as genai_errors

# ---------------------------------------------------------------------------
# Token pricing (USD per 1M tokens, standard tier)
# Gemini: "output" covers both response + thinking tokens.
# OpenAI: "output" covers both completion + reasoning tokens.
# Vertex AI / Azure pricing may differ – update values as needed.
# ---------------------------------------------------------------------------
MODEL_PRICING = {
    # Gemini  (https://ai.google.dev/gemini-api/docs/pricing)
    "gemini-3.1-flash-lite-preview": {"input": 0.25,  "cached_input": 0.025,  "output": 1.50},
    "gemini-3.1-pro-preview":        {"input": 2.00,  "cached_input": 0.20,   "output": 12.00},
    # OpenAI / Azure  (https://developers.openai.com/api/docs/pricing)
    "gpt-5.4-nano": {"input": 0.20,  "cached_input": 0.02,   "output": 1.25},
    "gpt-5.4-mini": {"input": 0.75,  "cached_input": 0.075,  "output": 4.50},
    "gpt-5.4":      {"input": 2.50,  "cached_input": 0.25,   "output": 15.00},
}


def _lookup_pricing(model_id: str) -> dict:
    """Find best-matching pricing entry by longest substring match."""
    model_lower = model_id.lower()
    best, best_len = {}, 0
    for key, val in MODEL_PRICING.items():
        if key in model_lower and len(key) > best_len:
            best, best_len = val, len(key)
    return best


def calculate_cost(token_info: dict, model_id: str) -> float:
    """Estimate USD cost for a single API call.

    Accounts for cached input tokens at the lower cached_input price.
    Uncached input = total input - cached input.
    """
    pricing = _lookup_pricing(model_id)
    if not pricing:
        return 0.0
    total_input = token_info.get("input_tokens", 0)
    cached_input = token_info.get("cached_input_tokens", 0)
    uncached_input = total_input - cached_input

    input_price = pricing.get("input", 0)
    cached_price = pricing.get("cached_input", input_price)
    output_price = pricing.get("output", 0)

    input_cost = (uncached_input * input_price + cached_input * cached_price) / 1_000_000
    output_cost = token_info.get("output_tokens", 0) * output_price / 1_000_000
    return input_cost + output_cost


def summarize_token_usage(token_log: list, model_id: str) -> dict:
    """Aggregate a list of per-call token_info dicts into a summary."""
    totals = {
        "model_id": model_id,
        "num_calls": 0,
        "input_tokens": 0,
        "cached_input_tokens": 0,
        "output_tokens": 0,
        "thinking_tokens": 0,
        "total_tokens": 0,
        "estimated_cost_usd": 0.0,
    }
    per_call = []
    for info in token_log:
        if info is None:
            continue
        totals["input_tokens"] += info.get("input_tokens", 0)
        totals["cached_input_tokens"] += info.get("cached_input_tokens", 0)
        totals["output_tokens"] += info.get("output_tokens", 0)
        totals["thinking_tokens"] += info.get("thinking_tokens", 0)
        totals["total_tokens"] += info.get("total_tokens", 0)
        totals["estimated_cost_usd"] += info.get("cost_usd", 0.0)
        totals["num_calls"] += 1
        per_call.append(info)
    totals["per_call"] = per_call
    return totals


def _extract_answer_number(text: str) -> str:
    """Extract the option number(s) following 'Answer:' from any model response.

    Steps:
      1. Strip <think>...</think> reasoning blocks.
      2. Find all 'Answer: <digits>' occurrences and take the last one
         (handles duplicated output seen in some models like InternVL).
      3. Fallback: if the entire response is just a bare number (e.g. "4" or "4."),
         extract and return that digit string.
      4. Otherwise return the stripped original text as fallback.
    """
    # Remove reasoning blocks
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    # Unwrap \boxed{N} → N  (models like Qwen/DeepSeek emit this)
    text = re.sub(r'\\boxed\{([\d,\s]+)\}', r'\1', text)
    # Match "Answer: 3", "Answer: 1,3", "Answer: (3)", "Answer: (1,3)"
    # Take the LAST match to handle models that emit multiple "Answer:" lines
    matches = re.findall(r'[Aa]nswer\s*:\s*\(?(\d+(?:\s*,\s*\d+)*)\)?', text)
    if matches:
        nums = re.findall(r'\d+', matches[-1])
        return ','.join(nums) if nums else text.strip()
    # Fallback: whole response is just a bare number(s), e.g. "4" or "4." or "1, 3"
    bare = re.fullmatch(r'[\d,\s.]+', text.strip())
    if bare:
        nums = re.findall(r'\d+', bare.group())
        if nums:
            return ','.join(nums)
    return text.strip()

def parse_response(raw_response: str, model_id: str) -> str:
    """Parse raw model output into a clean answer string (e.g. '2' or '1,3').

    Add model-specific pre-processing here as new models are integrated.
    The final step for every model is _extract_answer_number().
    """
    model_id_lower = model_id.lower()

    if 'glm' in model_id_lower:
        # Remove GLM special box tokens: <|begin_of_box|>2<|end_of_box|>
        raw_response = re.sub(
            r'<\|begin_of_box\|>(.*?)<\|end_of_box\|>', r'\1',
            raw_response, flags=re.DOTALL
        )

    return _extract_answer_number(raw_response)

def load_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def inference_vllm(model, sampling_params, query, img_path_lst, system_message, chat_history, chat_kwargs=None):
    if chat_kwargs is None:
        chat_kwargs = {}

    # Check if using DummyLLM (for debugging)
    if hasattr(model, '__class__') and model.__class__.__name__ == 'DummyLLM':
        print(f"[DEBUG] DummyLLM inference called")
        print(f"[DEBUG] Query: {query[:100]}...")
        print(f"[DEBUG] Images: {len(img_path_lst)}")
        response = "1"  # Dummy answer
        chat_history.append([query, img_path_lst, response])
        return response, chat_history, None
    
    conversation = []

    if system_message is not None:
        conversation.append({"role": "system", "content": system_message})

    # append chat history, if applicable
    for idx, turn in enumerate(chat_history):
        query_t, query_i, response_t = turn[0], turn[1], turn[2]
        conversation.extend([
            {"role": "user",
             "content": [{"type": "text", "text": query_t}] + [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{load_image(img_path)}"}} for img_path in query_i]},
            {"role": "assistant",
             "content": [{"type": "text", "text": response_t}]}
        ])

    # append current query
    conversation.append(
        {"role": "user",
         "content": [{"type": "text", "text": query}] + [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{load_image(img_path)}"}} for img_path in img_path_lst]}
    )

    with torch.inference_mode():
        outputs = model.chat(conversation, sampling_params=sampling_params, use_tqdm=False,
                             **chat_kwargs)

    response = outputs[0].outputs[0].text

    chat_history.append([query, img_path_lst, response])

    return response, chat_history, None

def inference_hulu_med(model, processor, query, img_path_lst, system_message, chat_history):
    conversation = []

    # append system message
    if system_message is not None:
        conversation.append({"role": "system", "content": system_message})

    # append chat history, if applicable
    img_path_lst_hist = []
    for idx, turn in enumerate(chat_history):
        query_t, query_i, response_t = turn[0], turn[1], turn[2]
        img_path_lst_hist.extend(query_i)
        conversation.extend([
            {"role": "user", "content": [{"type": "text", "text": query_t}] + [{"type": "image", "image": {"image_path": img_path}} for img_path in query_i]},
            {"role": "assistant",
             "content": [{"type": "text", "text": response_t}]}
        ])

    # append current query
    conversation.append({"role": "user",
                         "content": [{"type": "text", "text": query}] + [{"type": "image", "image": {"image_path": img_path}} for img_path in img_path_lst]
                         })

    inputs = processor(
        conversation=conversation,
        add_system_prompt=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )

    inputs = {k: v.cuda() if isinstance(v, torch.Tensor) else v 
          for k, v in inputs.items()}

    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

    output_ids = model.generate(**inputs, max_new_tokens=8192)
    response = processor.batch_decode(
        output_ids,
        skip_special_tokens=True,
        use_think=True,
    )[0].strip()

    chat_history.append([query, img_path_lst, response])

    return response, chat_history, None

def _gemini_send_with_backoff(chat, message,
                              max_retries: int = 6,
                              base_delay: float = 30.0,
                              max_delay: float = 300.0):
    """Call ``chat.send_message`` with explicit 429-aware exponential backoff.

    The google-genai SDK's internal tenacity retry surrenders quickly on
    persistent 429s. This wrapper adds a longer outer loop: when a 429 is
    raised, sleep for ``min(base_delay * 2**attempt, max_delay)`` seconds and
    retry, up to ``max_retries`` total attempts. Other errors propagate
    immediately.
    """
    delay = base_delay
    for attempt in range(max_retries):
        try:
            return chat.send_message(message)
        except genai_errors.ClientError as e:
            status = getattr(e, "status_code", None) or getattr(e, "code", None)
            if status != 429:
                raise
            if attempt == max_retries - 1:
                raise
            print(f"  [gemini-backoff] 429 hit, sleeping {delay:.0f}s "
                  f"(attempt {attempt + 1}/{max_retries})", flush=True)
            time.sleep(delay)
            delay = min(delay * 2, max_delay)


def inference_gemini(client, model_id, query, img_path_lst, system_message, chat_history):
    """Inference via Google Gemini API.

    Args:
        client:         google.genai.Client instance
        model_id:       Gemini model name (e.g. "gemini-2.0-flash-exp")
        query:          Text prompt for the current turn
        img_path_lst:   List of image file paths for the current turn
        system_message: System instruction
        chat_history:   List of [query, img_paths, response] turns
    """
    if hasattr(client, '__class__') and client.__class__.__name__ == 'DummyLLM':
        print(f"[DEBUG] DummyLLM inference called")
        print(f"[DEBUG] Query: {query[:100]}...")
        print(f"[DEBUG] Images: {len(img_path_lst)}")
        response = "1"
        chat_history.append([query, img_path_lst, response])
        return response, chat_history, None

    history = []
    for idx, turn in enumerate(chat_history):
        query_t, query_i, response_t = turn[0], turn[1], turn[2]
        user_parts = [query_t]
        for img_path in query_i:
            user_parts.append(PILImage.open(img_path).convert("RGB"))
        history.append(types.UserContent(user_parts))
        history.append(types.ModelContent(response_t))

    chat = client.chats.create(
        model=model_id,
        config=types.GenerateContentConfig(
            system_instruction=system_message,
            thinking_config=types.ThinkingConfig(thinking_level="medium"),
            max_output_tokens=8192,
            temperature=1.0,
        ),
        history=history if len(history) else None,
    )

    current_message = [query]
    for img_path in img_path_lst:
        current_message.append(PILImage.open(img_path).convert("RGB"))

    output = _gemini_send_with_backoff(chat, current_message)

    thinking_tokens = getattr(output.usage_metadata, 'thoughts_token_count', 0) or 0
    candidates_tokens = output.usage_metadata.candidates_token_count or 0
    cached_tokens = getattr(output.usage_metadata, 'cached_content_token_count', 0) or 0
    token_info = {
        'input_tokens': output.usage_metadata.prompt_token_count,
        'cached_input_tokens': cached_tokens,
        'output_tokens': candidates_tokens + thinking_tokens,
        'thinking_tokens': thinking_tokens,
        'total_tokens': output.usage_metadata.total_token_count,
    }
    token_info['cost_usd'] = calculate_cost(token_info, model_id)

    response = output.text

    chat_history.append([query, img_path_lst, response])
    return response, chat_history, token_info


def inference_gpt(client, model_id, query, img_path_lst, system_message, chat_history):
    """Inference via OpenAI / Azure OpenAI chat completions API.

    Args:
        client:         OpenAI or AzureOpenAI client instance
        model_id:       Model / deployment name (e.g. "gpt-4o")
        query:          Text prompt for the current turn
        img_path_lst:   List of image file paths for the current turn
        system_message: System instruction
        chat_history:   List of [query, img_paths, response] turns
    """
    if hasattr(client, '__class__') and client.__class__.__name__ == 'DummyLLM':
        print(f"[DEBUG] DummyLLM inference called")
        print(f"[DEBUG] Query: {query[:100]}...")
        print(f"[DEBUG] Images: {len(img_path_lst)}")
        response = "1"
        chat_history.append([query, img_path_lst, response])
        return response, chat_history, None

    from mimetypes import guess_type

    def _encode_image_url(img_path):
        mime_type, _ = guess_type(img_path)
        if mime_type is None:
            mime_type = 'image/png'
        return f"data:{mime_type};base64,{load_image(img_path)}"

    conversation = []
    if system_message is not None:
        conversation.append({"role": "system", "content": system_message})

    for idx, turn in enumerate(chat_history):
        query_t, query_i, response_t = turn[0], turn[1], turn[2]
        conversation.extend([
            {"role": "user",
             "content": [{"type": "text", "text": query_t}] +
                        [{"type": "image_url", "image_url": {"url": _encode_image_url(img_path)}} for img_path in query_i]},
            {"role": "assistant", "content": [{"type": "text", "text": response_t}]},
        ])

    conversation.append(
        {"role": "user",
         "content": [{"type": "text", "text": query}] +
                    [{"type": "image_url", "image_url": {"url": _encode_image_url(img_path)}} for img_path in img_path_lst]}
    )

    output = client.chat.completions.create(
        messages=conversation,
        max_completion_tokens=8192,
        temperature=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        model=model_id,
        reasoning_effort = "medium"
    )

    reasoning_tokens = 0
    cached_tokens = 0
    if hasattr(output.usage, 'completion_tokens_details') and output.usage.completion_tokens_details:
        reasoning_tokens = getattr(output.usage.completion_tokens_details, 'reasoning_tokens', 0) or 0
    if hasattr(output.usage, 'prompt_tokens_details') and output.usage.prompt_tokens_details:
        cached_tokens = getattr(output.usage.prompt_tokens_details, 'cached_tokens', 0) or 0

    token_info = {
        'input_tokens': output.usage.prompt_tokens,
        'cached_input_tokens': cached_tokens,
        'output_tokens': output.usage.completion_tokens,
        'thinking_tokens': reasoning_tokens,
        'total_tokens': output.usage.total_tokens,
    }
    token_info['cost_usd'] = calculate_cost(token_info, model_id)

    response = output.choices[0].message.content

    chat_history.append([query, img_path_lst, response])

    return response, chat_history, token_info


def inference_vllms(model_id: str):
    """Return the appropriate inference function based on model_id.

        fn = inference_vllms(model_id)
        response, history, token_info = fn(model, params, query, imgs, system_msg, history)
    """
    model_id_lower = model_id.lower()

    if 'gemini' in model_id_lower:
        return inference_gemini

    if 'gpt' in model_id_lower:
        return inference_gpt

    model_map = {
        'hulu-med': inference_hulu_med,
    }

    for keyword, func in model_map.items():
        if keyword in model_id_lower:
            return func

    return inference_vllm