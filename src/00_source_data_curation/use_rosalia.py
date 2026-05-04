import os
import cv2
import torch
import torch.nn.functional as F
import transformers
from torchvision.utils import save_image

# Core architecture based on LISA (Reasoning Segmentation via LLM)
from model.LISA import LISAForCausalLM
from model.llava import conversation as conversation_lib
from model.llava.mm_utils import tokenizer_image_token

# Vision and Segmentation modules
from model.segment_anything.utils.transforms import ResizeLongestSide
from transformers import CLIPImageProcessor

# Project utilities
from utils.utils import (
    DEFAULT_IM_END_TOKEN, 
    DEFAULT_IM_START_TOKEN, 
    DEFAULT_IMAGE_TOKEN, 
    IMAGE_TOKEN_INDEX
)

def preprocess(
    x,
    pixel_mean=torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1),
    pixel_std=torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1),
    img_size=1024,
) -> torch.Tensor:
    """Normalize pixel values and pad to a square input."""
    # Normalize colors
    x = (x - pixel_mean) / pixel_std
    # Pad
    h, w = x.shape[-2:]
    padh = img_size - h
    padw = img_size - w
    x = F.pad(x, (0, padw, 0, padh))
    return x


def initialize_model():
    """Initialize tokenizer, model, and image processors."""
    # Initialize tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        "xinlai/LISA-7B-v1",
        cache_dir=None,
        model_max_length=512,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token
    seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]

    # Initialize model
    torch_dtype = torch.bfloat16
    vision_tower_name = "openai/clip-vit-large-patch14"
    local_rank = 0
    image_size = 1024
    
    model = LISAForCausalLM.from_pretrained(
        "checkone/ROSALIA-7B-v1",
        low_cpu_mem_usage=True,
        vision_tower=vision_tower_name,
        seg_token_idx=seg_token_idx,
        torch_dtype=torch_dtype,
    )

    # Configure model tokens
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    # Initialize vision modules
    model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(dtype=torch_dtype, device=local_rank)

    # Move model to GPU and set precision
    model = model.bfloat16().cuda()
    model.eval()

    # Initialize image processors
    clip_image_processor = CLIPImageProcessor.from_pretrained(model.config.vision_tower)
    transform = ResizeLongestSide(image_size)

    return model, tokenizer, clip_image_processor, transform


def prepare_inputs(instruction, image_path, tokenizer, clip_image_processor, transform):
    """Prepare input images and prompt for inference."""
    # Build conversation prompt
    conv_type = "llava_v1"
    use_mm_start_end = True
    
    conv = conversation_lib.conv_templates[conv_type].copy()
    conv.messages = []
    
    prompt = DEFAULT_IMAGE_TOKEN + "\n" + instruction
    if use_mm_start_end:
        replace_token = (
            DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        )
        prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, replace_token)
    
    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], "")
    prompt = conv.get_prompt()

    # Load and preprocess image
    image_np = cv2.imread(image_path)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    original_size_list = [image_np.shape[:2]]

    # Process image for CLIP
    image_clip = (
        clip_image_processor.preprocess(image_np, return_tensors="pt")["pixel_values"][0]
        .unsqueeze(0)
        .cuda()
        .bfloat16()
    )

    # Process image for SAM
    image_resized = transform.apply_image(image_np)
    resize_list = [image_resized.shape[:2]]
    
    image = (
        preprocess(torch.from_numpy(image_resized).permute(2, 0, 1).contiguous())
        .unsqueeze(0)
        .cuda()
        .bfloat16()
    )

    # Tokenize prompt
    input_ids = tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
    input_ids = input_ids.unsqueeze(0).cuda()

    return image_clip, image, input_ids, resize_list, original_size_list


def run_inference(model, tokenizer, image_clip, image, input_ids, resize_list, original_size_list):
    """Run model inference."""
    output_ids, pred_masks = model.evaluate(
        image_clip,
        image,
        input_ids,
        resize_list,
        original_size_list,
        max_new_tokens=512,
        tokenizer=tokenizer,
    )

    # Convert output_ids to text
    # Filter out invalid token IDs (negative values and special tokens like 32001, 32002, etc.)
    output_ids = output_ids[0][output_ids[0] != IMAGE_TOKEN_INDEX]
    text_output = tokenizer.decode(output_ids, skip_special_tokens=False)
    text_output = text_output.replace("\n", "").replace("  ", " ")
    text_output = text_output.split("ASSISTANT:")[-1].split("</s>")[0].strip()
    
    return pred_masks, text_output


def save_results(pred_masks, output_path='./example/pred_mask_example.png'):
    """Save prediction masks to file."""
    final_pred = (pred_masks[0] > 0).int()
    save_image(final_pred.float(), output_path)


def main():
    
    # Input
    instruction = 'Segment the opacity in the right lung.'
    image_path = './example/cxr_example.png'

    # Initialize model
    model, tokenizer, clip_image_processor, transform = initialize_model()

    # Prepare inputs
    image_clip, image, input_ids, resize_list, original_size_list = prepare_inputs(
        instruction, image_path, tokenizer, clip_image_processor, transform
    )

    # Run inference
    pred_masks, output_text = run_inference(
        model, tokenizer, image_clip, image, input_ids, resize_list, original_size_list
    )

    # Print generated text
    print("Generated text:", output_text)

    # Save results
    save_results(pred_masks)

if __name__ == "__main__":
    main()