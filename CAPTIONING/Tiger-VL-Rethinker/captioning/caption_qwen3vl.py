#!/usr/bin/env python3
"""
Caption images using Qwen3VL-4B-Instruct model.

This script processes all images in a directory and generates captions using the Qwen3VL model.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Tuple, Optional, Any, List, Dict
import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
import time

# Caption prompts
CAPTION_PROMPTS = {
    'SIMPLE': "Describe this image in detail.",
    'SHORT': "Write a very short caption for the given image.",
    'LONG': "Write a very long and detailed caption describing the given image as comprehensively as possible."
}

# Supported image extensions
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp', '.tiff', '.tif'}

def _looks_like_path(value: str) -> bool:
    if not value:
        return False
    if value.startswith(("/", "./", "../", "~", "\\\\")):
        return True
    if len(value) >= 3 and value[1] == ":" and value[2] in ("/", "\\"):
        return True
    expanded = os.path.expandvars(os.path.expanduser(value))
    if os.path.exists(expanded):
        return True
    sep_count = value.count("/") + value.count("\\")
    return sep_count > 1

def _sanitize_name(value: str) -> str:
    safe_chars = []
    last_was_sep = False
    for ch in value:
        if ch.isascii() and (ch.isalnum() or ch in "._-"):
            safe_chars.append(ch)
            last_was_sep = False
        else:
            if not last_was_sep:
                safe_chars.append("_")
                last_was_sep = True
    cleaned = "".join(safe_chars).strip("._-")
    return cleaned or "model"


def make_model_safe(model: Optional[str]) -> str:
    raw = (model or "").strip()
    if not raw:
        return "model"

    # Handle HuggingFace model IDs (e.g., "Qwen/Qwen3-VL-4B-Instruct")
    # Take only the part after the last "/"
    if "/" in raw and not _looks_like_path(raw):
        raw = raw.split("/")[-1]

    if _looks_like_path(raw):
        expanded = os.path.expandvars(os.path.expanduser(raw))
        normalized = os.path.normpath(expanded)
        normalized = normalized.replace("\\", "/")
        base = os.path.basename(normalized)
        if base:
            raw = base
    return _sanitize_name(raw)




def get_image_files(directory: str) -> List[Path]:
    """
    Get all image files from a directory.

    Args:
        directory: Path to directory containing images

    Returns:
        List of Path objects for image files
    """
    dir_path = Path(directory)
    if not dir_path.exists():
        raise ValueError(f"Directory does not exist: {directory}")

    image_files = []
    for ext in IMAGE_EXTENSIONS:
        image_files.extend(dir_path.glob(f"*{ext}"))
        image_files.extend(dir_path.glob(f"*{ext.upper()}"))

    return sorted(image_files)


def load_model(model_path: str = "Qwen/Qwen3-VL-4B-Instruct"):
    """
    Load Qwen3VL model and processor.

    Args:
        model_path: Path or HuggingFace model ID

    Returns:
        Tuple of (model, processor)
    """
    print(f"Loading model from {model_path}...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_path)
    print("Model loaded successfully!")
    return model, processor


def generate_caption(
    image_path: str,
    model,
    processor,
    prompt_text: str,
    max_new_tokens: int = 256
) -> str:
    """
    Generate caption for a single image.

    Args:
        image_path: Path to image file
        model: Loaded Qwen3VL model
        processor: Loaded processor
        prompt_text: Prompt text to use
        max_new_tokens: Maximum number of tokens to generate

    Returns:
        Generated caption text
    """
    # Build messages
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]

    # Build the text prompt (NO tokenization here)
    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # Load/prepare vision inputs (this is what actually reads the image)
    image_inputs, video_inputs = process_vision_info(messages)

    # Create final model inputs
    inputs = processor(
        text=[prompt],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    # Generate
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)

    # Trim the prompt tokens and decode only new tokens
    new_tokens = generated_ids[:, inputs.input_ids.shape[1]:]
    output_text = processor.batch_decode(new_tokens, skip_special_tokens=True)

    return output_text[0].strip()


def caption_images(
    dataset_path: str,
    output_path: str,
    prompt_style: str = 'SIMPLE',
    model_path: str = "Qwen/Qwen3-VL-4B-Instruct",
    max_new_tokens: int = 256,
    overwrite: bool = False
):
    """
    Caption all images in a directory.

    Args:
        dataset_path: Path to directory containing images
        output_path: Path to save output JSON file
        prompt_style: Style of prompt to use (SIMPLE, SHORT, LONG)
        model_path: Path or HuggingFace model ID for Qwen3VL
        max_new_tokens: Maximum number of tokens to generate
        overwrite: Whether to overwrite existing captions
    """
    start_time = time.time()

    # Validate prompt style
    if prompt_style not in CAPTION_PROMPTS:
        raise ValueError(f"Invalid prompt_style: {prompt_style}. Must be one of {list(CAPTION_PROMPTS.keys())}")

    prompt_text = CAPTION_PROMPTS[prompt_style]
    print(f"Using prompt style: {prompt_style}")
    print(f"Prompt: {prompt_text}\n")

    # Get all image files
    print(f"Scanning directory: {dataset_path}")
    image_files = get_image_files(dataset_path)
    print(f"Found {len(image_files)} images\n")

    if len(image_files) == 0:
        print("No images found. Exiting.")
        return

    # Load existing results if available
    results = {}
    if os.path.exists(output_path):
        with open(output_path, 'r') as f:
            results = json.load(f)
        print(f"Loaded {len(results)} existing captions from {output_path}")

    # Load model
    model, processor = load_model(model_path)

    # Process each image
    print(f"\nProcessing images...")
    success_count = 0
    error_count = 0

    for image_path in tqdm(image_files, desc="Captioning"):
        image_name = image_path.name

        # Skip if already processed and not overwriting
        if image_name in results and not overwrite:
            continue

        try:
            caption = generate_caption(
                image_path=str(image_path),
                model=model,
                processor=processor,
                prompt_text=prompt_text,
                max_new_tokens=max_new_tokens
            )

            # Save caption
            results[image_name] = caption
            success_count += 1

            # Save after each successful caption (incremental saving)
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

        except Exception as e:
            print(f"\nError processing {image_name}: {e}")
            error_count += 1
            continue

    # Final save
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Print summary
    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Captioning complete!")
    print(f"{'='*60}")
    print(f"Total images: {len(image_files)}")
    print(f"Successfully captioned: {success_count}")
    print(f"Errors: {error_count}")
    print(f"Results saved to: {output_path}")
    print(f"Total time: {elapsed:.2f}s ({elapsed/60:.2f}m)")
    if success_count > 0:
        print(f"Average time per image: {elapsed/success_count:.2f}s")


def main():
    parser = argparse.ArgumentParser(
        description="Caption images using Qwen3VL-4B-Instruct model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Caption images with SIMPLE prompt
  python caption_qwen3vl.py \\
    --dataset-path /path/to/images \\
    --output-path ./captions.json \\
    --prompt-style SIMPLE

  # Caption images with LONG prompt and custom model
  python caption_qwen3vl.py \\
    --dataset-path /path/to/images \\
    --output-path ./captions_long.json \\
    --prompt-style LONG \\
    --model-path /path/to/local/model

  # Overwrite existing captions
  python caption_qwen3vl.py \\
    --dataset-path /path/to/images \\
    --output-path ./captions.json \\
    --prompt-style SHORT \\
    --overwrite
        """
    )

    # Required arguments
    parser.add_argument(
        '--dataset-path',
        type=str,
        required=True,
        help='Path to directory containing images to caption'
    )
    parser.add_argument(
        '--output-path',
        type=str,
        required=True,
        help='Path to save output JSON file with captions'
    )

    # Optional arguments
    parser.add_argument(
        '--prompt-style',
        type=str,
        default='SIMPLE',
        choices=['SIMPLE', 'SHORT', 'LONG'],
        help='Style of prompt to use (default: SIMPLE)'
    )
    parser.add_argument(
        '--model-path',
        type=str,
        default='Qwen/Qwen3-VL-4B-Instruct',
        help='Path or HuggingFace model ID for Qwen3VL (default: Qwen/Qwen3-VL-4B-Instruct)'
    )
    parser.add_argument(
        '--max-new-tokens',
        type=int,
        default=1024,
        help='Maximum number of tokens to generate (default: 256)'
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing captions'
    )

    args = parser.parse_args()
    
    
    model_safe = make_model_safe(args.model_path)
    out_dir = os.path.join(args.output_path, model_safe)
    os.makedirs(out_dir, exist_ok=True)
    output_path = os.path.join(out_dir, f"{model_safe}_{args.prompt_style.lower()}.json")
    print(f"Saving outputs to {output_path}...")

    # Run captioning
    caption_images(
        dataset_path=args.dataset_path,
        output_path=output_path,
        prompt_style=args.prompt_style,
        model_path=args.model_path,
        max_new_tokens=args.max_new_tokens,
        overwrite=args.overwrite
    )


if __name__ == "__main__":
    main()
