#!/usr/bin/env python3
"""
Caption images using Qwen3VL via vLLM server.

This script processes all images in a directory and generates captions using the Qwen3VL model
running on a vLLM server.
"""

import argparse
import json
import os
import base64
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm
import time
import requests

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


def make_model_safe(model: str) -> str:
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
    """Get all image files from a directory."""
    dir_path = Path(directory)
    if not dir_path.exists():
        raise ValueError(f"Directory does not exist: {directory}")

    image_files = []
    for ext in IMAGE_EXTENSIONS:
        image_files.extend(dir_path.glob(f"*{ext}"))
        image_files.extend(dir_path.glob(f"*{ext.upper()}"))

    return sorted(image_files)


def encode_image_base64(image_path: str) -> str:
    """Encode image to base64 string."""
    with open(image_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')


def generate_caption_server(
    server_url: str,
    image_path: str,
    prompt_text: str,
    max_tokens: int = 256
) -> str:
    """
    Generate caption using vLLM server API.

    Args:
        server_url: Base URL of vLLM server (e.g., "http://localhost:8000")
        image_path: Path to image file
        prompt_text: Prompt text to use
        max_tokens: Maximum number of tokens to generate

    Returns:
        Generated caption text
    """
    # Encode image to base64
    image_base64 = encode_image_base64(image_path)

    # Build messages in OpenAI format
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_base64}"
                    }
                },
                {
                    "type": "text",
                    "text": prompt_text
                }
            ]
        }
    ]

    # Make API call
    response = requests.post(
        f"{server_url}/v1/chat/completions",
        json={
            "model": "Qwen/Qwen3-VL-4B-Instruct",  # This can be any string when using vLLM
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.7
        },
        headers={"Content-Type": "application/json"}
    )

    if response.status_code != 200:
        raise Exception(f"Server error: {response.status_code} - {response.text}")

    result = response.json()
    return result['choices'][0]['message']['content'].strip()


def caption_images(
    dataset_path: str,
    output_dir: str,
    prompt_style: str,
    server_url: str,
    model_name: str = "Qwen3-VL-4B-Instruct",
    max_tokens: int = 256,
    overwrite: bool = False
):
    """Caption all images in a directory using vLLM server."""
    start_time = time.time()

    # Validate prompt style
    if prompt_style not in CAPTION_PROMPTS:
        raise ValueError(f"Invalid prompt_style: {prompt_style}. Must be one of {list(CAPTION_PROMPTS.keys())}")

    prompt_text = CAPTION_PROMPTS[prompt_style]
    print(f"Using prompt style: {prompt_style}")
    print(f"Prompt: {prompt_text}")
    print(f"Server URL: {server_url}\n")

    # Get all image files
    print(f"Scanning directory: {dataset_path}")
    image_files = get_image_files(dataset_path)
    print(f"Found {len(image_files)} images\n")

    if len(image_files) == 0:
        print("No images found. Exiting.")
        return

    # Construct output path
    model_safe = make_model_safe(model_name)
    out_dir = os.path.join(output_dir, "captions", model_safe)
    os.makedirs(out_dir, exist_ok=True)
    output_path = os.path.join(out_dir, f"{model_safe}_{prompt_style.lower()}.json")

    print(f"Output directory: {out_dir}")
    print(f"Output file: {output_path}\n")

    # Load existing results if available
    results = {}
    if os.path.exists(output_path):
        with open(output_path, 'r') as f:
            results = json.load(f)
        print(f"Loaded {len(results)} existing captions from {output_path}\n")

    # Process each image
    print(f"Processing images...")
    success_count = 0
    error_count = 0

    for image_path in tqdm(image_files, desc="Captioning"):
        image_name = image_path.name

        # Skip if already processed and not overwriting
        if image_name in results and not overwrite:
            continue

        try:
            caption = generate_caption_server(
                server_url=server_url,
                image_path=str(image_path),
                prompt_text=prompt_text,
                max_tokens=max_tokens
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
        description="Caption images using Qwen3VL via vLLM server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Caption images with SIMPLE prompt
  python caption_qwen3vl_server.py \\
    --dataset-path /path/to/images \\
    --output-dir ./output \\
    --server-url http://localhost:8000 \\
    --prompt-style SIMPLE

  # Caption images with LONG prompt
  python caption_qwen3vl_server.py \\
    --dataset-path /path/to/images \\
    --output-dir ./output \\
    --server-url http://10.1.64.88:8000 \\
    --prompt-style LONG \\
    --max-tokens 512
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
        '--server-url',
        type=str,
        required=True,
        help='Base URL of vLLM server (e.g., http://localhost:8000)'
    )

    # Optional arguments
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./output',
        help='Directory to save output files (default: ./output)'
    )
    parser.add_argument(
        '--prompt-style',
        type=str,
        default='SIMPLE',
        choices=['SIMPLE', 'SHORT', 'LONG'],
        help='Style of prompt to use (default: SIMPLE)'
    )
    parser.add_argument(
        '--model-name',
        type=str,
        default='Qwen3-VL-4B-Instruct',
        help='Model name for output folder (default: Qwen3-VL-4B-Instruct)'
    )
    parser.add_argument(
        '--max-tokens',
        type=int,
        default=256,
        help='Maximum number of tokens to generate (default: 256)'
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing captions'
    )

    args = parser.parse_args()

    # Run captioning
    caption_images(
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        prompt_style=args.prompt_style,
        server_url=args.server_url,
        model_name=args.model_name,
        max_tokens=args.max_tokens,
        overwrite=args.overwrite
    )


if __name__ == "__main__":
    main()
