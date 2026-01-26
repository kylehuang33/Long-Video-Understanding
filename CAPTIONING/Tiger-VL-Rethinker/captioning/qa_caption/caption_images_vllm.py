#!/usr/bin/env python3
"""
Caption images using Qwen3VL via vLLM server.
This script only does captioning - use qa_with_captions_vllm.py for QA.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Set, Optional
import pandas as pd
from tqdm import tqdm
import time
import base64
import requests


# Caption prompts (same as caption_qwen3vl.py)
CAPTION_PROMPTS = {
    'SIMPLE': "Describe this image in detail.",
    'SHORT': "Write a very short caption for the given image.",
    'LONG': "Write a very long and detailed caption describing the given image as comprehensively as possible."
}


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


def encode_image_base64(image_path: str) -> str:
    """Encode image to base64 string."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")



def caption_image_vllm(
    server_url: str,
    image_path: str,
    model: str,
    prompt: str,
    max_tokens: int = 512
) -> str:
    """
    Caption a single image using vLLM server.

    Args:
        server_url: vLLM server URL
        image_path: Path to image file
        model: Model name
        prompt: Caption prompt
        max_tokens: Maximum tokens to generate

    Returns:
        Generated caption
    """
    image_base64 = encode_image_base64(image_path)
    content = [
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{image_base64}"
            }
        },
        {
            "type": "text",
            "text": prompt
        }
    ]

    messages = [{
        "role": "user",
        "content": content
    }]

    response = requests.post(
        f"{server_url}/v1/chat/completions",
        json={
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.0
        },
        headers={"Content-Type": "application/json"}
    )

    if response.status_code != 200:
        raise Exception(f"Server error: {response.status_code} - {response.text}")

    result = response.json()
    return result['choices'][0]['message']['content'].strip()


def collect_all_images(df: pd.DataFrame, dataset_root: str) -> Set[str]:
    """Collect all unique image paths from the dataframe."""
    all_images = set()
    for idx, row in df.iterrows():
        image_array = row['image']
        if isinstance(image_array, str):
            image_paths = [image_array]
        else:
            image_paths = list(image_array)

        for img in image_paths:
            full_path = os.path.join(dataset_root, img)
            if os.path.exists(full_path):
                all_images.add(full_path)

    return all_images


def caption_images(
    parquet_path: str,
    dataset_root: str,
    output_dir: str,
    vllm_url: str = "http://localhost:8000/v1",
    model: str = "Qwen/Qwen3-VL-4B-Instruct",
    prompt_style: str = 'SIMPLE',
    max_tokens: int = 512,
    overwrite: bool = False
):
    """
    Caption all images from parquet file.

    Args:
        parquet_path: Path to parquet file
        dataset_root: Root directory for image paths
        output_dir: Base output directory
        vllm_url: vLLM server URL
        model: Model name for captioning
        prompt_style: Caption prompt style (SIMPLE, SHORT, LONG)
        max_tokens: Maximum tokens to generate
        overwrite: Whether to overwrite existing captions
    """
    start_time = time.time()

    # Validate prompt style
    if prompt_style not in CAPTION_PROMPTS:
        raise ValueError(f"Invalid prompt_style: {prompt_style}. Must be one of {list(CAPTION_PROMPTS.keys())}")

    caption_prompt = CAPTION_PROMPTS[prompt_style]
    print(f"Using prompt style: {prompt_style}")
    print(f"Caption prompt: {caption_prompt}\n")

    # Create output path: captions/{model}/{model}_{style}.json
    model_safe = make_model_safe(model)
    caption_out_dir = os.path.join(output_dir, "captions", model_safe)
    os.makedirs(caption_out_dir, exist_ok=True)
    caption_output_path = os.path.join(caption_out_dir, f"{model_safe}_{prompt_style.lower()}.json")

    print(f"Caption output: {caption_output_path}\n")

    # Load parquet file
    print(f"Loading parquet file: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    print(f"Loaded {len(df)} rows")
    print(f"Columns: {df.columns.tolist()}\n")

    # vLLM server URL
    print(f"Using vLLM server at {vllm_url}")

    # Collect all unique images
    print("\nCollecting all unique images...")
    all_images = collect_all_images(df, dataset_root)
    print(f"Found {len(all_images)} unique images")

    # Load existing captions if available
    captions = {}
    if os.path.exists(caption_output_path) and not overwrite:
        with open(caption_output_path, 'r') as f:
            captions = json.load(f)
        print(f"Loaded {len(captions)} existing captions")

    # Caption images
    print(f"\nCaptioning images...")
    success_count = 0
    error_count = 0

    for img_path in tqdm(sorted(all_images), desc="Captioning"):
        # Skip if already captioned
        if img_path in captions and not overwrite:
            continue

        try:
            caption = caption_image_vllm(
                server_url=vllm_url,
                image_path=img_path,
                model=model,
                prompt=caption_prompt,
                max_tokens=max_tokens
            )
            captions[img_path] = caption
            success_count += 1

            # Incremental save
            with open(caption_output_path, 'w') as f:
                json.dump(captions, f, indent=2, ensure_ascii=False)

        except Exception as e:
            print(f"\nError captioning {img_path}: {e}")
            error_count += 1
            continue

    # Print summary
    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Captioning complete!")
    print(f"{'='*60}")
    print(f"Total unique images: {len(all_images)}")
    print(f"Successfully captioned: {success_count}")
    print(f"Errors: {error_count}")
    print(f"Captions saved to: {caption_output_path}")
    print(f"Total time: {elapsed:.2f}s ({elapsed/60:.2f}m)")
    if success_count > 0:
        print(f"Average time per image: {elapsed/success_count:.2f}s")


def main():
    parser = argparse.ArgumentParser(
        description="Caption images using Qwen3VL via vLLM server",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--parquet-path',
        type=str,
        required=True,
        help='Path to parquet file'
    )
    parser.add_argument(
        '--dataset-root',
        type=str,
        required=True,
        help='Root directory for image paths'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Base output directory (will create captions/{model}/ subfolder)'
    )
    parser.add_argument(
        '--vllm-url',
        type=str,
        default='http://localhost:8000/v1',
        help='vLLM server URL (default: http://localhost:8000/v1)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='Qwen/Qwen3-VL-4B-Instruct',
        help='Model name for captioning (default: Qwen/Qwen3-VL-4B-Instruct)'
    )
    parser.add_argument(
        '--prompt-style',
        type=str,
        default='SIMPLE',
        choices=['SIMPLE', 'SHORT', 'LONG'],
        help='Caption prompt style (default: SIMPLE)'
    )
    parser.add_argument(
        '--max-tokens',
        type=int,
        default=512,
        help='Maximum tokens to generate (default: 512)'
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing captions'
    )

    args = parser.parse_args()

    caption_images(
        parquet_path=args.parquet_path,
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        vllm_url=args.vllm_url,
        model=args.model,
        prompt_style=args.prompt_style,
        max_tokens=args.max_tokens,
        overwrite=args.overwrite
    )


if __name__ == "__main__":
    main()
