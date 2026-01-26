#!/usr/bin/env python3
"""
Caption-based QA using Qwen3VL via vLLM server.
First captions all images, then answers questions using captions instead of images.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Set, Optional
import pandas as pd
from tqdm import tqdm
import time
import base64
from openai import OpenAI


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


def get_image_url(image_path: str) -> str:
    """Convert image path to data URL for vLLM."""
    base64_image = encode_image_base64(image_path)
    ext = Path(image_path).suffix.lower()
    mime_type = {
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.gif': 'image/gif',
        '.webp': 'image/webp',
        '.bmp': 'image/bmp',
    }.get(ext, 'image/jpeg')

    return f"data:{mime_type};base64,{base64_image}"


def caption_image_vllm(
    client: OpenAI,
    image_path: str,
    model: str,
    prompt: str = "Describe this image in detail.",
    max_tokens: int = 512
) -> str:
    """
    Caption a single image using vLLM server.

    Args:
        client: OpenAI client connected to vLLM server
        image_path: Path to image file
        model: Model name
        prompt: Caption prompt
        max_tokens: Maximum tokens to generate

    Returns:
        Generated caption
    """
    content = [
        {
            "type": "image_url",
            "image_url": {"url": get_image_url(image_path)}
        },
        {
            "type": "text",
            "text": prompt
        }
    ]

    response = client.chat.completions.create(
        model=model,
        messages=[{
            "role": "user",
            "content": content
        }],
        max_tokens=max_tokens,
        temperature=0.0
    )

    return response.choices[0].message.content.strip()


def answer_question_text_only(
    client: OpenAI,
    question: str,
    model: str,
    max_tokens: int = 512,
    temperature: float = 0.0
) -> str:
    """
    Answer a text-only question using vLLM server.

    Args:
        client: OpenAI client connected to vLLM server
        question: Question text (with captions replacing <image>)
        model: Model name
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature

    Returns:
        Generated answer
    """
    response = client.chat.completions.create(
        model=model,
        messages=[{
            "role": "user",
            "content": question
        }],
        max_tokens=max_tokens,
        temperature=temperature
    )

    return response.choices[0].message.content.strip()


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


def caption_all_images(
    client: OpenAI,
    image_paths: Set[str],
    caption_output_path: str,
    model: str,
    caption_prompt: str = "Describe this image in detail.",
    max_tokens: int = 512,
    overwrite: bool = False
) -> Dict[str, str]:
    """
    Caption all images and save to JSON.

    Args:
        client: OpenAI client
        image_paths: Set of image paths to caption
        caption_output_path: Path to save captions JSON
        model: Model name
        caption_prompt: Prompt for captioning
        max_tokens: Maximum tokens to generate
        overwrite: Whether to overwrite existing captions

    Returns:
        Dictionary mapping image paths to captions
    """
    # Load existing captions if available
    captions = {}
    if os.path.exists(caption_output_path) and not overwrite:
        with open(caption_output_path, 'r') as f:
            captions = json.load(f)
        print(f"Loaded {len(captions)} existing captions")

    # Caption images
    print(f"\nCaptioning {len(image_paths)} images...")
    success_count = 0
    error_count = 0

    for img_path in tqdm(sorted(image_paths), desc="Captioning"):
        # Skip if already captioned
        if img_path in captions and not overwrite:
            continue

        try:
            caption = caption_image_vllm(
                client=client,
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

    print(f"Captioning complete: {success_count} success, {error_count} errors")
    return captions


def process_parquet_with_captions(
    parquet_path: str,
    dataset_root: str,
    output_dir: str,
    vllm_url: str = "http://localhost:8000/v1",
    caption_model: str = "Qwen/Qwen3-VL-4B-Instruct",
    qa_model: str = "Qwen/Qwen3-VL-4B-Instruct",
    prompt_style: str = 'SIMPLE',
    max_tokens: int = 512,
    temperature: float = 0.0,
    overwrite: bool = False
):
    """
    Process parquet file: caption images first, then answer questions using captions.

    Args:
        parquet_path: Path to parquet file
        dataset_root: Root directory for image paths
        output_dir: Base output directory
        vllm_url: vLLM server URL
        caption_model: Model name for captioning
        qa_model: Model name for QA
        prompt_style: Caption prompt style (SIMPLE, SHORT, LONG)
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature for QA
        overwrite: Whether to overwrite existing results
    """
    start_time = time.time()

    # Validate prompt style
    if prompt_style not in CAPTION_PROMPTS:
        raise ValueError(f"Invalid prompt_style: {prompt_style}. Must be one of {list(CAPTION_PROMPTS.keys())}")

    caption_prompt = CAPTION_PROMPTS[prompt_style]
    print(f"Using prompt style: {prompt_style}")
    print(f"Caption prompt: {caption_prompt}\n")

    # Create output paths based on model names
    caption_model_safe = make_model_safe(caption_model)
    qa_model_safe = make_model_safe(qa_model)

    # Caption output: captions/{caption_model}/{caption_model}_{prompt_style}.json
    caption_out_dir = os.path.join(output_dir, "captions", caption_model_safe)
    os.makedirs(caption_out_dir, exist_ok=True)
    caption_output_path = os.path.join(caption_out_dir, f"{caption_model_safe}_{prompt_style.lower()}.json")

    # QA output: qa_results/{qa_model}/{qa_model}_with_{caption_model}_{prompt_style}.json
    qa_out_dir = os.path.join(output_dir, "qa_results", qa_model_safe)
    os.makedirs(qa_out_dir, exist_ok=True)
    output_path = os.path.join(qa_out_dir, f"{qa_model_safe}_with_{caption_model_safe}_{prompt_style.lower()}.json")

    print(f"Caption output: {caption_output_path}")
    print(f"QA output: {output_path}\n")

    # Load parquet file
    print(f"Loading parquet file: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    print(f"Loaded {len(df)} rows")
    print(f"Columns: {df.columns.tolist()}\n")

    # Initialize OpenAI client for vLLM
    print(f"Connecting to vLLM server at {vllm_url}")
    client = OpenAI(
        api_key="EMPTY",
        base_url=vllm_url
    )

    # Step 1: Collect all unique images
    print("\nStep 1: Collecting all unique images...")
    all_images = collect_all_images(df, dataset_root)
    print(f"Found {len(all_images)} unique images")

    # Step 2: Caption all images
    print("\nStep 2: Captioning all images...")
    captions = caption_all_images(
        client=client,
        image_paths=all_images,
        caption_output_path=caption_output_path,
        model=caption_model,
        caption_prompt=caption_prompt,
        max_tokens=max_tokens,
        overwrite=overwrite
    )

    # Step 3: Answer questions using captions
    print("\nStep 3: Answering questions using captions...")

    # Load existing results if available
    results = {}
    if os.path.exists(output_path) and not overwrite:
        with open(output_path, 'r') as f:
            results = json.load(f)
        print(f"Loaded {len(results)} existing results\n")

    success_count = 0
    error_count = 0

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Answering"):
        qid = row['qid']

        # Skip if already processed
        if qid in results and not overwrite:
            continue

        try:
            # Get image paths
            image_array = row['image']
            if isinstance(image_array, str):
                image_paths = [image_array]
            else:
                image_paths = list(image_array)

            # Convert to absolute paths
            image_paths = [os.path.join(dataset_root, img) for img in image_paths]

            # Get captions for images
            image_captions = []
            for img_path in image_paths:
                if img_path in captions:
                    image_captions.append(captions[img_path])
                else:
                    print(f"\nWarning: No caption found for {img_path}")
                    error_count += 1
                    continue

            if len(image_captions) != len(image_paths):
                continue

            # Replace <image> with captions
            question = row['question']
            # If multiple images, replace each <image> with corresponding caption
            for caption in image_captions:
                question = question.replace("<image>", f"[Image: {caption}]", 1)

            ground_truth = row['answer']

            # Generate answer using text-only question
            predicted_answer = answer_question_text_only(
                client=client,
                question=question,
                model=qa_model,
                max_tokens=max_tokens,
                temperature=temperature
            )

            # Save result
            results[qid] = {
                'question_original': row['question'],
                'question_with_captions': question,
                'ground_truth': ground_truth,
                'predicted_answer': predicted_answer,
                'category': row['category'],
                'source': row['source'],
                'image_paths': image_paths,
                'captions': image_captions
            }
            success_count += 1

            # Incremental save
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

        except Exception as e:
            print(f"\nError processing {qid}: {e}")
            error_count += 1
            continue

    # Final save
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Print summary
    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Caption-based QA complete!")
    print(f"{'='*60}")
    print(f"Total questions: {len(df)}")
    print(f"Successfully answered: {success_count}")
    print(f"Errors: {error_count}")
    print(f"Results saved to: {output_path}")
    print(f"Captions saved to: {caption_output_path}")
    print(f"Total time: {elapsed:.2f}s ({elapsed/60:.2f}m)")
    if success_count > 0:
        print(f"Average time per question: {elapsed/success_count:.2f}s")


def main():
    parser = argparse.ArgumentParser(
        description="Caption-based QA using Qwen3VL via vLLM server",
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
        help='Base output directory (will create model subfolders)'
    )
    parser.add_argument(
        '--vllm-url',
        type=str,
        default='http://localhost:8000/v1',
        help='vLLM server URL (default: http://localhost:8000/v1)'
    )
    parser.add_argument(
        '--caption-model',
        type=str,
        default='Qwen/Qwen3-VL-4B-Instruct',
        help='Model name for captioning (default: Qwen/Qwen3-VL-4B-Instruct)'
    )
    parser.add_argument(
        '--qa-model',
        type=str,
        default='Qwen/Qwen3-VL-4B-Instruct',
        help='Model name for QA (default: Qwen/Qwen3-VL-4B-Instruct)'
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
        '--temperature',
        type=float,
        default=0.0,
        help='Sampling temperature for QA (default: 0.0)'
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing results'
    )

    args = parser.parse_args()

    process_parquet_with_captions(
        parquet_path=args.parquet_path,
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        vllm_url=args.vllm_url,
        caption_model=args.caption_model,
        qa_model=args.qa_model,
        prompt_style=args.prompt_style,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        overwrite=args.overwrite
    )


if __name__ == "__main__":
    main()
