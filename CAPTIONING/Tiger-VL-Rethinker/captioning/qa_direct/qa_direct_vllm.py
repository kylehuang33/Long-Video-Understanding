#!/usr/bin/env python3
"""
Direct QA using Qwen3VL via vLLM server.
Reads parquet file, sends images + questions to vLLM server, gets answers.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
from tqdm import tqdm
import time
import base64
from openai import OpenAI


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
    # Detect image format from extension
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


def answer_question_vllm(
    client: OpenAI,
    image_paths: List[str],
    question: str,
    model: str,
    max_tokens: int = 512,
    temperature: float = 0.0
) -> str:
    """
    Answer a question using vLLM server.

    Args:
        client: OpenAI client connected to vLLM server
        image_paths: List of image file paths
        question: Question text (may contain <image> placeholder)
        model: Model name
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature

    Returns:
        Generated answer
    """
    # Build content list with images and text
    content = []

    # Add images
    for img_path in image_paths:
        content.append({
            "type": "image_url",
            "image_url": {"url": get_image_url(img_path)}
        })

    # Add question text (remove <image> placeholder as we're adding images separately)
    question_text = question.replace("<image>", "").strip()
    content.append({
        "type": "text",
        "text": question_text
    })

    # Call vLLM server
    response = client.chat.completions.create(
        model=model,
        messages=[{
            "role": "user",
            "content": content
        }],
        max_tokens=max_tokens,
        temperature=temperature
    )

    return response.choices[0].message.content.strip()


def process_parquet(
    parquet_path: str,
    dataset_root: str,
    output_dir: str,
    vllm_url: str = "http://localhost:8000/v1",
    model: str = "Qwen/Qwen3-VL-4B-Instruct",
    max_tokens: int = 512,
    temperature: float = 0.0,
    overwrite: bool = False
):
    """
    Process parquet file and answer questions using vLLM.

    Args:
        parquet_path: Path to parquet file
        dataset_root: Root directory for image paths
        output_dir: Base output directory
        vllm_url: vLLM server URL
        model: Model name
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        overwrite: Whether to overwrite existing results
    """
    start_time = time.time()

    # Create output path based on model name
    model_safe = make_model_safe(model)
    out_dir = os.path.join(output_dir, model_safe)
    os.makedirs(out_dir, exist_ok=True)
    output_path = os.path.join(out_dir, f"{model_safe}_results.json")
    print(f"Output will be saved to: {output_path}\n")

    # Load parquet file
    print(f"Loading parquet file: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    print(f"Loaded {len(df)} rows")
    print(f"Columns: {df.columns.tolist()}\n")

    # Initialize OpenAI client for vLLM
    print(f"Connecting to vLLM server at {vllm_url}")
    client = OpenAI(
        api_key="EMPTY",  # vLLM doesn't require API key
        base_url=vllm_url
    )

    # Load existing results if available
    results = {}
    if os.path.exists(output_path) and not overwrite:
        with open(output_path, 'r') as f:
            results = json.load(f)
        print(f"Loaded {len(results)} existing results from {output_path}\n")

    # Process each row
    print("Processing questions...")
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

            # Check if images exist
            missing = [img for img in image_paths if not os.path.exists(img)]
            if missing:
                print(f"\nWarning: Missing images for {qid}: {missing}")
                error_count += 1
                continue

            # Get question and ground truth answer
            question = row['question']
            ground_truth = row['answer']

            # Generate answer
            predicted_answer = answer_question_vllm(
                client=client,
                image_paths=image_paths,
                question=question,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature
            )

            # Save result
            results[qid] = {
                'question': question,
                'ground_truth': ground_truth,
                'predicted_answer': predicted_answer,
                'category': row['category'],
                'source': row['source'],
                'image_paths': image_paths
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
    print(f"QA complete!")
    print(f"{'='*60}")
    print(f"Total questions: {len(df)}")
    print(f"Successfully answered: {success_count}")
    print(f"Errors: {error_count}")
    print(f"Results saved to: {output_path}")
    print(f"Total time: {elapsed:.2f}s ({elapsed/60:.2f}m)")
    if success_count > 0:
        print(f"Average time per question: {elapsed/success_count:.2f}s")


def main():
    parser = argparse.ArgumentParser(
        description="Direct QA using Qwen3VL via vLLM server",
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
        help='Base output directory (will create model subfolder)'
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
        help='Model name (default: Qwen/Qwen3-VL-4B-Instruct)'
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
        help='Sampling temperature (default: 0.0)'
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing results'
    )

    args = parser.parse_args()

    process_parquet(
        parquet_path=args.parquet_path,
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        vllm_url=args.vllm_url,
        model=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        overwrite=args.overwrite
    )


if __name__ == "__main__":
    main()
