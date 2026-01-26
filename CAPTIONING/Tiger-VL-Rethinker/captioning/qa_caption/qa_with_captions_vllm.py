#!/usr/bin/env python3
"""
QA using existing captions via vLLM server.
This script requires captions to be generated first using caption_images_vllm.py.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Optional
import pandas as pd
from tqdm import tqdm
import time
import requests


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


def answer_question_text_only(
    server_url: str,
    question: str,
    model: str,
    max_tokens: int = 512,
    temperature: float = 0.0
) -> str:
    """
    Answer a text-only question using vLLM server.

    Args:
        server_url: vLLM server URL
        question: Question text (with captions replacing <image>)
        model: Model name
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature

    Returns:
        Generated answer
    """
    messages = [{
        "role": "user",
        "content": question
    }]

    response = requests.post(
        f"{server_url}/v1/chat/completions",
        json={
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature
        },
        headers={"Content-Type": "application/json"}
    )

    if response.status_code != 200:
        raise Exception(f"Server error: {response.status_code} - {response.text}")

    result = response.json()
    return result['choices'][0]['message']['content'].strip()


def qa_with_captions(
    parquet_path: str,
    dataset_root: str,
    caption_file: str,
    output_dir: str,
    vllm_url: str = "http://localhost:8000/v1",
    model: str = "Qwen/Qwen3-VL-4B-Instruct",
    max_tokens: int = 512,
    temperature: float = 0.0,
    overwrite: bool = False
):
    """
    Answer questions using existing captions.

    Args:
        parquet_path: Path to parquet file
        dataset_root: Root directory for image paths
        caption_file: Path to caption JSON file
        output_dir: Base output directory
        vllm_url: vLLM server URL
        model: Model name for QA
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature for QA
        overwrite: Whether to overwrite existing results
    """
    start_time = time.time()

    # Load captions
    print(f"Loading captions from: {caption_file}")
    if not os.path.exists(caption_file):
        raise FileNotFoundError(f"Caption file not found: {caption_file}\nPlease run caption_images_vllm.py first!")

    with open(caption_file, 'r') as f:
        captions = json.load(f)
    print(f"Loaded {len(captions)} captions\n")

    # Extract caption model and prompt style from filename
    # Format: {model}_{style}.json
    caption_filename = Path(caption_file).stem  # e.g., "Qwen3-VL-4B-Instruct_simple"
    parts = caption_filename.rsplit('_', 1)
    if len(parts) == 2:
        caption_model_safe = parts[0]
        prompt_style = parts[1]
    else:
        caption_model_safe = caption_filename
        prompt_style = "unknown"

    # Create output path: qa_results/{qa_model}/{qa_model}_with_{caption_model}_{style}.json
    model_safe = make_model_safe(model)
    qa_out_dir = os.path.join(output_dir, "qa_results", model_safe)
    os.makedirs(qa_out_dir, exist_ok=True)
    output_path = os.path.join(qa_out_dir, f"{model_safe}_with_{caption_model_safe}_{prompt_style}.json")

    print(f"QA output: {output_path}\n")

    # Load parquet file
    print(f"Loading parquet file: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    print(f"Loaded {len(df)} rows")
    print(f"Columns: {df.columns.tolist()}\n")

    # vLLM server URL
    print(f"Using vLLM server at {vllm_url}")

    # Load existing results if available
    results = {}
    if os.path.exists(output_path) and not overwrite:
        with open(output_path, 'r') as f:
            results = json.load(f)
        print(f"Loaded {len(results)} existing results\n")

    # Process questions
    print("Answering questions using captions...")
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
            missing_captions = []
            for img_path in image_paths:
                if img_path in captions:
                    image_captions.append(captions[img_path])
                else:
                    missing_captions.append(img_path)

            if missing_captions:
                print(f"\nWarning: No captions found for {qid}: {missing_captions}")
                error_count += 1
                continue

            # Replace <image> with captions
            question = row['question']
            # If multiple images, replace each <image> with corresponding caption
            for caption in image_captions:
                question = question.replace("<image>", f"[Image: {caption}]", 1)

            ground_truth = row['answer']

            # Generate answer using text-only question
            predicted_answer = answer_question_text_only(
                server_url=vllm_url,
                question=question,
                model=model,
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
                'captions': image_captions,
                'caption_file': caption_file
            }
            success_count += 1

            # Incremental save
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

        except Exception as e:
            print(f"\nError processing {qid}: {e}")
            error_count += 1
            continue

    # Print summary
    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"QA with captions complete!")
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
        description="QA using existing captions via vLLM server",
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
        '--caption-file',
        type=str,
        required=True,
        help='Path to caption JSON file (from caption_images_vllm.py)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Base output directory (will create qa_results/{model}/ subfolder)'
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
        help='Model name for QA (default: Qwen/Qwen3-VL-4B-Instruct)'
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

    qa_with_captions(
        parquet_path=args.parquet_path,
        dataset_root=args.dataset_root,
        caption_file=args.caption_file,
        output_dir=args.output_dir,
        vllm_url=args.vllm_url,
        model=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        overwrite=args.overwrite
    )


if __name__ == "__main__":
    main()
