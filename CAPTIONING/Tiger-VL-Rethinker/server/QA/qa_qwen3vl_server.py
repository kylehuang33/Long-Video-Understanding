#!/usr/bin/env python3
"""
Question Answering using Qwen3VL via vLLM server.

This script evaluates VQA questions using the Qwen3VL model running on a vLLM server.
"""

import argparse
import json
import os
import re
import base64
from pathlib import Path
from typing import Dict, Any, List, Optional
from tqdm import tqdm
import time
import pandas as pd
import requests

LETTER_ALPH = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


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

    # Handle HuggingFace model IDs
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


def extract_letter(answer_text: str, num_options: int) -> Optional[str]:
    """Extract answer letter from model output."""
    if not answer_text:
        return None

    if "</think>" in answer_text:
        after_think = answer_text.split("</think>", 1)[1]
        answer_text = after_think

    if "Answer: " in answer_text:
        after_answer = answer_text.split("Answer: ", 1)[1]
        answer_text = after_answer

    if "\n" in answer_text:
        after_n = answer_text.split("\n", 1)[0]
        answer_text = after_n

    m = re.search(r"\b([A-Z])\b", answer_text.upper())
    if m:
        letter = m.group(1)
        idx = LETTER_ALPH.find(letter)
        if 0 <= idx < max(1, num_options):
            return letter

    m = re.search(r"\b([1-9][0-9]?)\b", answer_text)
    if m:
        k = int(m.group(1))
        if 1 <= k <= max(1, num_options):
            return LETTER_ALPH[k - 1]

    return None


def extract_answer_from_boxed(answer_str: str) -> Optional[str]:
    """Extract answer from \\boxed{X} format."""
    if not answer_str:
        return None

    match = re.search(r'\\boxed\{([A-Z])\}', answer_str)
    if match:
        return match.group(1)

    match = re.search(r'\b([A-Z])\b', answer_str.upper())
    if match:
        return match.group(1)

    return None


def parse_question_choices(question_text: str) -> tuple[str, List[str]]:
    """Parse question text to extract question and choices."""
    text = question_text.replace("<image>", "").strip()

    if "Choices:" in text or "choices:" in text.lower():
        parts = re.split(r'[Cc]hoices:\s*\n', text, maxsplit=1)
        question_only = parts[0].strip()

        if len(parts) > 1:
            choices_text = parts[1].strip()
            choice_pattern = r'\([A-Z]\)\s*(.+?)(?=\n\([A-Z]\)|$)'
            matches = re.findall(choice_pattern, choices_text, re.DOTALL)
            choices = [m.strip() for m in matches]

            if choices:
                return question_only, choices

    return text, []


def build_qa_prompt(question: str, choices: List[str]) -> str:
    """Build prompt for QA with choices."""
    if not choices:
        return question

    lines = [f"{LETTER_ALPH[i]}. {choice}" for i, choice in enumerate(choices)]

    prompt = f"""{question}

Options:
{chr(10).join(lines)}

Answer with just the letter (A, B, C, etc.) of the correct option."""

    return prompt


def encode_image_base64(image_path: str) -> str:
    """Encode image to base64 string."""
    with open(image_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')


def generate_answer_server(
    server_url: str,
    image_paths: List[str],
    question_text: str,
    max_tokens: int = 128
) -> str:
    """
    Generate answer using vLLM server API.

    Args:
        server_url: Base URL of vLLM server
        image_paths: List of paths to image files
        question_text: Question text with prompt
        max_tokens: Maximum number of tokens to generate

    Returns:
        Generated answer text
    """
    # Build content with all images
    content = []
    for img_path in image_paths:
        image_base64 = encode_image_base64(img_path)
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{image_base64}"
            }
        })
    content.append({
        "type": "text",
        "text": question_text
    })

    # Build messages
    messages = [
        {
            "role": "user",
            "content": content
        }
    ]

    # Make API call
    response = requests.post(
        f"{server_url}/v1/chat/completions",
        json={
            "model": "Qwen/Qwen3-VL-4B-Instruct",
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


def load_questions_from_parquet(question_path: str) -> pd.DataFrame:
    """Load questions from parquet file."""
    print(f"Loading questions from {question_path}...")
    df = pd.read_parquet(question_path)
    print(f"Loaded {len(df)} questions")
    print(f"Columns: {list(df.columns)}")
    return df


def evaluate_questions(
    question_path: str,
    dataset_path: str,
    output_dir: str,
    server_url: str,
    model_name: str = "Qwen3-VL-4B-Instruct",
    max_tokens: int = 128,
    overwrite: bool = False
):
    """Evaluate VQA questions using vLLM server."""
    start_time = time.time()

    # Load questions
    df = load_questions_from_parquet(question_path)

    # Construct output path
    model_safe = make_model_safe(model_name)
    out_dir = os.path.join(output_dir, "qa", model_safe)
    os.makedirs(out_dir, exist_ok=True)
    output_path = os.path.join(out_dir, f"{model_safe}_qa_results.json")

    print(f"Output directory: {out_dir}")
    print(f"Output file: {output_path}")
    print(f"Server URL: {server_url}\n")

    # Load existing results if available
    results = {}
    if os.path.exists(output_path) and not overwrite:
        with open(output_path, 'r') as f:
            results = json.load(f)
        print(f"Loaded {len(results)} existing results from {output_path}\n")

    # Process each question
    print(f"Processing questions...")
    correct_count = 0
    total_count = 0

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):
        qid = row['qid']

        # Skip if already processed
        if qid in results and not overwrite:
            total_count += 1
            if results[qid].get('is_correct', False):
                correct_count += 1
            continue

        try:
            # Get image paths
            image_rel_paths = row['image']

            if hasattr(image_rel_paths, 'tolist'):
                image_rel_paths = image_rel_paths.tolist()
            elif not isinstance(image_rel_paths, (list, tuple)):
                image_rel_paths = [image_rel_paths]

            if isinstance(image_rel_paths, (list, tuple)):
                image_rel_paths = [str(p) for p in image_rel_paths]
            else:
                image_rel_paths = [str(image_rel_paths)]

            # Build full paths
            image_paths = [os.path.join(dataset_path, p) for p in image_rel_paths]

            # Check if all images exist
            missing_images = [p for p in image_paths if not os.path.exists(p)]
            if missing_images:
                print(f"\nWarning: Image(s) not found: {missing_images}")
                results[qid] = {
                    'qid': qid,
                    'question': row['question'],
                    'gt_answer': row['answer'],
                    'predicted_answer': None,
                    'model_output': f"Image(s) not found: {missing_images}",
                    'is_correct': False,
                    'category': row.get('category', 'unknown'),
                    'error': 'image_not_found'
                }
                total_count += 1
                continue

            # Parse question and choices
            question_only, choices = parse_question_choices(row['question'])

            # Build prompt
            prompt_text = build_qa_prompt(question_only, choices)

            # Generate answer
            model_output = generate_answer_server(
                server_url=server_url,
                image_paths=image_paths,
                question_text=prompt_text,
                max_tokens=max_tokens
            )

            # Extract predicted letter
            predicted_letter = extract_letter(model_output, len(choices))

            # Extract ground truth letter
            gt_answer = row['answer']
            gt_letter = extract_answer_from_boxed(gt_answer)

            # Check correctness
            is_correct = (predicted_letter == gt_letter) if (predicted_letter and gt_letter) else False

            # Save result
            results[qid] = {
                'qid': qid,
                'question': question_only,
                'choices': choices,
                'gt_answer': gt_answer,
                'gt_letter': gt_letter,
                'predicted_letter': predicted_letter,
                'model_output': model_output,
                'is_correct': is_correct,
                'category': row.get('category', 'unknown'),
                'source': row.get('source', 'unknown'),
                'image_paths': image_rel_paths,
                'num_images': len(image_rel_paths)
            }

            # Update counters
            total_count += 1
            if is_correct:
                correct_count += 1

            # Save incrementally every 10 questions
            if total_count % 10 == 0:
                with open(output_path, 'w') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)

        except Exception as e:
            print(f"\nError processing question {qid}: {e}")
            results[qid] = {
                'qid': qid,
                'question': row['question'],
                'gt_answer': row['answer'],
                'predicted_answer': None,
                'model_output': f"Error: {str(e)}",
                'is_correct': False,
                'category': row.get('category', 'unknown'),
                'error': str(e)
            }
            total_count += 1
            continue

    # Final save
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Compute metrics
    metrics = compute_metrics(results)
    metrics['model'] = model_name
    metrics['question_path'] = question_path
    metrics['dataset_path'] = dataset_path
    metrics['server_url'] = server_url

    # Save metrics
    metrics_path = output_path.replace('.json', '_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    # Print summary
    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Evaluation complete!")
    print(f"{'='*60}")
    print(f"Total questions: {total_count}")
    print(f"Correct answers: {correct_count} ({correct_count/total_count*100:.2f}%)")
    print(f"Overall accuracy: {metrics['overall_accuracy']:.4f}")
    print(f"Results saved to: {output_path}")
    print(f"Metrics saved to: {metrics_path}")
    print(f"Total time: {elapsed:.2f}s ({elapsed/60:.2f}m)")
    if total_count > 0:
        print(f"Average time per question: {elapsed/total_count:.2f}s")
    print(f"{'='*60}")

    # Print per-category metrics
    if metrics['category_metrics']:
        print(f"\nPer-category accuracy:")
        print(f"{'='*60}")
        for category, stats in sorted(metrics['category_metrics'].items()):
            print(f"{category}: {stats['correct']}/{stats['total']} ({stats['accuracy']:.2%})")
        print(f"{'='*60}")


def compute_metrics(results: Dict[str, Any]) -> Dict[str, Any]:
    """Compute evaluation metrics from results."""
    total_questions = len(results)
    correct_answers = sum(1 for v in results.values() if v.get('is_correct', False))
    overall_accuracy = correct_answers / total_questions if total_questions > 0 else 0.0

    # Compute per-category metrics
    category_metrics = {}
    for qid, item in results.items():
        category = item.get('category', 'unknown')
        if category not in category_metrics:
            category_metrics[category] = {
                'total': 0,
                'correct': 0,
            }
        category_metrics[category]['total'] += 1
        if item.get('is_correct'):
            category_metrics[category]['correct'] += 1

    # Compute accuracy per category
    for category, stats in category_metrics.items():
        stats['accuracy'] = stats['correct'] / stats['total'] if stats['total'] > 0 else 0.0

    return {
        'total_questions': total_questions,
        'correct_answers': correct_answers,
        'overall_accuracy': round(overall_accuracy, 4),
        'category_metrics': category_metrics
    }


def main():
    parser = argparse.ArgumentParser(
        description="Question Answering using Qwen3VL via vLLM server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate questions
  python qa_qwen3vl_server.py \\
    --question-path /path/to/questions.parquet \\
    --dataset-path /path/to/dataset \\
    --server-url http://localhost:8000

  # With custom output directory
  python qa_qwen3vl_server.py \\
    --question-path /path/to/questions.parquet \\
    --dataset-path /path/to/dataset \\
    --server-url http://10.1.64.88:8000 \\
    --output-dir ./custom_output
        """
    )

    # Required arguments
    parser.add_argument(
        '--question-path',
        type=str,
        required=True,
        help='Path to parquet file with questions'
    )
    parser.add_argument(
        '--dataset-path',
        type=str,
        required=True,
        help='Base path for dataset images'
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
        '--model-name',
        type=str,
        default='Qwen3-VL-4B-Instruct',
        help='Model name for output folder (default: Qwen3-VL-4B-Instruct)'
    )
    parser.add_argument(
        '--max-tokens',
        type=int,
        default=128,
        help='Maximum number of tokens to generate (default: 128)'
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing results'
    )

    args = parser.parse_args()

    # Run evaluation
    evaluate_questions(
        question_path=args.question_path,
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        server_url=args.server_url,
        model_name=args.model_name,
        max_tokens=args.max_tokens,
        overwrite=args.overwrite
    )


if __name__ == "__main__":
    main()
