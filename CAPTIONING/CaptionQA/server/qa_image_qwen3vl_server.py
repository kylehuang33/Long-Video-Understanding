#!/usr/bin/env python3
"""
Image-Based Question Answering using Qwen3VL via vLLM server.

This script evaluates VQA questions using images directly (not captions) with the Qwen3VL model
running on a vLLM server. It loads questions from the CaptionQA dataset and answers them using
the visual content.

Features:
- Loads questions from HuggingFace dataset (Borise/CaptionQA)
- Uses images directly instead of captions
- Supports multiple images per question
- Runs inference via vLLM server (OpenAI-compatible API)
- Automatic answer extraction and accuracy computation
- Per-category performance metrics
- Incremental saving and auto-resume

Usage:
    # Basic usage
    python qa_image_qwen3vl_server.py \
        --dataset "Borise/CaptionQA" \
        --split "natural" \
        --server-url "http://localhost:8000" \
        --output-dir "./outputs"

    # With custom configuration
    python qa_image_qwen3vl_server.py \
        --dataset "Borise/CaptionQA" \
        --split "all" \
        --server-url "http://10.1.64.88:8000" \
        --output-dir "./outputs" \
        --model-name "Qwen3-VL-4B-Instruct" \
        --max-tokens 128 \
        --temperature 0.0
"""

import argparse
import json
import os
import re
import base64
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Optional
from tqdm import tqdm
import time
import requests
from datasets import load_dataset
from PIL import Image

LETTER_ALPH = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
CANNOT_ANSWER_TEXT = "Cannot answer from the image"

# Available domain splits in the dataset
DOMAIN_SPLITS = ["natural", "document", "ecommerce", "embodiedai"]


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


def normalize_gt_letter(choices: List[str], answer: str) -> Optional[str]:
    """Extract ground truth answer letter from question.

    Args:
        choices: List of choice strings
        answer: The correct answer text

    Returns:
        Letter corresponding to the correct choice, or None if not found
    """
    if not choices or not isinstance(answer, str):
        return None

    # Match answer text to one of the choices
    for i, choice in enumerate(choices):
        if answer.strip() == str(choice).strip():
            return LETTER_ALPH[i]

    return None


def is_yesno_question(question_text: str, choices: List[str]) -> bool:
    """
    Check if question is a yes/no question.

    A question is considered yes/no if:
    1. The choices contain "Yes" and "No" (in any order), OR
    2. The question starts with common yes/no question words
    """
    # Check if choices contain yes and no
    choice_texts = [str(c).strip().lower() for c in choices]

    has_yes = any("yes" in choice for choice in choice_texts)
    has_no = any("no" in choice for choice in choice_texts)

    if has_yes and has_no:
        return True

    # Check if question starts with yes/no question words
    question_lower = question_text.strip().lower()
    yesno_starters = [
        "is ", "are ", "was ", "were ",
        "do ", "does ", "did ",
        "have ", "has ", "had ",
        "can ", "could ",
        "will ", "would ",
        "should ", "shall ",
        "may ", "might ", "must "
    ]

    for starter in yesno_starters:
        if question_lower.startswith(starter):
            return True

    return False


def add_cannot_answer_option(question_text: str, choices: List[str]) -> List[str]:
    """Add 'cannot answer from the image' option to non-yes/no questions."""
    if is_yesno_question(question_text, choices):
        return choices

    return choices + [CANNOT_ANSWER_TEXT]


def build_image_qa_prompt(question: str, choices: List[str]) -> str:
    """Build prompt for image-based QA with choices."""
    if not choices:
        return question

    lines = [f"{LETTER_ALPH[i]}. {choice}" for i, choice in enumerate(choices)]

    prompt = f"""Question:
{question}

Options:
{chr(10).join(lines)}

Answer with just the letter (A, B, C, etc.) of the correct option."""

    return prompt


def save_pil_image_to_temp(pil_image: Image.Image, suffix: str = ".jpg") -> str:
    """Save a PIL image to a temporary file and return the path."""
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    pil_image.save(temp_file.name)
    return temp_file.name


def encode_image_base64(image: Any) -> str:
    """Encode image to base64 string.

    Args:
        image: Can be a file path (str) or PIL Image object

    Returns:
        Base64 encoded string
    """
    if isinstance(image, str):
        # It's a file path
        with open(image, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')
    elif isinstance(image, Image.Image):
        # It's a PIL Image, save to temp file first
        temp_path = save_pil_image_to_temp(image)
        try:
            with open(temp_path, 'rb') as f:
                return base64.b64encode(f.read()).decode('utf-8')
        finally:
            # Clean up temp file
            try:
                os.unlink(temp_path)
            except:
                pass
    else:
        raise ValueError(f"Unsupported image type: {type(image)}")


def generate_answer_server(
    server_url: str,
    images: List[Any],
    question_text: str,
    max_tokens: int = 128,
    temperature: float = 0.0
) -> str:
    """
    Generate answer using vLLM server API.

    Args:
        server_url: Base URL of vLLM server
        images: List of images (can be PIL Image objects or file paths)
        question_text: Question text with prompt
        max_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature

    Returns:
        Generated answer text
    """
    # Build content with all images
    content = []
    for img in images:
        image_base64 = encode_image_base64(img)
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
            "temperature": temperature
        },
        headers={"Content-Type": "application/json"}
    )

    if response.status_code != 200:
        raise Exception(f"Server error: {response.status_code} - {response.text}")

    result = response.json()
    return result['choices'][0]['message']['content'].strip()


def load_captionqa_dataset(dataset_name: str, split: str):
    """
    Load CaptionQA dataset from HuggingFace.

    Args:
        dataset_name: HuggingFace dataset name (default: "Borise/CaptionQA")
        split: Domain split to use: "natural", "document", "ecommerce", "embodiedai", or "all"

    Returns:
        HuggingFace dataset object
    """
    print(f"Loading dataset {dataset_name} (split: {split})...")

    # Load the specified split directly
    dataset = load_dataset(dataset_name, split=split)

    print(f"Loaded {len(dataset)} entries")
    return dataset


def evaluate_questions(
    dataset_name: str,
    split: str,
    output_dir: str,
    server_url: str,
    model_name: str = "Qwen3-VL-4B-Instruct",
    max_tokens: int = 128,
    temperature: float = 0.0,
    overwrite: bool = False,
    save_every: int = 10
):
    """Evaluate VQA questions using vLLM server."""
    start_time = time.time()

    # Load dataset
    dataset = load_captionqa_dataset(dataset_name, split)

    # Construct output path
    model_safe = make_model_safe(model_name)
    out_dir = os.path.join(output_dir, "qa_image", model_safe)
    os.makedirs(out_dir, exist_ok=True)
    output_path = os.path.join(out_dir, f"{model_safe}_qa_image_results.json")

    print(f"Output directory: {out_dir}")
    print(f"Output file: {output_path}")
    print(f"Server URL: {server_url}\n")

    # Load existing results if available
    results = {}
    if os.path.exists(output_path) and not overwrite:
        with open(output_path, 'r') as f:
            results = json.load(f)
        print(f"Loaded {len(results)} existing results from {output_path}\n")

    # Process each entry
    print(f"Processing questions...")
    correct_count = 0
    total_count = 0
    cannot_answer_count = 0

    # Track processed entries for resume capability
    processed_entries = set(results.keys())

    for entry in tqdm(dataset, desc="Evaluating"):
        # Get image identifier
        image_id = entry.get('id')
        if image_id is None:
            continue

        image_key = str(image_id)

        # Skip if already processed
        if image_key in processed_entries and not overwrite:
            # Count existing results
            entry_results = results[image_key]
            if isinstance(entry_results, list):
                total_count += len(entry_results)
                correct_count += sum(1 for r in entry_results if r.get('is_correct', False))
                cannot_answer_count += sum(1 for r in entry_results if r.get('is_cannot_answer', False))
            continue

        try:
            # Get image(s) - CaptionQA dataset has 'image' field with PIL Image(s)
            image = entry.get('image')
            if image is None:
                print(f"\nWarning: No image for entry {image_key}")
                continue

            # Ensure image is in a list
            if not isinstance(image, list):
                images = [image]
            else:
                images = image

            # Get questions for this entry
            questions = entry.get('questions', [])
            if not questions:
                # Single question format
                if 'question' in entry:
                    cat = entry.get('category', [])
                    if isinstance(cat, list):
                        cat = cat[0] if cat else ''
                    questions = [{
                        'question': entry['question'],
                        'choices': entry.get('choices', []),
                        'answer': entry.get('answer'),
                        'category': cat
                    }]
                else:
                    continue

            # Process all questions for this entry
            entry_results = []

            for q in questions:
                question_text = q.get('question', '')
                choices = q.get('choices', [])
                answer = q.get('answer')
                category = q.get('category', [])
                if isinstance(category, list):
                    category = category[0] if category else ''

                if not choices or len(choices) < 2:
                    continue

                # Get original ground truth
                gt_letter_orig = normalize_gt_letter(choices, answer)
                if gt_letter_orig is None:
                    continue

                # Add "cannot answer" option for non-yes/no questions
                choices_with_option = add_cannot_answer_option(question_text, choices)

                # Build prompt
                prompt_text = build_image_qa_prompt(question_text, choices_with_option)

                # Generate answer
                model_output = generate_answer_server(
                    server_url=server_url,
                    images=images,
                    question_text=prompt_text,
                    max_tokens=max_tokens,
                    temperature=temperature
                )

                # Extract predicted letter
                predicted_letter = extract_letter(model_output, len(choices_with_option))

                # Check correctness
                is_correct = False
                is_cannot_answer = False
                model_answer_text = None
                score = 0.0

                if predicted_letter is not None:
                    pred_idx = LETTER_ALPH.find(predicted_letter)
                    if 0 <= pred_idx < len(choices_with_option):
                        model_answer_text = str(choices_with_option[pred_idx])

                        if model_answer_text == CANNOT_ANSWER_TEXT:
                            is_cannot_answer = True
                            score = (1.0 / len(choices)) + 0.05
                        elif predicted_letter == gt_letter_orig:
                            is_correct = True
                            score = 1.0
                        else:
                            score = 0.0

                # Save result
                result_entry = {
                    'question': question_text,
                    'choices': choices,
                    'ground_truth': answer,
                    'gt_letter': gt_letter_orig,
                    'predicted_letter': predicted_letter,
                    'model_answer': model_answer_text,
                    'model_output': model_output,
                    'is_correct': is_correct,
                    'is_cannot_answer': is_cannot_answer,
                    'score': round(score, 4),
                    'category': category
                }
                entry_results.append(result_entry)

                # Update counters
                total_count += 1
                if is_correct:
                    correct_count += 1
                if is_cannot_answer:
                    cannot_answer_count += 1

            # Save all results for this entry
            results[image_key] = entry_results

            # Save incrementally
            if total_count % save_every == 0:
                with open(output_path, 'w') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)

                # Compute and save metrics
                metrics = compute_metrics(results)
                metrics['model'] = model_name
                metrics['split'] = split
                metrics['dataset'] = dataset_name
                metrics['server_url'] = server_url

                metrics_path = output_path.replace('.json', '_metrics.json')
                with open(metrics_path, 'w') as f:
                    json.dump(metrics, f, indent=2, ensure_ascii=False)

        except Exception as e:
            print(f"\nError processing entry {image_key}: {e}")
            continue

    # Final save
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Compute metrics
    metrics = compute_metrics(results)
    metrics['model'] = model_name
    metrics['split'] = split
    metrics['dataset'] = dataset_name
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
    print(f"'Cannot answer' selections: {cannot_answer_count}")
    print(f"Overall accuracy: {metrics['overall_accuracy']:.4f}")
    print(f"Average score: {metrics['average_score']:.4f}")
    print(f"Results saved to: {output_path}")
    print(f"Metrics saved to: {metrics_path}")
    print(f"Total time: {elapsed:.2f}s ({elapsed/60:.2f}m)")
    if total_count > 0:
        print(f"Average time per question: {elapsed/total_count:.2f}s")
    print(f"{'='*60}")

    # Print per-category metrics
    if metrics['category_metrics']:
        print(f"\nPer-category metrics:")
        print(f"{'='*60}")
        for category, stats in sorted(metrics['category_metrics'].items()):
            print(f"{category}: {stats['correct']}/{stats['total']} ({stats['accuracy']:.2%}), "
                  f"avg_score={stats['average_score']:.4f}, cannot_answer={stats['cannot_answer']}")
        print(f"{'='*60}")


def compute_metrics(results: Dict[str, List]) -> Dict[str, Any]:
    """Compute evaluation metrics from results."""
    total_questions = sum(len(v) for v in results.values())
    total_score = sum(sum(item.get("score", 0.0) for item in v) for v in results.values())
    correct_answers = sum(
        sum(1 for item in v if item.get("is_correct")) for v in results.values()
    )
    cannot_answer_count = sum(
        sum(1 for item in v if item.get("is_cannot_answer")) for v in results.values()
    )
    overall_accuracy = correct_answers / total_questions if total_questions > 0 else 0.0
    average_score = total_score / total_questions if total_questions > 0 else 0.0

    # Compute per-category metrics
    category_metrics = {}
    for image_key, items in results.items():
        for item in items:
            category = item.get('category', 'unknown')
            if category not in category_metrics:
                category_metrics[category] = {
                    'total': 0,
                    'correct': 0,
                    'cannot_answer': 0,
                    'total_score': 0.0
                }
            category_metrics[category]['total'] += 1
            category_metrics[category]['total_score'] += item.get('score', 0.0)
            if item.get('is_correct'):
                category_metrics[category]['correct'] += 1
            if item.get('is_cannot_answer'):
                category_metrics[category]['cannot_answer'] += 1

    # Compute accuracy and average score per category
    for category, stats in category_metrics.items():
        stats['accuracy'] = stats['correct'] / stats['total'] if stats['total'] > 0 else 0.0
        stats['average_score'] = stats['total_score'] / stats['total'] if stats['total'] > 0 else 0.0

    return {
        'total_questions': total_questions,
        'correct_answers': correct_answers,
        'overall_accuracy': round(overall_accuracy, 4),
        'cannot_answer_count': cannot_answer_count,
        'total_score': round(total_score, 2),
        'average_score': round(average_score, 4),
        'category_metrics': category_metrics
    }


def main():
    parser = argparse.ArgumentParser(
        description="Image-based Question Answering using Qwen3VL via vLLM server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate on natural split
  python qa_image_qwen3vl_server.py \\
    --dataset "Borise/CaptionQA" \\
    --split "natural" \\
    --server-url "http://localhost:8000" \\
    --output-dir "./outputs"

  # Evaluate on all splits
  python qa_image_qwen3vl_server.py \\
    --dataset "Borise/CaptionQA" \\
    --split "all" \\
    --server-url "http://10.1.64.88:8000" \\
    --output-dir "./outputs" \\
    --model-name "Qwen3-VL-4B-Instruct"
        """
    )

    # Required arguments
    parser.add_argument(
        '--dataset',
        type=str,
        default="Borise/CaptionQA",
        help='HuggingFace dataset name (default: Borise/CaptionQA)'
    )
    parser.add_argument(
        '--split',
        type=str,
        default="all",
        choices=["natural", "document", "ecommerce", "embodiedai", "all"],
        help='Domain split to evaluate (default: all)'
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
        default='./outputs',
        help='Directory to save output files (default: ./outputs)'
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
        '--temperature',
        type=float,
        default=0.0,
        help='Sampling temperature (default: 0.0 for deterministic)'
    )
    parser.add_argument(
        '--save-every',
        type=int,
        default=10,
        help='Save incremental results every N questions (default: 10)'
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing results'
    )

    args = parser.parse_args()

    # Run evaluation
    evaluate_questions(
        dataset_name=args.dataset,
        split=args.split,
        output_dir=args.output_dir,
        server_url=args.server_url,
        model_name=args.model_name,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        overwrite=args.overwrite,
        save_every=args.save_every
    )


if __name__ == "__main__":
    main()
