#!/usr/bin/env python3
"""
Question Answering using Qwen3VL-4B-Instruct model.

This script evaluates VQA questions using the Qwen3VL model with images.
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, Any, List, Optional
import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
import time
import pandas as pd

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


def extract_letter(answer_text: str, num_options: int) -> Optional[str]:
    """
    Extract answer letter from model output.

    Args:
        answer_text: Model's generated response
        num_options: Number of available choices

    Returns:
        Extracted letter (A, B, C, etc.) or None if extraction failed
    """
    if not answer_text:
        return None

    # If response contains </think>, extract letter from text after it
    if "</think>" in answer_text:
        after_think = answer_text.split("</think>", 1)[1]
        answer_text = after_think

    if "Answer: " in answer_text:
        after_answer = answer_text.split("Answer: ", 1)[1]
        answer_text = after_answer

    if "\n" in answer_text:
        after_n = answer_text.split("\n", 1)[0]  # Take first line
        answer_text = after_n

    # Look for letter pattern (A, B, C, etc.)
    m = re.search(r"\b([A-Z])\b", answer_text.upper())
    if m:
        letter = m.group(1)
        idx = LETTER_ALPH.find(letter)
        if 0 <= idx < max(1, num_options):
            return letter

    # Look for number pattern (1, 2, 3, etc.)
    m = re.search(r"\b([1-9][0-9]?)\b", answer_text)
    if m:
        k = int(m.group(1))
        if 1 <= k <= max(1, num_options):
            return LETTER_ALPH[k - 1]

    return None


def extract_answer_from_boxed(answer_str: str) -> Optional[str]:
    """
    Extract answer from \\boxed{X} format.

    Args:
        answer_str: Answer string in format like "\\boxed{A}"

    Returns:
        Extracted letter or None
    """
    if not answer_str:
        return None

    # Match \boxed{X} pattern
    match = re.search(r'\\boxed\{([A-Z])\}', answer_str)
    if match:
        return match.group(1)

    # Try direct letter match
    match = re.search(r'\b([A-Z])\b', answer_str.upper())
    if match:
        return match.group(1)

    return None


def parse_question_choices(question_text: str) -> tuple[str, List[str]]:
    """
    Parse question text to extract question and choices.

    Args:
        question_text: Full question text with <image> token and choices

    Returns:
        Tuple of (question_only, list_of_choices)

    Example:
        Input: "<image>\nWhat color is the sky?\nChoices:\n(A) blue\n(B) red"
        Output: ("What color is the sky?", ["blue", "red"])
    """
    # Remove <image> token
    text = question_text.replace("<image>", "").strip()

    # Split by "Choices:" if present
    if "Choices:" in text or "choices:" in text.lower():
        parts = re.split(r'[Cc]hoices:\s*\n', text, maxsplit=1)
        question_only = parts[0].strip()

        if len(parts) > 1:
            choices_text = parts[1].strip()
            # Parse choices like "(A) option1\n(B) option2"
            choice_pattern = r'\([A-Z]\)\s*(.+?)(?=\n\([A-Z]\)|$)'
            matches = re.findall(choice_pattern, choices_text, re.DOTALL)
            choices = [m.strip() for m in matches]

            if choices:
                return question_only, choices

    # If no choices found, return question as-is with empty choices
    return text, []


def build_qa_prompt(question: str, choices: List[str]) -> str:
    """
    Build prompt for QA with choices.

    Args:
        question: Question text
        choices: List of choice strings

    Returns:
        Formatted prompt string
    """
    if not choices:
        return question

    lines = [f"{LETTER_ALPH[i]}. {choice}" for i, choice in enumerate(choices)]

    prompt = f"""{question}

Options:
{chr(10).join(lines)}

Answer with just the letter (A, B, C, etc.) of the correct option."""

    return prompt


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


def generate_answer(
    image_paths: List[str],
    question_text: str,
    model,
    processor,
    max_new_tokens: int = 128
) -> str:
    """
    Generate answer for a VQA question.

    Args:
        image_paths: List of paths to image files (can be single or multiple)
        question_text: Question text with prompt
        model: Loaded Qwen3VL model
        processor: Loaded processor
        max_new_tokens: Maximum number of tokens to generate

    Returns:
        Generated answer text
    """
    # Build content with all images
    content = []
    for img_path in image_paths:
        content.append({"type": "image", "image": str(img_path)})
    content.append({"type": "text", "text": question_text})

    # Build messages
    messages = [
        {
            "role": "user",
            "content": content,
        }
    ]

    # Build the text prompt
    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # Load/prepare vision inputs
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


def load_questions_from_parquet(question_path: str) -> pd.DataFrame:
    """
    Load questions from parquet file.

    Args:
        question_path: Path to parquet file

    Returns:
        DataFrame with questions
    """
    print(f"Loading questions from {question_path}...")
    df = pd.read_parquet(question_path)
    print(f"Loaded {len(df)} questions")

    # Show available columns
    print(f"Columns: {list(df.columns)}")

    return df


def evaluate_questions(
    question_path: str,
    dataset_path: str,
    output_path: str,
    model_path: str = "Qwen/Qwen3-VL-4B-Instruct",
    max_new_tokens: int = 128,
    overwrite: bool = False
):
    """
    Evaluate VQA questions using Qwen3VL model.

    Args:
        question_path: Path to parquet file with questions
        dataset_path: Base path for dataset images
        output_path: Path to save output JSON file
        model_path: Path or HuggingFace model ID for Qwen3VL
        max_new_tokens: Maximum number of tokens to generate
        overwrite: Whether to overwrite existing results
    """
    start_time = time.time()

    # Load questions
    df = load_questions_from_parquet(question_path)

    # Load model
    model, processor = load_model(model_path)

    # Load existing results if available
    results = {}
    if os.path.exists(output_path) and not overwrite:
        with open(output_path, 'r') as f:
            results = json.load(f)
        print(f"Loaded {len(results)} existing results from {output_path}")

    # Process each question
    print(f"\nProcessing questions...")
    correct_count = 0
    total_count = 0

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):
        qid = row['qid']

        # Skip if already processed and not overwriting
        if qid in results and not overwrite:
            # Count towards total
            total_count += 1
            if results[qid].get('is_correct', False):
                correct_count += 1
            continue

        try:
            # Get image paths (handle numpy arrays, lists, tuples)
            image_rel_paths = row['image']

            # Convert numpy array to list if needed
            if hasattr(image_rel_paths, 'tolist'):
                image_rel_paths = image_rel_paths.tolist()
            elif not isinstance(image_rel_paths, (list, tuple)):
                image_rel_paths = [image_rel_paths]

            # Convert to list and ensure strings
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

            # Generate answer (now handles multiple images)
            model_output = generate_answer(
                image_paths=image_paths,
                question_text=prompt_text,
                model=model,
                processor=processor,
                max_new_tokens=max_new_tokens
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
                'image_paths': image_rel_paths,  # Store list of relative paths
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
    metrics['model'] = model_path
    metrics['question_path'] = question_path
    metrics['dataset_path'] = dataset_path

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
    """
    Compute evaluation metrics from results.

    Args:
        results: Dictionary of results keyed by qid

    Returns:
        Dictionary containing metrics
    """
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
        description="Question Answering using Qwen3VL-4B-Instruct model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate questions with default model
  python qa_qwen3vl.py \\
    --question-path /path/to/questions.parquet \\
    --dataset-path /path/to/dataset

  # Evaluate with custom output directory
  python qa_qwen3vl.py \\
    --question-path /path/to/questions.parquet \\
    --dataset-path /path/to/dataset \\
    --output-dir ./custom_output

  # Evaluate with custom model
  python qa_qwen3vl.py \\
    --question-path /path/to/questions.parquet \\
    --dataset-path /path/to/dataset \\
    --model-path /path/to/local/model

  # Overwrite existing results
  python qa_qwen3vl.py \\
    --question-path /path/to/questions.parquet \\
    --dataset-path /path/to/dataset \\
    --overwrite
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
        '--output-dir',
        type=str,
        default='./output/qa',
        help='Directory to save output files (default: ./output/qa)'
    )

    # Optional arguments
    parser.add_argument(
        '--model-path',
        type=str,
        default='Qwen/Qwen3-VL-4B-Instruct',
        help='Path or HuggingFace model ID for Qwen3VL (default: Qwen/Qwen3-VL-4B-Instruct)'
    )
    parser.add_argument(
        '--max-new-tokens',
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

    # Construct output path based on model name
    model_safe = make_model_safe(args.model_path)
    output_dir = os.path.join(args.output_dir, model_safe)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{model_safe}_qa_results.json")

    print(f"Output directory: {output_dir}")
    print(f"Output file: {output_path}")

    # Run evaluation
    evaluate_questions(
        question_path=args.question_path,
        dataset_path=args.dataset_path,
        output_path=output_path,
        model_path=args.model_path,
        max_new_tokens=args.max_new_tokens,
        overwrite=args.overwrite
    )


if __name__ == "__main__":
    main()
