#!/usr/bin/env python3
"""
Recalculate metrics from existing results JSON file using math_verify.
This script reads an existing results file, re-extracts answers using math_verify,
recalculates accuracy, and saves updated results and metrics files.
"""

import argparse
import json
import os
import re
from typing import Dict, Optional

# Import math_verify for answer extraction and verification
try:
    from math_verify import parse, verify
    from math_verify.parser import LatexExtractionConfig
    MATH_VERIFY_AVAILABLE = True
    print("✓ math_verify is available")
except ImportError:
    MATH_VERIFY_AVAILABLE = False
    print("Warning: math_verify not installed. Using fallback extraction.")
    print("Install with: pip install math-verify")


LETTER_ALPH = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def has_choices(question: str) -> bool:
    """Check if question has multiple choice options."""
    return bool(re.search(r'\([A-Z]\)', question) or 'Choices:' in question or 'choices:' in question.lower())


def count_options(question: str) -> int:
    """Count number of options in question."""
    matches = re.findall(r'\([A-Z]\)', question)
    return len(matches)


def normalize_answer(answer: str) -> str:
    """Normalize answer for comparison."""
    if not answer:
        return ""
    normalized = answer.lower().strip()
    normalized = re.sub(r'\s+', ' ', normalized)
    return normalized


def extract_answer_math_verify(answer_text: str) -> Optional[str]:
    """Extract answer using math_verify library."""
    if not MATH_VERIFY_AVAILABLE or not answer_text:
        return None

    try:
        config = LatexExtractionConfig(
            normalization_config=None,
            boxed_match_priority=0,
            try_extract_without_anchor=True,
        )
        parsed = parse(answer_text, extraction_config=config)
        if parsed:
            return str(parsed)
    except Exception:
        pass

    return None


def extract_answer(answer_text: str, question: str) -> Optional[str]:
    """Extract answer from model output."""
    if not answer_text:
        return None

    cleaned = answer_text.strip()

    # If response contains </think>, extract text after it
    if "</think>" in cleaned:
        cleaned = cleaned.split("</think>", 1)[1].strip()

    # If contains "Answer: ", extract text after it
    if "Answer: " in cleaned:
        cleaned = cleaned.split("Answer: ", 1)[1].strip()
    elif "answer: " in cleaned.lower():
        idx = cleaned.lower().find("answer: ")
        cleaned = cleaned[idx + 8:].strip()

    # Take first line if multiline
    if "\n" in cleaned:
        first_line = cleaned.split("\n", 1)[0].strip()
        if len(first_line) < 50:
            cleaned = first_line

    # Check if question has choices - extract letter
    if has_choices(question):
        num_options = count_options(question)

        m = re.search(r"\b([A-Z])\b", cleaned.upper())
        if m:
            letter = m.group(1)
            idx = LETTER_ALPH.find(letter)
            if 0 <= idx < max(1, num_options):
                return letter

        m = re.search(r"\b([1-9][0-9]?)\b", cleaned)
        if m:
            k = int(m.group(1))
            if 1 <= k <= max(1, num_options):
                return LETTER_ALPH[k - 1]
    else:
        # Open-ended question - try math_verify first
        if MATH_VERIFY_AVAILABLE:
            math_answer = extract_answer_math_verify(cleaned)
            if math_answer:
                return math_answer

        # Fallback: manual extraction
        cleaned_lower = cleaned.lower()
        if cleaned_lower in ['yes', 'no']:
            return cleaned.capitalize()
        if cleaned_lower.startswith('yes,') or cleaned_lower.startswith('yes.') or cleaned_lower.startswith('yes '):
            return 'Yes'
        if cleaned_lower.startswith('no,') or cleaned_lower.startswith('no.') or cleaned_lower.startswith('no '):
            return 'No'

        m = re.search(r'\b(\d+)\b', cleaned)
        if m:
            return m.group(1)

        for prefix in ['the answer is ', 'it is ', 'there are ', 'there is ']:
            if cleaned_lower.startswith(prefix):
                cleaned = cleaned[len(prefix):]
                break

        return cleaned[:100].strip() if cleaned else None

    return None


def extract_ground_truth_math_verify(answer: str) -> Optional[str]:
    """Extract ground truth answer using math_verify library."""
    if not MATH_VERIFY_AVAILABLE or not answer:
        return None

    try:
        config = LatexExtractionConfig(
            normalization_config=None,
            boxed_match_priority=0,
            try_extract_without_anchor=True,
        )
        parsed = parse(answer, extraction_config=config)
        if parsed:
            return str(parsed)
    except Exception:
        pass

    return None


def extract_ground_truth(answer: str, question: str) -> Optional[str]:
    """Extract answer from ground truth."""
    if not answer:
        return None

    # For multiple choice, extract letter directly
    if has_choices(question):
        m = re.search(r'\\boxed\{([A-Z])\}', answer)
        if m:
            return m.group(1)
        m = re.search(r'\b([A-Z])\b', answer.upper())
        if m:
            return m.group(1)
        return None

    # For open-ended questions, try math_verify first
    if MATH_VERIFY_AVAILABLE:
        math_answer = extract_ground_truth_math_verify(answer)
        if math_answer:
            return math_answer

    # Fallback: manual extraction
    m = re.search(r'\\boxed\{([^}]+)\}', answer)
    if m:
        content = m.group(1).strip()
        if '=' in content:
            result = content.split('=')[-1].strip()
            return result
        return content

    m = re.search(r'\b(\d+)\b', answer)
    if m:
        return m.group(1)

    return None


def compare_answers_math_verify(predicted: str, ground_truth: str) -> bool:
    """Compare answers using math_verify library."""
    if not MATH_VERIFY_AVAILABLE:
        return False

    try:
        config = LatexExtractionConfig(
            normalization_config=None,
            boxed_match_priority=0,
            try_extract_without_anchor=True,
        )
        pred_parsed = parse(predicted, extraction_config=config)
        gt_parsed = parse(ground_truth, extraction_config=config)

        if pred_parsed is not None and gt_parsed is not None:
            return verify(pred_parsed, gt_parsed)
    except Exception:
        pass

    return False


def compare_answers(predicted: str, ground_truth: str) -> bool:
    """Compare predicted answer with ground truth."""
    if not predicted or not ground_truth:
        return False

    if predicted == ground_truth:
        return True

    if predicted.lower() == ground_truth.lower():
        return True

    if MATH_VERIFY_AVAILABLE:
        if compare_answers_math_verify(predicted, ground_truth):
            return True

    pred_norm = normalize_answer(predicted)
    gt_norm = normalize_answer(ground_truth)

    if pred_norm == gt_norm:
        return True

    try:
        pred_num = float(predicted)
        gt_num = float(ground_truth)
        if pred_num == gt_num:
            return True
    except (ValueError, TypeError):
        pass

    return False


def compute_metrics(results: Dict) -> Dict:
    """Compute evaluation metrics from results."""
    total_questions = len(results)
    correct_answers = sum(1 for v in results.values() if v.get('is_correct', False))

    overall_accuracy = correct_answers / total_questions if total_questions > 0 else 0.0

    # Per-category metrics
    category_metrics = {}
    for qid, item in results.items():
        category = item.get('category', 'unknown')
        if category not in category_metrics:
            category_metrics[category] = {'total': 0, 'correct': 0}
        category_metrics[category]['total'] += 1
        if item.get('is_correct', False):
            category_metrics[category]['correct'] += 1

    for category, stats in category_metrics.items():
        stats['accuracy'] = stats['correct'] / stats['total'] if stats['total'] > 0 else 0.0

    # Per question type metrics
    type_metrics = {'multiple_choice': {'total': 0, 'correct': 0}, 'open_ended': {'total': 0, 'correct': 0}}
    for qid, item in results.items():
        qtype = item.get('question_type', 'unknown')
        if qtype in type_metrics:
            type_metrics[qtype]['total'] += 1
            if item.get('is_correct', False):
                type_metrics[qtype]['correct'] += 1

    for qtype, stats in type_metrics.items():
        stats['accuracy'] = stats['correct'] / stats['total'] if stats['total'] > 0 else 0.0

    return {
        'total_questions': total_questions,
        'correct_answers': correct_answers,
        'overall_accuracy': round(overall_accuracy, 4),
        'category_metrics': category_metrics,
        'question_type_metrics': type_metrics
    }


def recalculate_results(input_path: str, output_path: str, metrics_path: str):
    """
    Recalculate results from existing JSON file.

    Args:
        input_path: Path to existing results JSON
        output_path: Path to save updated results JSON
        metrics_path: Path to save metrics JSON
    """
    print(f"\nLoading results from: {input_path}")
    with open(input_path, 'r') as f:
        results = json.load(f)
    print(f"Loaded {len(results)} results")

    # Track changes
    changed_count = 0
    newly_correct = 0
    newly_incorrect = 0

    print("\nRecalculating answers...")
    for qid, item in results.items():
        question = item.get('question', item.get('question_original', ''))
        ground_truth = item.get('ground_truth', '')
        raw_output = item.get('raw_output', '')
        old_is_correct = item.get('is_correct', False)

        # Re-extract ground truth
        new_gt_answer = extract_ground_truth(ground_truth, question)

        # Re-extract predicted answer
        new_predicted = extract_answer(raw_output, question)

        # Re-compare
        new_is_correct = compare_answers(new_predicted, new_gt_answer)

        # Determine question type
        question_type = 'multiple_choice' if has_choices(question) else 'open_ended'

        # Update item
        item['ground_truth_answer'] = new_gt_answer
        item['predicted_answer'] = new_predicted
        item['is_correct'] = new_is_correct
        item['question_type'] = question_type

        # Track changes
        if old_is_correct != new_is_correct:
            changed_count += 1
            if new_is_correct and not old_is_correct:
                newly_correct += 1
            elif not new_is_correct and old_is_correct:
                newly_incorrect += 1

    # Compute new metrics
    metrics = compute_metrics(results)
    metrics['math_verify_available'] = MATH_VERIFY_AVAILABLE
    metrics['recalculated_from'] = input_path

    # Save updated results
    print(f"\nSaving updated results to: {output_path}")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Save metrics
    print(f"Saving metrics to: {metrics_path}")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    # Print summary
    print(f"\n{'='*60}")
    print("Recalculation Complete!")
    print(f"{'='*60}")
    print(f"Total questions: {metrics['total_questions']}")
    print(f"Correct answers: {metrics['correct_answers']} ({metrics['overall_accuracy']:.2%})")
    print(f"\nChanges from original:")
    print(f"  Total changed: {changed_count}")
    print(f"  Newly correct: {newly_correct}")
    print(f"  Newly incorrect: {newly_incorrect}")
    print(f"\nBy question type:")
    for qtype, stats in metrics['question_type_metrics'].items():
        print(f"  {qtype}: {stats['correct']}/{stats['total']} ({stats['accuracy']:.2%})")
    print(f"\nBy category:")
    for category, stats in sorted(metrics['category_metrics'].items()):
        print(f"  {category}: {stats['correct']}/{stats['total']} ({stats['accuracy']:.2%})")


def main():
    parser = argparse.ArgumentParser(
        description="Recalculate metrics from existing results JSON using math_verify",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to existing results JSON file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Path to save updated results JSON (default: overwrites input)'
    )
    parser.add_argument(
        '--metrics',
        type=str,
        default=None,
        help='Path to save metrics JSON (default: {input_dir}/{input_name}_metrics.json)'
    )

    args = parser.parse_args()

    input_path = args.input

    # Default output path: overwrite input
    if args.output:
        output_path = args.output
    else:
        output_path = input_path

    # Default metrics path: same directory as input
    if args.metrics:
        metrics_path = args.metrics
    else:
        input_dir = os.path.dirname(input_path)
        input_name = os.path.splitext(os.path.basename(input_path))[0]
        # Remove _results suffix if present
        if input_name.endswith('_results'):
            input_name = input_name[:-8]
        metrics_path = os.path.join(input_dir, f"{input_name}_metrics.json")

    recalculate_results(input_path, output_path, metrics_path)


if __name__ == "__main__":
    main()
