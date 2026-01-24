"""
QA Evaluation with Images using Borise/CaptionQA Dataset (Local Inference)

Evaluates questions using images directly with local model inference (no vLLM server).
Uses Qwen3VL with batch processing for efficient inference.

Features:
- Loads questions from Hugging Face dataset (Borise/CaptionQA)
- Uses images directly (no captioning required)
- Direct model inference with transformers (no server needed)
- Batch processing for efficiency
- Adds "Cannot answer from the image" option to non-yes/no questions
- Automatic shuffling of answer choices (with order tracking)
- Per-category performance metrics

Usage:
    # Evaluate on a specific domain
    python qa_image_local.py \
        --split natural \
        --model Qwen/Qwen3-VL-8B-Instruct \
        --batch-size 4

    # Evaluate on all domains
    python qa_image_local.py \
        --split all \
        --model Qwen/Qwen3-VL-8B-Instruct \
        --batch-size 8 \
        --output-dir ./outputs
"""

import os
import json
import re
import argparse
import random
import tempfile
import warnings
from typing import Dict, Any, List, Optional, Tuple
from tqdm import tqdm
from datasets import load_dataset
from PIL import Image
import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

LETTER_ALPH = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
CANNOT_ANSWER_TEXT = "Cannot answer from the image"

# Default model for evaluation
DEFAULT_EVAL_MODEL = "Qwen/Qwen3-VL-8B-Instruct"

# Available domain splits in the dataset
DOMAIN_SPLITS = ["natural", "document", "ecommerce", "embodiedai"]

# ---------- Helper Functions ----------

def extract_letter(answer_text: str, num_options: int) -> Optional[str]:
    """Extract answer letter from model output."""
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
    1. The choices contain "Yes" and "No" (in any order, possibly with other choices), OR
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


def prepare_images_for_inference(images: List[Any], dataset_base_path: str = None) -> List[str]:
    """Convert PIL Images to temporary file paths for inference.

    Args:
        images: List of PIL Images or file paths
        dataset_base_path: Base path for relative image paths (optional)

    Returns:
        List of file paths
    """
    image_paths = []
    for img in images:
        if isinstance(img, str):
            # Already a file path (might be relative)
            img_path = img
            # If path is relative and dataset_base_path is provided, make it absolute
            if dataset_base_path and not os.path.isabs(img_path):
                img_path = os.path.join(dataset_base_path, img_path)
            image_paths.append(img_path)
        elif isinstance(img, Image.Image):
            # PIL Image, save to temp file
            temp_path = save_pil_image_to_temp(img)
            image_paths.append(temp_path)
        else:
            raise ValueError(f"Unsupported image type: {type(img)}")
    return image_paths


def model_base_name(model: str) -> str:
    if not model:
        return ""
    trimmed = model.strip().rstrip("/\\")
    return os.path.basename(trimmed)


def derive_output_path(split: str, model: str, output_dir: str = ".") -> str:
    """Derive output path based on split and model."""
    model_base = model_base_name(model)
    qa_tag = f"qa_image_local_{model_base}" if model_base else "qa_image_local"
    filename = f"{split}_{qa_tag}.json"
    return os.path.join(output_dir, filename)


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


def batch_generate_answers(
    model,
    processor,
    batch_messages: List[List[Dict]],
    max_new_tokens: int = 128
) -> List[str]:
    """
    Generate answers for a batch of questions.

    Args:
        model: Qwen3VL model
        processor: Qwen3VL processor
        batch_messages: List of message lists (each with images and text)
        max_new_tokens: Maximum tokens to generate

    Returns:
        List of generated answer texts
    """
    # Suppress repeated padding warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*padding.*", category=UserWarning)

        # Build prompts for all messages in batch
        prompts = []
        all_image_inputs = []
        all_video_inputs = []

            for messages in batch_messages:
            # Build text prompt
            prompt = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            prompts.append(prompt)

            # Load vision inputs
            image_inputs, video_inputs = process_vision_info(messages)
            all_image_inputs.extend(image_inputs if image_inputs else [])
            all_video_inputs.extend(video_inputs if video_inputs else [])

        # Prepare model inputs
        inputs = processor(
            text=prompts,
            images=all_image_inputs if all_image_inputs else None,
            videos=all_video_inputs if all_video_inputs else None,
            padding=True,
            return_tensors="pt",
        ).to(model.device)

        # Generate
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)

        # Decode only new tokens
        new_tokens = generated_ids[:, inputs.input_ids.shape[1]:]
        output_texts = processor.batch_decode(new_tokens, skip_special_tokens=True)

        return output_texts


# ---------- Main Evaluation Function ----------

def evaluate_qa_with_images(args):
    """
    Evaluate questions using images directly with local model inference.
    """

    # Load dataset from HuggingFace
    dataset = load_captionqa_dataset(args.dataset, args.split)

    # Determine dataset base path for relative image paths
    dataset_base_path = None
    if os.path.isdir(args.dataset):
        # Local dataset directory
        dataset_base_path = args.dataset
    elif '/' in args.dataset and not args.dataset.startswith(('http://', 'https://')):
        # Might be a HuggingFace dataset with cached files
        # In this case, images should already be loaded as PIL Images
        dataset_base_path = None
    else:
        dataset_base_path = None

    # Load model and processor
    print(f"Loading model {args.model}...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype="auto",
        device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(args.model)

    # Fix padding side for decoder-only architecture
    if hasattr(processor, 'tokenizer'):
        processor.tokenizer.padding_side = 'left'
    elif hasattr(processor, 'padding_side'):
        processor.padding_side = 'left'

    model.eval()
    print(f"Model loaded on device: {model.device}")
    print(f"Processor padding side: {getattr(processor.tokenizer if hasattr(processor, 'tokenizer') else processor, 'padding_side', 'N/A')}")

    # Setup RNG for shuffling
    rng = random.Random(args.seed)

    # Prepare questions
    print("Preparing questions...")
    batch_data = []  # List of (prompt, meta_info)

    skipped_no_image = 0
    skipped_no_choices = 0

    for entry in tqdm(dataset, desc="Preparing questions", unit="entry"):
        # Get image identifier
        image_id = entry.get('id')
        if image_id is None:
            continue

        image_key = str(image_id)

        # Get images from the dataset entry
        # Try 'images' first (PIL Image objects from processed dataset)
        # Fall back to 'image_paths' (string paths from raw dataset)
        images = entry.get('images')
        if images is None:
            images = entry.get('image_paths', [])

        if not images:
            skipped_no_image += 1
            continue

        # Ensure images is a list
        if not isinstance(images, list):
            images = [images]

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

        for q_idx, q in enumerate(questions):
            question_text = q.get('question', '')
            choices = q.get('choices', [])
            answer = q.get('answer')
            category = q.get('category', [])
            if isinstance(category, list):
                category = category[0] if category else ''

            if not choices or len(choices) < 2:
                skipped_no_choices += 1
                continue

            # Get original ground truth
            gt_letter_orig = normalize_gt_letter(choices, answer)
            if gt_letter_orig is None:
                continue
            gt_idx_orig = LETTER_ALPH.index(gt_letter_orig)

            # Add "cannot answer" option for non-yes/no questions
            choices_with_option = add_cannot_answer_option(question_text, choices)

            # Shuffle choices
            n_opts = len(choices_with_option)
            perm = list(range(n_opts))
            rng.shuffle(perm)

            # Create shuffled choices
            shuffled_opts = [choices_with_option[i] for i in perm]

            # Build prompt
            prompt_text = build_image_qa_prompt(question_text, shuffled_opts)

            # Prepare images for inference
            image_paths = prepare_images_for_inference(images, dataset_base_path)

            # Build messages for model
            messages = [
                {
                    "role": "user",
                    "content": [
                        *[{"type": "image", "image": img_path} for img_path in image_paths],
                        {"type": "text", "text": prompt_text}
                    ],
                }
            ]

            meta_info = {
                'image_key': image_key,
                'q_idx': q_idx,
                'perm': perm,
                'n_opts': n_opts,
                'gt_idx_orig': gt_idx_orig,
                'q_data': {
                    'question': question_text,
                    'choices': choices,
                    'answer': answer,
                    'category': category
                },
                'image_paths': image_paths
            }

            batch_data.append((messages, meta_info))

    print(f"Prepared {len(batch_data)} questions")
    print(f"Skipped: {skipped_no_image} (no image), {skipped_no_choices} (no choices)")

    if not batch_data:
        print("No questions to evaluate!")
        return

    # Load existing results if present (auto-resume)
    results = {}
    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    if os.path.exists(args.output_path):
        try:
            with open(args.output_path, "r", encoding="utf-8") as f:
                results = json.load(f) or {}
            print(f"Loaded existing results from {args.output_path} (resume mode)")
        except Exception as e:
            print(f"Warning: could not load existing results ({e}); starting fresh")
            results = {}

    # Map image -> already processed count
    processed_count = {k: len(v) for k, v in results.items() if isinstance(v, list)}

    # Print intermediate totals from loaded results
    if results:
        existing_total = sum(len(v) for v in results.values())
        existing_total_score = sum(sum(item.get("score", 0.0) for item in v) for v in results.values())
        existing_correct = sum(
            sum(1 for item in v if item.get("is_correct")) for v in results.values()
        )
        existing_cannot = sum(
            sum(1 for item in v if item.get("is_cannot_answer")) for v in results.values()
        )
        existing_avg = (existing_total_score / existing_total) if existing_total else 0.0
        existing_acc = (existing_correct / existing_total) if existing_total else 0.0
        print(
            f"[resume] loaded={existing_total} | total_score={existing_total_score:.2f} "
            f"| avg_score={existing_avg:.4f} | accuracy={existing_acc:.2%} | cannot_answer={existing_cannot}"
        )

    # Determine which questions still need processing
    indices_to_process = []
    for i, (messages, meta_info) in enumerate(batch_data):
        image_key = meta_info['image_key']
        q_idx = meta_info['q_idx']
        done = processed_count.get(image_key, 0)
        if q_idx >= done:
            indices_to_process.append(i)

    total_remaining = len(indices_to_process)
    print(f"Already processed: {len(batch_data) - total_remaining}; remaining: {total_remaining}")

    if total_remaining == 0:
        # Print summary from existing results and exit
        print_final_summary(results, args.model)
        return

    # Process in batches
    batch_size = args.batch_size
    num_batches = (total_remaining + batch_size - 1) // batch_size

    print(f"Processing {total_remaining} questions in {num_batches} batches (batch_size={batch_size})...")

    progress = tqdm(total=total_remaining, desc="Evaluating questions", unit="q",
                   bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

    try:
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, total_remaining)
            batch_indices = indices_to_process[start_idx:end_idx]

            # Prepare batch
            batch_messages = []
            batch_meta = []

            for i in batch_indices:
                messages, meta_info = batch_data[i]
                batch_messages.append(messages)
                batch_meta.append(meta_info)

            try:
                # Generate answers for batch
                batch_responses = batch_generate_answers(
                    model=model,
                    processor=processor,
                    batch_messages=batch_messages,
                    max_new_tokens=args.max_tokens
                )

                # Process responses
                for resp, meta_info in zip(batch_responses, batch_meta):
                    image_key = meta_info['image_key']
                    perm = meta_info['perm']
                    n_opts = meta_info['n_opts']
                    gt_idx_orig = meta_info['gt_idx_orig']
                    q_data = meta_info['q_data']

                    # Extract answer
                    letter = extract_letter(resp, n_opts)
                    is_correct = False
                    is_cannot_answer = False
                    model_answer_text = None
                    score = 0.0

                    original_choices = q_data['choices']
                    n_original_choices = len(original_choices)
                    choices_with_option = add_cannot_answer_option(q_data['question'], original_choices)

                    if letter is not None:
                        shuf_idx = LETTER_ALPH.find(letter)
                        if 0 <= shuf_idx < len(perm):
                            orig_idx = perm[shuf_idx]

                            if orig_idx < len(choices_with_option):
                                model_answer_text = str(choices_with_option[orig_idx])

                                if model_answer_text == CANNOT_ANSWER_TEXT:
                                    is_cannot_answer = True
                                    score = (1.0 / n_original_choices) + 0.05
                                elif orig_idx == gt_idx_orig:
                                    is_correct = True
                                    score = 1.0
                                else:
                                    score = 0.0

                    if image_key not in results:
                        results[image_key] = []

                    result_entry = {
                        "question": q_data['question'],
                        "choices": q_data['choices'],
                        "ground_truth": q_data['answer'],
                        "model_answer": model_answer_text,
                        "model_response": resp,
                        "is_correct": is_correct,
                        "is_cannot_answer": is_cannot_answer,
                        "score": round(score, 4),
                        "category": q_data.get('category', '')
                    }
                    results[image_key].append(result_entry)

                # Compute current metrics for progress bar
                metrics = compute_metrics(results)
                progress.set_postfix({
                    'acc': f"{metrics['overall_accuracy']:.2%}",
                    'avg_score': f"{metrics['average_score']:.3f}",
                    'cannot': metrics['cannot_answer_count']
                })
                progress.update(len(batch_indices))

            except Exception as e:
                print(f"\nError processing batch {batch_idx}: {e}")
                progress.update(len(batch_indices))
                continue

            # Save after each batch
            with open(args.output_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

            # Compute and save metrics
            metrics = compute_metrics(results)
            metrics['model'] = args.model
            metrics['split'] = args.split

            metrics_path = args.output_path.replace('.json', '_metrics.json')
            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2, ensure_ascii=False)

            # Print running totals every 10 batches
            if (batch_idx + 1) % 10 == 0 or batch_idx == num_batches - 1:
                print(
                    f"\n[Batch {batch_idx + 1}/{num_batches}] Processed={metrics['total_questions']} | "
                    f"Accuracy={metrics['overall_accuracy']:.2%} | Avg_score={metrics['average_score']:.4f} | "
                    f"Cannot_answer={metrics['cannot_answer_count']}"
                )

    finally:
        progress.close()

        # Clean up temp files
        for messages, meta_info in batch_data:
            for img_path in meta_info.get('image_paths', []):
                if img_path.startswith('/tmp/') or 'tmp' in img_path:
                    try:
                        os.unlink(img_path)
                    except:
                        pass

    # Final summary
    print_final_summary(results, args.model)

    # Save final metrics
    metrics = compute_metrics(results)
    metrics['model'] = args.model
    metrics['split'] = args.split

    metrics_path = args.output_path.replace('.json', '_metrics.json')
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"\nMetrics saved to: {metrics_path}")


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


def print_final_summary(results: Dict[str, List], model_name: str):
    """Print final evaluation summary."""
    metrics = compute_metrics(results)

    print(f"\n{'='*60}")
    print(f"Evaluation Results:")
    print(f"{'='*60}")
    print(f"Model: {model_name}")
    print(f"Total questions: {metrics['total_questions']}")
    print(f"Correct answers: {metrics['correct_answers']} ({metrics['overall_accuracy']:.2%})")
    print(f"'Cannot answer' selections: {metrics['cannot_answer_count']}")
    print(f"Total score: {metrics['total_score']:.2f} / {metrics['total_questions']}")
    print(f"Average score: {metrics['average_score']:.4f}")
    print(f"{'='*60}")
    print(f"\nScoring rules:")
    print(f"  - Correct answer: 1.0 point")
    print(f"  - Incorrect answer: 0.0 points")
    print(f"  - 'Cannot answer': 1/n_choices + 0.05 points")
    print(f"{'='*60}")

    # Print per-category metrics if available
    if metrics['category_metrics']:
        print(f"\nPer-category metrics:")
        print(f"{'='*60}")
        for category, stats in sorted(metrics['category_metrics'].items()):
            print(f"{category}: {stats['correct']}/{stats['total']} ({stats['accuracy']:.2%}), "
                  f"avg_score={stats['average_score']:.4f}, cannot_answer={stats['cannot_answer']}")
        print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate questions using images directly with local model inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate on natural split with batch processing
  python qa_image_local.py \\
    --split natural \\
    --model Qwen/Qwen3-VL-8B-Instruct \\
    --batch-size 4

  # Evaluate on all splits with larger batch
  python qa_image_local.py \\
    --split all \\
    --model Qwen/Qwen3-VL-8B-Instruct \\
    --batch-size 8 \\
    --output-dir ./outputs
        """
    )

    # Dataset configuration
    parser.add_argument("--dataset", type=str, default="/mnt/data-alpha-sg-02/team-agent/ai_glasses/datasets/CaptionQA",
                       help="Dataset path or HuggingFace dataset name (default: local CaptionQA)")
    parser.add_argument("--split", type=str, default="all",
                       choices=["natural", "document", "ecommerce", "embodiedai", "all"],
                       help="Domain split to evaluate (default: all)")

    # Model configuration
    parser.add_argument("--model", type=str, default=DEFAULT_EVAL_MODEL,
                       help=f"Model name (default: {DEFAULT_EVAL_MODEL})")
    parser.add_argument("--batch-size", type=int, default=4,
                       help="Batch size for inference (default: 4)")

    # Output
    parser.add_argument("--output-path", type=str, default=None,
                       help="Path to save evaluation results (default: results/QA_IMAGE/<split>_qa_image_local_<model>.json)")
    parser.add_argument("--output-dir", type=str, default="results/QA_IMAGE",
                       help="Output directory (default: results/QA_IMAGE)")

    # Evaluation parameters
    parser.add_argument("--max-tokens", type=int, default=128,
                       help="Maximum tokens for response (default: 128)")
    parser.add_argument("--seed", type=int, default=0,
                       help="Random seed for option shuffling (default: 0)")

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Auto-generate output path if not specified
    if not args.output_path:
        args.output_path = derive_output_path(args.split, args.model, args.output_dir)
        print(f"Auto output path: {args.output_path}")

    evaluate_qa_with_images(args)


if __name__ == "__main__":
    main()
