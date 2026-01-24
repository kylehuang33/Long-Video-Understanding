"""
QA Evaluation with Images using Borise/CaptionQA Dataset

Evaluates questions using images directly (not captions).
Each question is answered by a vision-language model using the actual image.

Features:
- Loads questions from Hugging Face dataset (Borise/CaptionQA)
- Uses images directly (no captioning required)
- Uses Qwen3VL via vLLM server for evaluation
- Adds "Cannot answer from the image" option to non-yes/no questions
- Automatic shuffling of answer choices (with order tracking)
- Per-category performance metrics

Usage:
    # Evaluate on a specific domain with vLLM server
    python qa_image.py \
        --split natural \
        --vllm-server-url http://localhost:8000 \
        --output-path results.json

    # Evaluate on all domains
    python qa_image.py \
        --split all \
        --vllm-server-url http://localhost:8000 \
        --output-path results.json
"""

import os
import json
import re
import argparse
import random
import base64
import tempfile
from typing import Dict, Any, List, Optional
from tqdm import tqdm
from datasets import load_dataset
from PIL import Image
import requests

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
    2. The question starts with common yes/no question words (is/are, do/does/did,
       have/has, can/could, will/would, should)
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


def call_vllm_server(
    server_url: str,
    images: List[Any],
    question_text: str,
    max_tokens: int = 128,
    temperature: float = 0.0,
    model: str = DEFAULT_EVAL_MODEL
) -> str:
    """
    Call vLLM server API for image-based QA.

    Args:
        server_url: Base URL of vLLM server
        images: List of images (can be PIL Image objects or file paths)
        question_text: Question text with prompt
        max_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature
        model: Model name

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


def model_base_name(model: str) -> str:
    if not model:
        return ""
    trimmed = model.strip().rstrip("/\\")
    return os.path.basename(trimmed)


def derive_output_path(split: str, model: str, output_dir: str = ".") -> str:
    """Derive output path based on split and model."""
    model_base = model_base_name(model)
    qa_tag = f"qa_image_{model_base}" if model_base else "qa_image"
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

    # Load the specified split directly (including "all" which combines all domains)
    dataset = load_dataset(dataset_name, split=split)

    print(f"Loaded {len(dataset)} entries")
    return dataset


# ---------- Main Evaluation Function ----------

def evaluate_qa_with_images(args):
    """
    Evaluate questions using images directly.
    Each question is answered once with shuffled choices.
    """

    # Load dataset from HuggingFace
    dataset = load_captionqa_dataset(args.dataset, args.split)

    model_id = args.model
    server_url = args.vllm_server_url

    print(f"Using vLLM server: {server_url}")
    print(f"Using model: {model_id}")

    # Setup RNG for shuffling
    rng = random.Random(args.seed)

    # Prepare questions
    print("Preparing questions...")
    prompts: List[str] = []
    meta: List[tuple] = []  # (image_id, q_idx, perm, n_opts, gt_idx_orig, original_question_data, images)

    skipped_no_image = 0
    skipped_no_choices = 0

    for entry in dataset:
        # Get image identifier (e.g., "nat_001", "doc_042")
        image_id = entry.get('id')
        if image_id is None:
            continue

        image_key = str(image_id)

        # Get image(s) - CaptionQA dataset has 'image' field with PIL Image(s)
        image = entry.get('image')
        if image is None:
            skipped_no_image += 1
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

        for q_idx, q in enumerate(questions):
            question_text = q.get('question', '')
            choices = q.get('choices', [])
            answer = q.get('answer')
            # category can be a list or string
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

            prompt = build_image_qa_prompt(question_text, shuffled_opts)

            prompts.append(prompt)
            meta.append((image_key, q_idx, perm, n_opts, gt_idx_orig, {
                'question': question_text,
                'choices': choices,
                'answer': answer,
                'category': category
            }, images))

    print(f"Prepared {len(prompts)} questions")
    print(f"Skipped: {skipped_no_image} (no image), {skipped_no_choices} (no choices)")

    if not prompts:
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

    # Determine which (image_key, q_idx) still need processing
    indices_to_process = []
    for i, (image_key, q_idx, _perm, _n_opts, _gt_idx_orig, _q_data, _images) in enumerate(meta):
        done = processed_count.get(image_key, 0)
        if q_idx >= done:
            indices_to_process.append(i)

    total_remaining = len(indices_to_process)
    print(f"Already processed: {len(prompts) - total_remaining}; remaining: {total_remaining}")

    if total_remaining == 0:
        # Print summary from existing results and exit
        print_final_summary(results, model_id)
        return

    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)

    # Process questions and save incrementally
    batch_size = max(1, int(args.save_every))
    progress = tqdm(total=total_remaining, desc="Evaluating", unit="q")

    try:
        for idx_pos, i in enumerate(indices_to_process):
            image_key, q_idx, perm, n_opts, gt_idx_orig, q_data, images = meta[i]
            prompt = prompts[i]

            try:
                # Call vLLM server with image
                resp = call_vllm_server(
                    server_url=server_url,
                    images=images,
                    question_text=prompt,
                    max_tokens=args.max_tokens,
                    temperature=0.0,
                    model=model_id
                )

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

                progress.update(1)

            except Exception as e:
                print(f"\nError processing question {i}: {e}")
                progress.update(1)
                continue

            # Save results after each batch
            if (idx_pos + 1) % batch_size == 0:
                with open(args.output_path, "w", encoding="utf-8") as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)

                # Compute and save metrics
                metrics = compute_metrics(results)
                metrics['model'] = model_id
                metrics['split'] = args.split
                metrics['server_url'] = server_url

                metrics_path = args.output_path.replace('.json', '_metrics.json')
                with open(metrics_path, "w", encoding="utf-8") as f:
                    json.dump(metrics, f, indent=2, ensure_ascii=False)

                # Print running totals
                print(
                    f"\n[batch saved] processed={metrics['total_questions']} | total_score={metrics['total_score']:.2f} "
                    f"| avg_score={metrics['average_score']:.4f} | accuracy={metrics['overall_accuracy']:.2%} "
                    f"| cannot_answer={metrics['cannot_answer_count']}"
                )
    finally:
        progress.close()

    # Final save
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Final summary and save final metrics
    print_final_summary(results, model_id)

    # Save final metrics
    metrics = compute_metrics(results)
    metrics['model'] = model_id
    metrics['split'] = args.split
    metrics['server_url'] = server_url

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
        description="Evaluate questions using images directly with Borise/CaptionQA dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate on natural split with vLLM server
  python qa_image.py \\
    --split natural \\
    --vllm-server-url http://localhost:8000

  # Evaluate on all splits
  python qa_image.py \\
    --split all \\
    --vllm-server-url http://localhost:8000 \\
    --output-path ./outputs/qa_all.json
        """
    )

    # Dataset configuration
    parser.add_argument("--dataset", type=str, default="Borise/CaptionQA",
                       help="HuggingFace dataset name (default: Borise/CaptionQA)")
    parser.add_argument("--split", type=str, default="all",
                       choices=["natural", "document", "ecommerce", "embodiedai", "all"],
                       help="Domain split to evaluate (default: all)")

    # vLLM server configuration
    parser.add_argument("--vllm-server-url", type=str, required=True,
                       help="vLLM server URL (e.g., http://localhost:8000)")

    # Output
    parser.add_argument("--output-path", type=str, default=None,
                       help="Path to save evaluation results (default: <split>_qa_image_<model>.json)")
    parser.add_argument("--output-dir", type=str, default="./outputs",
                       help="Output directory (default: ./outputs)")

    # Evaluation parameters
    parser.add_argument("--model", type=str, default=DEFAULT_EVAL_MODEL,
                       help=f"Model name (default: {DEFAULT_EVAL_MODEL})")
    parser.add_argument("--max-tokens", type=int, default=128,
                       help="Maximum tokens for response (default: 128)")
    parser.add_argument("--seed", type=int, default=0,
                       help="Random seed for option shuffling (default: 0)")
    parser.add_argument("--save-every", type=int, default=10,
                       help="Save incremental results every N questions (default: 10)")

    args = parser.parse_args()

    # Auto-generate output path if not specified
    if not args.output_path:
        args.output_path = derive_output_path(args.split, args.model, args.output_dir)
        print(f"Auto output path: {args.output_path}")

    evaluate_qa_with_images(args)


if __name__ == "__main__":
    main()
