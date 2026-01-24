# Image-Based QA with Qwen3VL

This guide explains how to perform visual question answering (VQA) using **images directly** (no captioning) with the CaptionQA dataset and Qwen3VL model.

## Overview

The new `qa_image.py` script follows the same pattern as `caption.py` and `qa.py`, but uses **images directly** instead of captions.

### Workflow Comparison

**Traditional Caption-Based Approach:**
```
caption.py → Generate captions from images
    ↓
qa.py → Use captions to answer questions
```

**New Image-Based Approach:** ✅
```
qa_image.py → Use images directly to answer questions
```

## Quick Start

### 1. Start vLLM Server

```bash
# Simple command
vllm serve Qwen/Qwen3-VL-8B-Instruct

# Or with SLURM
sbatch start_vllm_server.sh
```

### 2. Run Image-Based QA

```bash
# Direct command
python qa_image.py \
  --split natural \
  --vllm-server-url http://localhost:8000

# Or with SLURM
sbatch run_qa_image.sh
```

That's it! No captioning needed.

## Usage

### Basic Usage

```bash
python qa_image.py \
  --split natural \
  --vllm-server-url http://localhost:8000
```

### All Available Options

```bash
python qa_image.py \
  --dataset "Borise/CaptionQA" \
  --split all \
  --vllm-server-url http://localhost:8000 \
  --output-dir ./outputs \
  --model "Qwen/Qwen3-VL-8B-Instruct" \
  --max-tokens 128 \
  --seed 0 \
  --save-every 10
```

### Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--dataset` | str | Borise/CaptionQA | HuggingFace dataset name |
| `--split` | str | all | Domain split (natural, document, ecommerce, embodiedai, all) |
| `--vllm-server-url` | str | required | vLLM server URL (e.g., http://localhost:8000) |
| `--output-dir` | str | ./outputs | Output directory |
| `--output-path` | str | auto | Full output path (auto-generated if not specified) |
| `--model` | str | Qwen/Qwen3-VL-8B-Instruct | Model name |
| `--max-tokens` | int | 128 | Maximum tokens to generate |
| `--seed` | int | 0 | Random seed for shuffling |
| `--save-every` | int | 10 | Save every N questions |

## Complete Workflow

```bash
# ============================================
# Step 1: Start vLLM Server
# ============================================
cd CAPTIONING/CaptionQA

# Option A: Direct command
vllm serve Qwen/Qwen3-VL-8B-Instruct --port 8000

# Option B: SLURM
sbatch start_vllm_server.sh

# Verify server is running
curl http://localhost:8000/v1/models

# ============================================
# Step 2: Run Image-Based QA
# ============================================

# Evaluate on natural split
python qa_image.py \
  --split natural \
  --vllm-server-url http://localhost:8000

# Evaluate on all splits
python qa_image.py \
  --split all \
  --vllm-server-url http://localhost:8000

# ============================================
# Step 3: Check Results
# ============================================
cat outputs/all_qa_image_Qwen3-VL-8B-Instruct.json
cat outputs/all_qa_image_Qwen3-VL-8B-Instruct_metrics.json
```

## Features

### 1. Direct Image Understanding
- Model sees actual images, not text descriptions
- Better accuracy for visual questions
- No information loss from captioning

### 2. Automatic Answer Shuffling
- Randomizes answer order to prevent bias
- Tracks permutations for correct scoring
- Same shuffling behavior as `qa.py`

### 3. "Cannot Answer" Option
- Adds extra option for non-yes/no questions
- Allows model to abstain when uncertain
- Partial credit scoring

### 4. Incremental Saving
- Saves results every N questions (default: 10)
- Automatic resume on interruption
- Progress tracking

### 5. Per-Category Metrics
- Overall accuracy
- Per-category breakdown
- "Cannot answer" tracking
- Score-based evaluation

## Output Structure

```
outputs/
├── natural_qa_image_Qwen3-VL-8B-Instruct.json
├── natural_qa_image_Qwen3-VL-8B-Instruct_metrics.json
├── all_qa_image_Qwen3-VL-8B-Instruct.json
└── all_qa_image_Qwen3-VL-8B-Instruct_metrics.json
```

### Results Format

```json
{
  "nat_001": [
    {
      "question": "What color is the car?",
      "choices": ["red", "blue", "green"],
      "ground_truth": "red",
      "model_answer": "red",
      "model_response": "The car in the image is red. Answer: A",
      "is_correct": true,
      "is_cannot_answer": false,
      "score": 1.0,
      "category": "Color Recognition"
    }
  ]
}
```

### Metrics Format

```json
{
  "model": "Qwen/Qwen3-VL-8B-Instruct",
  "split": "natural",
  "server_url": "http://localhost:8000",
  "total_questions": 1000,
  "correct_answers": 850,
  "overall_accuracy": 0.8500,
  "cannot_answer_count": 50,
  "total_score": 875.5,
  "average_score": 0.8755,
  "category_metrics": {
    "Color Recognition": {
      "total": 200,
      "correct": 180,
      "cannot_answer": 5,
      "total_score": 185.25,
      "accuracy": 0.9000,
      "average_score": 0.9263
    }
  }
}
```

## Script Comparison

### qa.py (Caption-Based)
```python
# Requires captions
python qa.py \
  --caption-path captions.json \
  --split natural

# Uses text-only model
# Model: Qwen2.5-72B-Instruct
# Input: Caption text
# Cannot see images
```

### qa_image.py (Image-Based) ✅
```python
# No captions needed
python qa_image.py \
  --vllm-server-url http://localhost:8000 \
  --split natural

# Uses vision-language model
# Model: Qwen3-VL-8B-Instruct
# Input: Actual images
# Can see visual content
```

## How It Works

### 1. Dataset Loading
Loads CaptionQA dataset from HuggingFace:
```python
dataset = load_dataset("Borise/CaptionQA", split="natural")
```

Each entry contains:
- `image`: PIL Image object(s)
- `questions`: List of questions with choices and answers
- `id`: Unique identifier
- `category`: Question category

### 2. Question Processing

For each question:

**a) Add "Cannot Answer" Option**
```python
# Original choices
["red", "blue", "green"]

# With "cannot answer" (for non-yes/no questions)
["red", "blue", "green", "Cannot answer from the image"]
```

**b) Shuffle Choices**
```python
# Track permutation for scoring
perm = [2, 0, 3, 1]  # Shuffled order
shuffled = [choices[i] for i in perm]
```

**c) Build Visual Prompt**
```
Question:
What color is the car?

Options:
A. Green
B. Red
C. Cannot answer from the image
D. Blue

Answer with just the letter (A, B, C, etc.) of the correct option.
```

### 3. vLLM Server Call

**Encode Image**
```python
# Convert PIL Image to base64
image_base64 = encode_image_base64(pil_image)
```

**Send Request**
```python
response = requests.post(
    f"{server_url}/v1/chat/completions",
    json={
        "model": "Qwen/Qwen3-VL-8B-Instruct",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
                    {"type": "text", "text": prompt}
                ]
            }
        ],
        "max_tokens": 128
    }
)
```

### 4. Answer Extraction

Extract letter from model output:
```python
# Model output: "The car is red. Answer: B"
# Extract: "B"
# Map back through permutation: B → shuffled[1] → original[0] → "red"
```

### 5. Scoring

```python
if predicted == ground_truth:
    score = 1.0  # Correct
elif predicted == "Cannot answer":
    score = (1.0 / n_choices) + 0.05  # Partial credit
else:
    score = 0.0  # Incorrect
```

## Evaluation Splits

| Split | Description | Examples |
|-------|-------------|----------|
| `natural` | Natural images | Photos, scenes, objects |
| `document` | Document/text images | Forms, diagrams, charts |
| `ecommerce` | Product images | Product photos, listings |
| `embodiedai` | Robot scenarios | Robot views, actions |
| `all` | All combined | All of the above |

## Resume Support

The script automatically resumes if interrupted:

```bash
# First run (processes 500/1000, then crashes)
python qa_image.py --split all --vllm-server-url http://localhost:8000

# Output:
# [resume] loaded=500 | avg_score=0.8500 | accuracy=85.00%
# Already processed: 500; remaining: 500

# Second run (automatically resumes from 501)
python qa_image.py --split all --vllm-server-url http://localhost:8000

# Output:
# [resume] loaded=500 | avg_score=0.8500 | accuracy=85.00%
# Already processed: 500; remaining: 500
# Evaluating: 100%|██████████| 500/500 [03:45<00:00, 2.22q/s]
```

## Monitor Progress

```bash
# Watch SLURM output
tail -f slurm-*-qa-image.out

# Expected output:
Evaluating: 45%|████████▌     | 450/1000 [02:15<02:45, 3.33q/s]

[batch saved] processed=450 | total_score=425.50 | avg_score=0.9456 | accuracy=92.22% | cannot_answer=15
```

## Multiple Splits

Evaluate all splits separately:

```bash
# Natural images
python qa_image.py --split natural --vllm-server-url http://localhost:8000

# Document images
python qa_image.py --split document --vllm-server-url http://localhost:8000

# E-commerce images
python qa_image.py --split ecommerce --vllm-server-url http://localhost:8000

# Embodied AI scenarios
python qa_image.py --split embodiedai --vllm-server-url http://localhost:8000

# All combined
python qa_image.py --split all --vllm-server-url http://localhost:8000
```

## Comparison: Caption-Based vs Image-Based

### Performance Comparison

Run both approaches to compare:

```bash
# ============================================
# Caption-Based QA
# ============================================

# Step 1: Generate captions
python caption.py \
  --dataset "Borise/CaptionQA" \
  --split natural \
  --model gpt-4o \
  --prompt SIMPLE

# Step 2: Run caption-based QA
python qa.py \
  --caption-path outputs/captions/gpt-4o/gpt-4o_simple.json \
  --split natural

# ============================================
# Image-Based QA
# ============================================

# Single step: Run image-based QA
python qa_image.py \
  --split natural \
  --vllm-server-url http://localhost:8000

# ============================================
# Compare Results
# ============================================

# Caption-based accuracy
cat outputs/captions/gpt-4o/cap_simple-gpt-4o__qa_Qwen2.5-72B-Instruct_metrics.json

# Image-based accuracy
cat outputs/natural_qa_image_Qwen3-VL-8B-Instruct_metrics.json
```

### Expected Differences

| Aspect | Caption-Based | Image-Based |
|--------|--------------|-------------|
| **Steps** | 2 (caption + qa) | 1 (qa only) |
| **Model Size** | 72B (text) | 8B (vision) |
| **Accuracy** | 70-80% | 75-85% |
| **Speed** | Slower (2 steps) | Faster (1 step) |
| **Fine Details** | Limited by caption | Full visual access |
| **Best For** | Text-rich questions | Visual questions |

## Troubleshooting

### Issue: Server Connection Error
```bash
# Check server is running
curl http://localhost:8000/v1/models

# Expected output:
{
  "data": [{"id": "Qwen/Qwen3-VL-8B-Instruct"}]
}

# If not running, start server
vllm serve Qwen/Qwen3-VL-8B-Instruct
```

### Issue: Out of Memory
```bash
# Start server with reduced memory
vllm serve Qwen/Qwen3-VL-8B-Instruct \
  --gpu-memory-utilization 0.8 \
  --max-model-len 4096
```

### Issue: Slow Inference
```bash
# Check server GPU usage
nvidia-smi

# Reduce max tokens
python qa_image.py \
  --max-tokens 64 \
  --vllm-server-url http://localhost:8000
```

### Issue: Dataset Loading Error
```bash
# Check HuggingFace cache
ls ~/.cache/huggingface/datasets/

# Re-download if needed
rm -rf ~/.cache/huggingface/datasets/Borise___caption_qa
python qa_image.py --vllm-server-url http://localhost:8000
```

## Advanced Usage

### Custom Output Path
```bash
python qa_image.py \
  --split natural \
  --vllm-server-url http://localhost:8000 \
  --output-path ./custom/results.json
```

### Different Model
```bash
# Start server with different model
vllm serve Qwen/Qwen3-VL-30B-Instruct

# Run QA
python qa_image.py \
  --vllm-server-url http://localhost:8000 \
  --model "Qwen/Qwen3-VL-30B-Instruct"
```

### Save More Frequently
```bash
python qa_image.py \
  --vllm-server-url http://localhost:8000 \
  --save-every 5  # Save every 5 questions
```

## Summary

**Key Points:**
- ✅ Use `qa_image.py` to skip captioning entirely
- ✅ Requires vLLM server running Qwen3VL
- ✅ Uses images directly for better accuracy
- ✅ Same interface pattern as `caption.py` and `qa.py`
- ✅ Automatic resume and progress tracking
- ✅ Per-category performance metrics

**Simple Workflow:**
```bash
# 1. Start server
vllm serve Qwen/Qwen3-VL-8B-Instruct

# 2. Run image QA
python qa_image.py --vllm-server-url http://localhost:8000

# Done!
```

No captioning needed! 🚀
