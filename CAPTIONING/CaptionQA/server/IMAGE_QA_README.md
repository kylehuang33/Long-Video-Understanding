# Image-Based Question Answering with Qwen3VL

This document explains how to perform visual question answering (VQA) directly from images using the Qwen3-VL-4B-Instruct model via vLLM server on the CaptionQA dataset.

## Overview

Unlike the standard QA approach which uses captions + text-only LLM, this method:
- ✅ Uses **images directly** (not captions)
- ✅ Leverages **vision-language model** (Qwen3VL) to see and understand images
- ✅ Answers questions based on visual content
- ✅ Runs via **vLLM server** for efficient inference
- ✅ Supports **multiple images** per question
- ✅ Provides **per-category accuracy** metrics

## Comparison: Caption-Based vs Image-Based QA

| Aspect | Caption-Based QA | Image-Based QA |
|--------|------------------|----------------|
| **Input** | Text caption | Raw image(s) |
| **Model** | Text-only LLM (Qwen2.5-72B) | Vision-Language Model (Qwen3VL) |
| **Visual Info** | Indirect (via caption) | Direct (sees image) |
| **Fine Details** | Limited by caption | Can see all details |
| **Setup** | Requires captioning first | Direct evaluation |
| **Accuracy** | Depends on caption quality | Ground truth visual understanding |

## Prerequisites

1. **Running vLLM Server**: You need a vLLM server running with Qwen3-VL-4B-Instruct model
   ```bash
   # Start server (see VLLM_SERVER_GUIDE.md in Tiger-VL-Rethinker/)
   sbatch start_vllm_server.sh
   ```

2. **Python Environment**: Install required packages
   ```bash
   conda activate captionqa
   pip install datasets pillow requests tqdm
   ```

## Usage

### Basic Usage

```bash
python qa_image_qwen3vl_server.py \
  --dataset "Borise/CaptionQA" \
  --split "natural" \
  --server-url "http://localhost:8000" \
  --output-dir "./outputs"
```

### Using SLURM

```bash
# Edit run_qa_image_server.sh to configure:
# - SERVER_URL: Your vLLM server URL
# - SPLIT: Dataset split (natural, document, ecommerce, embodiedai, all)
# - OUTPUT_DIR: Where to save results

sbatch run_qa_image_server.sh
```

### Available Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--dataset` | str | Borise/CaptionQA | HuggingFace dataset name |
| `--split` | str | all | Domain split: natural, document, ecommerce, embodiedai, all |
| `--server-url` | str | required | vLLM server URL (e.g., http://localhost:8000) |
| `--output-dir` | str | ./outputs | Output directory |
| `--model-name` | str | Qwen3-VL-4B-Instruct | Model name for output folders |
| `--max-tokens` | int | 128 | Maximum tokens to generate |
| `--temperature` | float | 0.0 | Sampling temperature (0.0 = deterministic) |
| `--save-every` | int | 10 | Save results every N questions |
| `--overwrite` | flag | False | Overwrite existing results |

## How It Works

### 1. Dataset Loading
The script loads questions from the CaptionQA dataset via HuggingFace:
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
- Non-yes/no questions get an extra option: "Cannot answer from the image"
- Yes/no questions remain unchanged

**b) Build Visual Prompt**
```
Question:
What color is the car?

Options:
A. Red
B. Blue
C. Green
D. Cannot answer from the image

Answer with just the letter (A, B, C, etc.) of the correct option.
```

**c) Send to vLLM Server**
- Encode image(s) to base64
- Send via OpenAI-compatible API
- Model sees the image(s) and question together

### 3. Answer Extraction

The model output is parsed to extract the answer letter:
```python
# Model output: "The car is red. Answer: A"
# Extracted: "A"
```

Extraction handles:
- Direct letter responses: "A"
- Natural language: "The answer is B"
- Reasoning chains: "<think>...</think> Answer: C"

### 4. Scoring

- **Correct answer**: 1.0 point
- **Incorrect answer**: 0.0 points
- **"Cannot answer"**: (1/n_choices) + 0.05 points

### 5. Metrics Computation

Per-category and overall metrics:
- Total questions
- Correct answers
- Overall accuracy
- Average score
- "Cannot answer" count

## Output Structure

```
outputs/
└── qa_image/
    └── Qwen3-VL-4B-Instruct/
        ├── Qwen3-VL-4B-Instruct_qa_image_results.json
        └── Qwen3-VL-4B-Instruct_qa_image_metrics.json
```

### Results JSON Format

```json
{
  "nat_001": [
    {
      "question": "What color is the sky?",
      "choices": ["blue", "red", "green"],
      "ground_truth": "blue",
      "gt_letter": "A",
      "predicted_letter": "A",
      "model_answer": "blue",
      "model_output": "The sky in the image is blue. Answer: A",
      "is_correct": true,
      "is_cannot_answer": false,
      "score": 1.0,
      "category": "Color Recognition"
    }
  ]
}
```

### Metrics JSON Format

```json
{
  "model": "Qwen3-VL-4B-Instruct",
  "split": "natural",
  "dataset": "Borise/CaptionQA",
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

## Features

### 1. Multiple Image Support
Handles questions with multiple images:
```python
images = [image1, image2, image3]  # All sent to model together
```

### 2. Incremental Saving
Results saved every N questions (default: 10):
- Prevents data loss from interruptions
- Allows monitoring progress
- Supports auto-resume

### 3. Auto-Resume
Automatically skips already-processed entries:
```bash
# First run (processes 100 questions, then crashes)
python qa_image_qwen3vl_server.py ...

# Second run (resumes from question 101)
python qa_image_qwen3vl_server.py ...
```

### 4. Per-Category Metrics
Breaks down performance by question category:
- Color Recognition
- Object Detection
- Spatial Reasoning
- Text Reading
- etc.

### 5. "Cannot Answer" Handling
Allows model to abstain when image doesn't provide enough information:
- Partial credit scoring
- Tracks abstention rate
- Better than random guessing

## Example Workflow

### Step 1: Start vLLM Server
```bash
cd Tiger-VL-Rethinker/
sbatch start_vllm_server.sh

# Wait for server to start
tail -f slurm-*-vllm-server.out
# Look for: "Uvicorn running on http://0.0.0.0:8000"
```

### Step 2: Run Image-Based QA
```bash
cd CaptionQA/

# Option A: Direct Python
python qa_image_qwen3vl_server.py \
  --dataset "Borise/CaptionQA" \
  --split "natural" \
  --server-url "http://localhost:8000" \
  --output-dir "./outputs"

# Option B: SLURM
sbatch run_qa_image_server.sh
```

### Step 3: Monitor Progress
```bash
# Watch SLURM output
tail -f slurm-*-qa-image-server.out

# Expected output:
# Evaluating: 45%|████████▌     | 450/1000 [02:15<02:45, 3.33q/s]
```

### Step 4: Check Results
```bash
# View results
cat outputs/qa_image/Qwen3-VL-4B-Instruct/Qwen3-VL-4B-Instruct_qa_image_metrics.json

# View detailed answers
cat outputs/qa_image/Qwen3-VL-4B-Instruct/Qwen3-VL-4B-Instruct_qa_image_results.json | head -50
```

## Comparison with Caption-Based QA

To compare image-based vs caption-based performance:

```bash
# 1. Generate captions first
python caption.py \
  --dataset "Borise/CaptionQA" \
  --split "natural" \
  --model "gpt-4o" \
  --prompt "SIMPLE"

# 2. Run caption-based QA
python qa.py \
  --caption-path outputs/captions/gpt-4o/gpt-4o_simple.json \
  --split "natural"

# 3. Run image-based QA
python qa_image_qwen3vl_server.py \
  --split "natural" \
  --server-url "http://localhost:8000"

# 4. Compare metrics
# Caption-based: outputs/captions/gpt-4o/cap_simple-gpt-4o__qa_Qwen2.5-72B-Instruct_metrics.json
# Image-based:   outputs/qa_image/Qwen3-VL-4B-Instruct/Qwen3-VL-4B-Instruct_qa_image_metrics.json
```

## Expected Performance

Based on the model capabilities:

| Dataset Split | Expected Accuracy |
|--------------|------------------|
| Natural | 75-85% |
| Document | 60-70% |
| E-commerce | 70-80% |
| EmbodiedAI | 65-75% |
| All | 70-80% |

**Note**: Performance depends on:
- Question difficulty
- Image quality
- Model size (4B vs larger models)
- Server configuration

## Troubleshooting

### Issue: Server Connection Error
```bash
# Check server is running
curl http://localhost:8000/v1/models

# Check server logs
tail -100 slurm-*-vllm-server.out
```

### Issue: Out of Memory
```bash
# Reduce max_tokens
python qa_image_qwen3vl_server.py ... --max-tokens 64

# Or restart server with more memory
#SBATCH --mem=96G
```

### Issue: Slow Inference
```bash
# Check if server is overloaded
# Wait for other jobs to finish

# Or reduce concurrent requests
# (This script runs sequentially, so no concurrent load)
```

### Issue: Image Loading Error
```bash
# Check dataset is accessible
python -c "from datasets import load_dataset; ds = load_dataset('Borise/CaptionQA', split='natural'); print(ds[0])"

# Check HuggingFace cache
ls ~/.cache/huggingface/datasets/
```

## Performance Tips

1. **Use Temperature 0.0**: For deterministic, reproducible results
2. **Save Frequently**: Use `--save-every 10` to save often
3. **Monitor Progress**: Watch SLURM output for speed/errors
4. **Resume Capability**: Don't use `--overwrite` unless necessary

## Differences from Tiger-VL-Rethinker QA

| Aspect | CaptionQA/qa_image_qwen3vl_server.py | Tiger-VL-Rethinker/QA/qa_qwen3vl_server.py |
|--------|-------------------------------------|-------------------------------------------|
| **Dataset** | CaptionQA (HuggingFace) | Custom parquet files |
| **Image Format** | PIL Image objects from dataset | File paths joined with dataset path |
| **Dataset Structure** | Entry → Questions list | Flat question list with image references |
| **Cannot Answer** | Added for non-yes/no questions | Not supported |
| **Scoring** | Score-based with partial credit | Binary correct/incorrect |

## Advanced Usage

### Evaluate Multiple Splits

```bash
for split in natural document ecommerce embodiedai; do
  python qa_image_qwen3vl_server.py \
    --split "$split" \
    --server-url "http://localhost:8000" \
    --output-dir "./outputs"
done
```

### Compare Different Models

```bash
# Server 1: Qwen3-VL-4B
python qa_image_qwen3vl_server.py \
  --server-url "http://localhost:8000" \
  --model-name "Qwen3-VL-4B-Instruct"

# Server 2: Larger model
python qa_image_qwen3vl_server.py \
  --server-url "http://localhost:8001" \
  --model-name "Qwen3-VL-30B-Instruct"
```

### Custom Temperature for Exploration

```bash
# Deterministic (default)
python qa_image_qwen3vl_server.py --temperature 0.0

# Slightly random (may help with edge cases)
python qa_image_qwen3vl_server.py --temperature 0.3

# Very random (not recommended for evaluation)
python qa_image_qwen3vl_server.py --temperature 0.7
```

## Summary

This image-based QA approach provides:
- ✅ **Direct visual understanding** (no caption bottleneck)
- ✅ **End-to-end evaluation** (single model sees and answers)
- ✅ **Efficient inference** (vLLM server for fast processing)
- ✅ **Robust metrics** (per-category + overall performance)
- ✅ **Production-ready** (incremental saving, auto-resume)

For best results, compare both caption-based and image-based QA to understand:
1. How much information is lost in captioning
2. Which types of questions benefit from direct vision
3. Trade-offs between model complexity and accuracy
