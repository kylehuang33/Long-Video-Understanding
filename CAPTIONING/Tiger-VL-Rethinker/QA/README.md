# Qwen3VL Question Answering Evaluation

This directory contains scripts for evaluating VQA (Visual Question Answering) questions using the Qwen3VL-4B-Instruct model.

## Overview

The QA evaluation script:
- Loads questions from a parquet file (ViRL39K format)
- Processes images with the Qwen3VL model
- Generates answers for multiple-choice questions
- Computes accuracy metrics (overall and per-category)
- Saves results and metrics to JSON files

## Files

- `qa_qwen3vl.py` - Main QA evaluation script
- `run_qa.sh` - SLURM batch script for running evaluation
- `README.md` - This file

## Installation

Make sure you have the required dependencies installed:

```bash
pip install torch transformers qwen-vl-utils tqdm pandas pyarrow
```

## Data Format

The script expects a parquet file with the following columns:

- `question` - Question text with `<image>` token and choices
- `answer` - Ground truth answer in `\boxed{X}` format
- `category` - Question category (e.g., "GradeSchool Geometric")
- `source` - Data source (e.g., "Processed")
- `qid` - Unique question ID
- `image` - Relative path(s) to image file(s)

Example question format:
```
<image>
Is the dotted line a line of symmetry?
Choices:
(A) yes
(B) no
```

## Usage

### Basic Usage

Evaluate questions from a parquet file:

```bash
python qa_qwen3vl.py \
  --question-path /path/to/questions.parquet \
  --dataset-path /path/to/dataset \
  --output-path ./results.json
```

### Command-Line Arguments

**Required:**
- `--question-path` - Path to parquet file with questions
- `--dataset-path` - Base path for dataset (images paths are joined with this)
- `--output-path` - Path to save output JSON file with results

**Optional:**
- `--model-path` - Path or HuggingFace model ID (default: Qwen/Qwen3-VL-4B-Instruct)
- `--max-new-tokens` - Maximum tokens to generate (default: 128)
- `--overwrite` - Overwrite existing results (default: False)

### Examples

#### Example 1: Evaluate with default model
```bash
python qa_qwen3vl.py \
  --question-path /mnt/data-alpha-sg-02/team-agent/ai_glasses/datasets/ViRL39K/39Krelease.parquet \
  --dataset-path /mnt/data-alpha-sg-02/team-agent/ai_glasses/datasets/ViRL39K \
  --output-path ./qa_results.json
```

#### Example 2: Evaluate with local model
```bash
python qa_qwen3vl.py \
  --question-path /path/to/questions.parquet \
  --dataset-path /path/to/dataset \
  --output-path ./qa_results.json \
  --model-path /mnt/data-alpha-sg-02/team-agent/ai_glasses/models/Qwen3-VL-4B-Instruct
```

#### Example 3: Overwrite existing results
```bash
python qa_qwen3vl.py \
  --question-path /path/to/questions.parquet \
  --dataset-path /path/to/dataset \
  --output-path ./qa_results.json \
  --overwrite
```

### Using SLURM

To run as a batch job:

1. Edit `run_qa.sh` to set your parameters:
   - `QUESTION_PATH` - Path to your questions parquet file
   - `DATASET_PATH` - Base path for images
   - `OUTPUT_PATH` - Where to save results

2. Submit the job:
```bash
sbatch run_qa.sh
```

3. Monitor the job:
```bash
# Check job status
squeue -u $USER

# View log output (replace JOBID with your job ID)
tail -f slurm-JOBID-qa.out
```

## Output Format

### Results JSON (`qa_results.json`)

The script generates a JSON file with predictions for each question:

```json
{
  "qid_001": {
    "qid": "qid_001",
    "question": "Is the dotted line a line of symmetry?",
    "choices": ["yes", "no"],
    "gt_answer": "\\boxed{A}",
    "gt_letter": "A",
    "predicted_letter": "A",
    "model_output": "The answer is A.",
    "is_correct": true,
    "category": "(GradeSchool) Geometric",
    "source": "Processed",
    "image_path": "images/example.jpg"
  },
  "qid_002": {
    ...
  }
}
```

### Metrics JSON (`qa_results_metrics.json`)

The script also generates a metrics file with evaluation statistics:

```json
{
  "total_questions": 39000,
  "correct_answers": 25000,
  "overall_accuracy": 0.6410,
  "model": "Qwen/Qwen3-VL-4B-Instruct",
  "question_path": "/path/to/questions.parquet",
  "dataset_path": "/path/to/dataset",
  "category_metrics": {
    "(GradeSchool) Geometric": {
      "total": 5000,
      "correct": 3200,
      "accuracy": 0.64
    },
    "(GradeSchool) Counting": {
      "total": 4500,
      "correct": 2900,
      "accuracy": 0.6444
    },
    ...
  }
}
```

## Features

- **Incremental Saving**: Results are saved every 10 questions to prevent data loss
- **Resume Support**: By default, the script skips questions that already have results (use `--overwrite` to regenerate)
- **Progress Bar**: Shows real-time progress with `tqdm`
- **Error Handling**: Continues processing even if individual questions fail
- **Category Metrics**: Computes accuracy breakdown by question category
- **Answer Extraction**: Robust parsing of model outputs to extract answer letters

## Answer Extraction Logic

The script extracts answers using multiple patterns:

1. **Letter pattern**: Looks for single letters (A, B, C, etc.)
2. **Number pattern**: Converts numbers (1, 2, 3) to letters (A, B, C)
3. **After markers**: Extracts answers after "Answer:" or "</think>" markers
4. **Boxed format**: Handles ground truth in `\boxed{X}` format

## Performance Notes

### Expected Processing Times

For ViRL39K dataset (~39,000 questions):

| Dataset | Avg Time/Question | Total Time (1 GPU) |
|---------|------------------|-------------------|
| ViRL39K | ~5-8 seconds | ~55-90 hours |

*Times are approximate and vary based on GPU model and image resolution*

### Optimization Tips

1. **Use faster GPU**: A100 is ~2x faster than V100
2. **Reduce max_tokens**: Lower values = faster generation
3. **Batch processing**: Split dataset and run multiple jobs in parallel
4. **Use local model**: Avoid download time on first run

## Troubleshooting

### Image Not Found Errors

If you see "Image not found" errors:
- Check that `DATASET_PATH` is correct
- Verify that image paths in the parquet file are relative to `DATASET_PATH`
- Ensure images exist: `ls $DATASET_PATH/images/example.jpg`

### Out of Memory

If you get OOM errors:
- Reduce `--max-new-tokens`
- Use a GPU with more memory
- Close other processes using GPU memory

### Low Accuracy

If accuracy is unexpectedly low:
- Check that answer extraction is working (look at `model_output` vs `predicted_letter`)
- Verify that ground truth format is correct (`\boxed{X}`)
- Examine failed cases in the results JSON

### Model Download Issues

If model download fails:
- Check internet connection
- Use a local model with `--model-path /path/to/local/model`
- Set HuggingFace cache: `export HF_HOME=/path/to/cache`

## Monitoring Evaluation

### Check Progress

```bash
# View real-time output
tail -f slurm-JOBID-qa.out

# Check how many questions processed
grep "Evaluating:" slurm-JOBID-qa.out | tail -1

# Check current accuracy
grep "correct" slurm-JOBID-qa.out | tail -1
```

### Analyze Results

```bash
# View results file
cat qa_results.json | jq '.' | head -50

# View metrics
cat qa_results_metrics.json | jq '.'

# Count correct answers
cat qa_results.json | jq '[.[] | select(.is_correct == true)] | length'

# Find failed questions
cat qa_results.json | jq '[.[] | select(.is_correct == false)] | length'

# Check category breakdown
cat qa_results_metrics.json | jq '.category_metrics'
```

## Example SLURM Output

```
======================================
Qwen3VL Question Answering Evaluation
======================================
Questions: /mnt/data-alpha-sg-02/team-agent/ai_glasses/datasets/ViRL39K/39Krelease.parquet
Dataset: /mnt/data-alpha-sg-02/team-agent/ai_glasses/datasets/ViRL39K
Output: ./qa_results.json
Model: Qwen/Qwen3-VL-4B-Instruct
======================================

Loading questions from /mnt/data-alpha-sg-02/team-agent/ai_glasses/datasets/ViRL39K/39Krelease.parquet...
Loaded 39000 questions
Columns: ['question', 'answer', 'PassRate_32BTrained', 'PassRate_7BBase', 'category', 'source', 'qid', 'image']

Loading model from Qwen/Qwen3-VL-4B-Instruct...
Model loaded successfully!

Processing questions...
Evaluating: 100%|██████████| 39000/39000 [2:15:30<00:00, 4.79it/s]

============================================================
Evaluation complete!
============================================================
Total questions: 39000
Correct answers: 25000 (64.10%)
Overall accuracy: 0.6410
Results saved to: ./qa_results.json
Metrics saved to: ./qa_results_metrics.json
Total time: 8130.45s (135.51m)
Average time per question: 4.79s
============================================================

Per-category accuracy:
============================================================
(GradeSchool) Counting: 2900/4500 (64.44%)
(GradeSchool) Geometric: 3200/5000 (64.00%)
(HighSchool) Algebra: 3100/4800 (64.58%)
...
============================================================
```

## Extending the Script

### Add Custom Answer Parsing

Edit the `extract_letter()` function in `qa_qwen3vl.py` to handle different output formats:

```python
def extract_letter(answer_text: str, num_options: int) -> Optional[str]:
    # Add your custom parsing logic here
    ...
```

### Change Prompt Format

Edit the `build_qa_prompt()` function to customize the question format:

```python
def build_qa_prompt(question: str, choices: List[str]) -> str:
    # Customize prompt format
    ...
```

### Add Additional Metrics

Edit the `compute_metrics()` function to compute additional statistics:

```python
def compute_metrics(results: Dict[str, Any]) -> Dict[str, Any]:
    # Add your custom metrics
    ...
```

## Citation

If you use Qwen3VL in your research, please cite:

```bibtex
@misc{qwen3vl,
  title={Qwen3-VL: Vision-Language Models},
  author={Qwen Team},
  year={2024},
  url={https://github.com/QwenLM/Qwen3-VL}
}
```

## Contact

For issues or questions, please check the main repository or open an issue.
