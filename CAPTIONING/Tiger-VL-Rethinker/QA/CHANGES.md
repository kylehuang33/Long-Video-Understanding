# QA Script Changes Summary

## Changes Made

### 1. Options Format ✓
The options are already formatted as **"A."** instead of "(A)":
```python
lines = [f"{LETTER_ALPH[i]}. {choice}" for i, choice in enumerate(choices)]
```

Example output:
```
Options:
A. yes
B. no
```

### 2. Output Folder Structure ✓
Now generates output paths based on model name, similar to captioning script:

**Before:**
- Required `--output-path` argument
- Output: `./qa_results.json`

**After:**
- Optional `--output-dir` argument (default: `./output/qa`)
- Output: `./output/qa/Qwen3-VL-4B-Instruct/Qwen3-VL-4B-Instruct_qa_results.json`
- Metrics: `./output/qa/Qwen3-VL-4B-Instruct/Qwen3-VL-4B-Instruct_qa_results_metrics.json`

### 3. Model Name Handling ✓
Added helper functions to sanitize model names:
- `_looks_like_path()` - Checks if string is a file path
- `_sanitize_name()` - Cleans up special characters
- `make_model_safe()` - Extracts clean model name

**Examples:**
- `"Qwen/Qwen3-VL-4B-Instruct"` → `"Qwen3-VL-4B-Instruct"`
- `/path/to/model` → `"model"` (base directory name)

## Usage

### Old Way (still works for backward compatibility)
```bash
python qa_qwen3vl.py \
  --question-path /path/to/questions.parquet \
  --dataset-path /path/to/dataset \
  --output-path ./results.json
```

### New Way (recommended)
```bash
python qa_qwen3vl.py \
  --question-path /path/to/questions.parquet \
  --dataset-path /path/to/dataset \
  --output-dir ./output/qa
```

The output will be automatically organized:
```
output/qa/
└── Qwen3-VL-4B-Instruct/
    ├── Qwen3-VL-4B-Instruct_qa_results.json
    └── Qwen3-VL-4B-Instruct_qa_results_metrics.json
```

### With Custom Model
```bash
python qa_qwen3vl.py \
  --question-path /path/to/questions.parquet \
  --dataset-path /path/to/dataset \
  --model-path "meta-llama/Llama-3.2-11B-Vision-Instruct" \
  --output-dir ./output/qa
```

Output:
```
output/qa/
└── Llama-3.2-11B-Vision-Instruct/
    ├── Llama-3.2-11B-Vision-Instruct_qa_results.json
    └── Llama-3.2-11B-Vision-Instruct_qa_results_metrics.json
```

## Updated Shell Script

The `run_qa.sh` script has been updated to use `--output-dir` instead of `--output-path`:

```bash
OUTPUT_DIR="./output/qa"

python qa_qwen3vl.py \
  --question-path "$QUESTION_PATH" \
  --dataset-path "$DATASET_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --model-path "$MODEL_PATH"
```

## File Structure

The script now creates a clean folder structure matching the captioning script:

```
Tiger-VL-Rethinker/
├── captioning/
│   └── output/
│       └── captions/
│           └── Qwen3-VL-4B-Instruct/
│               ├── Qwen3-VL-4B-Instruct_simple.json
│               ├── Qwen3-VL-4B-Instruct_short.json
│               └── Qwen3-VL-4B-Instruct_long.json
└── QA/
    └── output/
        └── qa/
            └── Qwen3-VL-4B-Instruct/
                ├── Qwen3-VL-4B-Instruct_qa_results.json
                └── Qwen3-VL-4B-Instruct_qa_results_metrics.json
```

## Benefits

1. **Consistent naming**: No more `Qwen_Qwen3-VL...` duplication
2. **Organized by model**: Easy to compare results from different models
3. **Automatic directories**: Folders are created automatically
4. **Clean names**: HuggingFace IDs are properly parsed

## Testing

Test the changes:
```bash
cd /path/to/QA

# Test with default settings
python qa_qwen3vl.py \
  --question-path /path/to/questions.parquet \
  --dataset-path /path/to/dataset

# Check output structure
ls -la output/qa/Qwen3-VL-4B-Instruct/
```

Expected output:
```
Qwen3-VL-4B-Instruct_qa_results.json
Qwen3-VL-4B-Instruct_qa_results_metrics.json
```
