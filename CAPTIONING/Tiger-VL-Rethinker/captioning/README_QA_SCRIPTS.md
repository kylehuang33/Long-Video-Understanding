# QA Scripts for ViRL39K Dataset - Complete Version

This document describes all QA approaches for processing the ViRL39K dataset using Qwen3VL via vLLM server.

## Directory Structure

```
captioning/
├── qa_direct/                              # Direct QA approach (Image → Answer)
│   ├── qa_direct_vllm.py                  # Python script for direct QA
│   ├── run_qa_direct.sh                   # Bash script to run direct QA
│   └── results_direct/                    # Output directory (created at runtime)
│       └── {qa_model}/
│           └── {qa_model}_results.json
│
├── qa_caption/                             # Caption-based QA approaches
│   ├── caption_images_vllm.py             # Step 1: Caption images only
│   ├── qa_with_captions_vllm.py           # Step 2: QA using existing captions
│   ├── qa_caption_vllm.py                 # All-in-one: Caption + QA together
│   ├── run_caption.sh                     # Run captioning only
│   ├── run_qa_with_captions.sh            # Run QA with existing captions
│   ├── run_qa_caption.sh                  # Run caption + QA together
│   └── results_caption/                   # Output directory (created at runtime)
│       ├── captions/                      # Caption outputs
│       │   └── {caption_model}/
│       │       └── {caption_model}_{prompt_style}.json
│       └── qa_results/                    # QA outputs
│           └── {qa_model}/
│               └── {qa_model}_with_{caption_model}_{prompt_style}.json
│
└── caption_qwen3vl.py                     # Original captioning script (reference)
```

## File Naming Convention

Following the pattern from `caption_qwen3vl.py`, all output files are organized by model name and parameters:

### Direct QA Output
```
results_direct/
└── Qwen3-VL-4B-Instruct/
    └── Qwen3-VL-4B-Instruct_results.json
```

### Caption-Based QA Output (Separated)
```
results_caption/
├── captions/                                    # Caption outputs
│   └── Qwen3-VL-4B-Instruct/
│       ├── Qwen3-VL-4B-Instruct_simple.json
│       ├── Qwen3-VL-4B-Instruct_short.json
│       └── Qwen3-VL-4B-Instruct_long.json
│
└── qa_results/                                  # QA outputs
    └── Qwen3-VL-4B-Instruct/
        ├── Qwen3-VL-4B-Instruct_with_Qwen3-VL-4B-Instruct_simple.json
        ├── Qwen3-VL-4B-Instruct_with_Qwen3-VL-4B-Instruct_short.json
        └── Qwen3-VL-4B-Instruct_with_Qwen3-VL-4B-Instruct_long.json
```

The naming pattern is:
- **Captions**: `captions/{caption_model}/{caption_model}_{prompt_style}.json`
- **QA with captions**: `qa_results/{qa_model}/{qa_model}_with_{caption_model}_{prompt_style}.json`
- **Direct QA**: `{qa_model}/{qa_model}_results.json`

## Approach 1: Direct QA (`qa_direct/`)

### What It Does
Directly sends images with questions to the vLLM server and gets answers.

### Workflow
1. Reads parquet file with questions and image paths
2. For each question:
   - Loads the image(s)
   - Encodes image to base64
   - Sends image + question to vLLM server
   - Gets answer directly from the model
3. Saves results with ground truth for evaluation

### Files
- **`qa_direct_vllm.py`**: Main Python script
- **`run_qa_direct.sh`**: Bash script with configured paths

### Usage
```bash
cd qa_direct
./run_qa_direct.sh
```

Output: `results_direct/Qwen3-VL-4B-Instruct/Qwen3-VL-4B-Instruct_results.json`

## Approach 2: Caption-Based QA - Separate Steps (`qa_caption/`)

### What It Does
Separates captioning and QA into two independent steps for maximum flexibility.

### Workflow

#### Step 1: Caption Images (`caption_images_vllm.py`)
1. Reads parquet file
2. Collects all unique images
3. Captions each image using vLLM
4. Saves captions to `captions/{model}/{model}_{style}.json`

#### Step 2: QA with Captions (`qa_with_captions_vllm.py`)
1. Loads existing caption file
2. Reads parquet file with questions
3. Replaces `<image>` with captions
4. Sends text-only questions to vLLM
5. Saves results to `qa_results/{model}/{model}_with_{caption_model}_{style}.json`

### Files
- **`caption_images_vllm.py`**: Caption images only
- **`qa_with_captions_vllm.py`**: QA using existing captions
- **`run_caption.sh`**: Run captioning
- **`run_qa_with_captions.sh`**: Run QA with captions

### Usage

#### Step 1: Generate Captions
```bash
cd qa_caption
./run_caption.sh
```

Output: `results_caption/captions/Qwen3-VL-4B-Instruct/Qwen3-VL-4B-Instruct_simple.json`

#### Step 2: Run QA with Captions
```bash
cd qa_caption
# Edit run_qa_with_captions.sh to point to the caption file
./run_qa_with_captions.sh
```

Output: `results_caption/qa_results/Qwen3-VL-4B-Instruct/Qwen3-VL-4B-Instruct_with_Qwen3-VL-4B-Instruct_simple.json`

### Benefits of Separate Steps
1. **Reusability**: Caption once, use for multiple QA experiments
2. **Flexibility**: Use different QA models with same captions
3. **Efficiency**: Skip captioning if captions already exist
4. **Debugging**: Easier to debug each step independently

## Approach 3: Caption-Based QA - All-in-One (`qa_caption/`)

### What It Does
Combines captioning and QA into a single script for convenience.

### Files
- **`qa_caption_vllm.py`**: All-in-one caption + QA
- **`run_qa_caption.sh`**: Run both steps together

### Usage
```bash
cd qa_caption
./run_qa_caption.sh
```

Outputs:
- Captions: `results_caption/captions/Qwen3-VL-4B-Instruct/Qwen3-VL-4B-Instruct_simple.json`
- QA: `results_caption/qa_results/Qwen3-VL-4B-Instruct/Qwen3-VL-4B-Instruct_with_Qwen3-VL-4B-Instruct_simple.json`

### When to Use
- Quick experiments where you want both steps
- First time running on a dataset
- Don't need to reuse captions

## Prompt Styles

Three caption prompt styles are available (same as `caption_qwen3vl.py`):
- **SIMPLE**: "Describe this image in detail."
- **SHORT**: "Write a very short caption for the given image."
- **LONG**: "Write a very long and detailed caption describing the given image as comprehensively as possible."

## Output Formats

### Direct QA Output
```json
{
  "qid": {
    "question": "original question text",
    "ground_truth": "\\boxed{A}",
    "predicted_answer": "model's answer",
    "category": "(GradeSchool) Geometric",
    "source": "Processed",
    "image_paths": ["/full/path/to/image.jpg"]
  }
}
```

### Caption Output
```json
{
  "/full/path/to/image.jpg": "A geometric shape with a dotted line...",
  "/full/path/to/another.jpg": "Another image description..."
}
```

### QA with Captions Output
```json
{
  "qid": {
    "question_original": "<image>\nIs the dotted line...",
    "question_with_captions": "[Image: A geometric shape...]\nIs the dotted line...",
    "ground_truth": "\\boxed{A}",
    "predicted_answer": "model's answer",
    "category": "(GradeSchool) Geometric",
    "source": "Processed",
    "image_paths": ["/full/path/to/image.jpg"],
    "captions": ["A geometric shape with a dotted line..."],
    "caption_file": "path/to/caption/file.json"
  }
}
```

## Example Workflows

### Workflow 1: Direct QA
```bash
cd qa_direct
./run_qa_direct.sh
```

### Workflow 2: Caption Once, QA Multiple Times
```bash
cd qa_caption

# Step 1: Caption with SIMPLE style
# Edit run_caption.sh: PROMPT_STYLE="SIMPLE"
./run_caption.sh

# Step 2a: QA with model A
# Edit run_qa_with_captions.sh:
#   CAPTION_FILE="./results_caption/captions/Qwen3-VL-4B-Instruct/Qwen3-VL-4B-Instruct_simple.json"
#   QA_MODEL="Qwen/Qwen3-VL-4B-Instruct"
./run_qa_with_captions.sh

# Step 2b: QA with model B (reusing same captions)
# Edit run_qa_with_captions.sh:
#   CAPTION_FILE="./results_caption/captions/Qwen3-VL-4B-Instruct/Qwen3-VL-4B-Instruct_simple.json"
#   QA_MODEL="Qwen/Qwen3-VL-7B-Instruct"
./run_qa_with_captions.sh
```

### Workflow 3: Compare Different Caption Styles
```bash
cd qa_caption

# Caption with SIMPLE
# Edit run_caption.sh: PROMPT_STYLE="SIMPLE"
./run_caption.sh

# Caption with LONG
# Edit run_caption.sh: PROMPT_STYLE="LONG"
./run_caption.sh

# QA with SIMPLE captions
# Edit run_qa_with_captions.sh: CAPTION_FILE points to simple.json
./run_qa_with_captions.sh

# QA with LONG captions
# Edit run_qa_with_captions.sh: CAPTION_FILE points to long.json
./run_qa_with_captions.sh
```

### Workflow 4: All-in-One Quick Test
```bash
cd qa_caption
./run_qa_caption.sh
```

## Custom Parameters

### Caption Images Only
```bash
python caption_images_vllm.py \
    --parquet-path /path/to/data.parquet \
    --dataset-root /path/to/images \
    --output-dir ./my_results \
    --vllm-url http://localhost:8000/v1 \
    --model Qwen/Qwen3-VL-4B-Instruct \
    --prompt-style LONG \
    --max-tokens 512 \
    --overwrite
```

### QA with Existing Captions
```bash
python qa_with_captions_vllm.py \
    --parquet-path /path/to/data.parquet \
    --dataset-root /path/to/images \
    --caption-file ./captions/model/model_simple.json \
    --output-dir ./my_results \
    --vllm-url http://localhost:8000/v1 \
    --model Qwen/Qwen3-VL-4B-Instruct \
    --max-tokens 512 \
    --temperature 0.0 \
    --overwrite
```

### Direct QA
```bash
python qa_direct_vllm.py \
    --parquet-path /path/to/data.parquet \
    --dataset-root /path/to/images \
    --output-dir ./my_results \
    --vllm-url http://localhost:8000/v1 \
    --model Qwen/Qwen3-VL-4B-Instruct \
    --max-tokens 512 \
    --temperature 0.0 \
    --overwrite
```

## Comparison of Approaches

| Feature | Direct QA | Separate Steps | All-in-One |
|---------|-----------|----------------|------------|
| **Speed** | Slower (processes images each time) | Fast (reuse captions) | Medium |
| **Accuracy** | Highest (sees actual images) | Lower (caption quality dependent) | Lower |
| **Flexibility** | Low | High (reuse captions) | Medium |
| **Use Case** | Best accuracy needed | Multiple experiments | Quick testing |
| **Caption Reuse** | N/A | ✅ Yes | ❌ No |
| **Debugging** | Simple | Easy (separate steps) | Medium |

## Dependencies

All scripts require:
```bash
pip install pandas pyarrow requests tqdm
```

The vLLM server should be running separately:
```bash
pip install vllm
vllm serve Qwen/Qwen3-VL-4B-Instruct --port 8000
```

**Note**: The scripts use `requests` library to call vLLM server API directly (not the OpenAI client library).

## Notes

1. **Separate Outputs**: Captions and QA results are now in separate directories (`captions/` and `qa_results/`)

2. **Caption Reusability**: With separate scripts, you can caption once and run QA multiple times with different models or parameters

3. **Incremental Saving**: All scripts save results after each successful processing

4. **Error Handling**: Missing images or processing errors are logged but don't stop the entire process

5. **Model Name Sanitization**: Model names like "Qwen/Qwen3-VL-4B-Instruct" are automatically converted to safe directory names

6. **Caption File Reference**: QA results include the caption file path used, making it easy to trace which captions were used

## Quick Start Guide

### For Direct QA (Best Accuracy)
```bash
cd qa_direct
./run_qa_direct.sh
```

### For Caption-Based QA (Best Flexibility)
```bash
cd qa_caption

# Step 1: Caption images
./run_caption.sh

# Step 2: Run QA
./run_qa_with_captions.sh
```

### For Quick Testing
```bash
cd qa_caption
./run_qa_caption.sh
```
