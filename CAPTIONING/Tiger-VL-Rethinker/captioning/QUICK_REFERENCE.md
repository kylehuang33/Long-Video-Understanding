# Quick Reference Guide

## All Scripts Overview

### Direct QA (Image → Answer)
```bash
cd qa_direct
./run_qa_direct.sh
```
- **Output**: `results_direct/Qwen3-VL-4B-Instruct/Qwen3-VL-4B-Instruct_results.json`
- **Use when**: You need best accuracy

### Caption-Based QA - Separate Steps (Recommended)
```bash
cd qa_caption

# Step 1: Caption images
./run_caption.sh
# Output: results_caption/captions/Qwen3-VL-4B-Instruct/Qwen3-VL-4B-Instruct_simple.json

# Step 2: QA with captions
./run_qa_with_captions.sh
# Output: results_caption/qa_results/Qwen3-VL-4B-Instruct/Qwen3-VL-4B-Instruct_with_Qwen3-VL-4B-Instruct_simple.json
```
- **Use when**: You want to reuse captions or run multiple experiments

### Caption-Based QA - All-in-One
```bash
cd qa_caption
./run_qa_caption.sh
```
- **Output**: Both caption and QA files in separated directories
- **Use when**: Quick testing, first-time runs

## File Structure

```
captioning/
├── qa_direct/
│   ├── qa_direct_vllm.py
│   ├── run_qa_direct.sh
│   └── results_direct/
│       └── {model}/
│           └── {model}_results.json
│
└── qa_caption/
    ├── caption_images_vllm.py          # Caption only
    ├── qa_with_captions_vllm.py        # QA with existing captions
    ├── qa_caption_vllm.py              # All-in-one
    ├── run_caption.sh
    ├── run_qa_with_captions.sh
    ├── run_qa_caption.sh
    └── results_caption/
        ├── captions/                   # Caption outputs
        │   └── {model}/
        │       └── {model}_{style}.json
        └── qa_results/                 # QA outputs
            └── {model}/
                └── {model}_with_{caption_model}_{style}.json
```

## Output File Naming

| Type | Location | Filename Pattern | Example |
|------|----------|------------------|---------|
| **Direct QA** | `results_direct/{model}/` | `{model}_results.json` | `Qwen3-VL-4B-Instruct_results.json` |
| **Captions** | `results_caption/captions/{model}/` | `{model}_{style}.json` | `Qwen3-VL-4B-Instruct_simple.json` |
| **QA with Captions** | `results_caption/qa_results/{model}/` | `{model}_with_{caption_model}_{style}.json` | `Qwen3-VL-4B-Instruct_with_Qwen3-VL-4B-Instruct_simple.json` |

## Key Features

### ✅ Separated Outputs
- Captions: `results_caption/captions/`
- QA results: `results_caption/qa_results/`
- Easy to evaluate each independently

### ✅ Caption Reusability
- Caption once, use for multiple QA experiments
- Mix and match different models

### ✅ Flexible Prompt Styles
- SIMPLE: "Describe this image in detail."
- SHORT: "Write a very short caption for the given image."
- LONG: "Write a very long and detailed caption..."

### ✅ Model Name Sanitization
- "Qwen/Qwen3-VL-4B-Instruct" → "Qwen3-VL-4B-Instruct"
- Safe for file system

## Common Workflows

### Workflow 1: Best Accuracy
```bash
cd qa_direct
./run_qa_direct.sh
```

### Workflow 2: Multiple Experiments
```bash
cd qa_caption

# Caption once
./run_caption.sh

# QA multiple times with different models
# Edit run_qa_with_captions.sh to change QA_MODEL
./run_qa_with_captions.sh  # Model A
./run_qa_with_captions.sh  # Model B
./run_qa_with_captions.sh  # Model C
```

### Workflow 3: Compare Caption Styles
```bash
cd qa_caption

# Caption with different styles
# Edit run_caption.sh: PROMPT_STYLE="SIMPLE"
./run_caption.sh

# Edit run_caption.sh: PROMPT_STYLE="LONG"
./run_caption.sh

# QA with each caption style
# Edit run_qa_with_captions.sh to point to different caption files
./run_qa_with_captions.sh  # Using SIMPLE captions
./run_qa_with_captions.sh  # Using LONG captions
```

## Prerequisites

1. **Install dependencies**:
```bash
pip install pandas pyarrow requests tqdm
```

2. **Start vLLM server**:
```bash
vllm serve Qwen/Qwen3-VL-4B-Instruct --port 8000
```

3. **Run scripts**:
```bash
cd qa_direct  # or qa_caption
./run_*.sh
```

**Note**: Scripts use `requests.post()` to call vLLM server API directly (same pattern as `@CAPTIONING/CaptionQA/server/qa_image.py`).

## Customization

### Change Model
Edit bash scripts:
```bash
MODEL="Qwen/Qwen3-VL-7B-Instruct"  # or any other model
```

### Change Prompt Style
Edit `run_caption.sh`:
```bash
PROMPT_STYLE="LONG"  # Options: SIMPLE, SHORT, LONG
```

### Change Output Directory
Edit bash scripts:
```bash
OUTPUT_DIR="./my_custom_results"
```

### Change vLLM Server URL
Edit bash scripts:
```bash
VLLM_URL="http://localhost:8001/v1"
```

## Troubleshooting

### Caption file not found
```bash
# Make sure you run captioning first
cd qa_caption
./run_caption.sh

# Then run QA
./run_qa_with_captions.sh
```

### vLLM server not running
```bash
# Start vLLM server in a separate terminal
vllm serve Qwen/Qwen3-VL-4B-Instruct --port 8000
```

### Permission denied
```bash
# Make scripts executable
chmod +x qa_direct/*.sh qa_caption/*.sh
```

## Documentation Files

- **README_QA_SCRIPTS.md**: Complete documentation with all details
- **MODIFICATIONS.md**: Summary of what was changed
- **QUICK_REFERENCE.md**: This file - quick reference guide
