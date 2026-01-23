# vLLM Server Guide for Image Captioning and QA

This guide explains how to use vLLM server-based inference for both image captioning and visual question answering with the Qwen3-VL-4B-Instruct model.

## Table of Contents
- [Overview](#overview)
- [Architecture](#architecture)
- [Server vs Local Inference](#server-vs-local-inference)
- [Setup](#setup)
- [Usage](#usage)
  - [1. Start vLLM Server](#1-start-vllm-server)
  - [2. Run Captioning](#2-run-captioning)
  - [3. Run QA Evaluation](#3-run-qa-evaluation)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)

## Overview

The vLLM server approach separates model hosting from inference execution:
- **Server**: Loads the model once and exposes an OpenAI-compatible API
- **Client**: Makes HTTP requests to the server for inference

This architecture provides significant benefits for large-scale processing:
- ✅ One-time model loading (no repeated loading overhead)
- ✅ Concurrent request handling
- ✅ Resource separation (GPU server + CPU clients)
- ✅ Better scalability for batch jobs

## Architecture

```
┌─────────────────────┐
│   GPU Node          │
│                     │
│  ┌──────────────┐   │
│  │ vLLM Server  │   │
│  │ (Port 8000)  │   │
│  │              │   │
│  │ Qwen3-VL-4B  │   │
│  └──────────────┘   │
└──────────┬──────────┘
           │
           │ HTTP API
           │ (OpenAI Compatible)
           │
     ┌─────┴─────┐
     │           │
┌────▼────┐ ┌───▼─────┐
│ Caption │ │   QA    │
│ Client  │ │ Client  │
│ (CPU)   │ │ (CPU)   │
└─────────┘ └─────────┘
```

## Server vs Local Inference

| Aspect | Local Inference | Server Inference |
|--------|----------------|------------------|
| **Model Loading** | Every job loads model | Load once, reuse forever |
| **Memory** | Each job uses GPU memory | Shared across jobs |
| **Startup Time** | ~30-60s per job | Instant (after server start) |
| **Concurrency** | One job at a time | Multiple concurrent clients |
| **Resource Usage** | High per-job overhead | Efficient shared resources |
| **Best For** | Single tasks, development | Production, batch processing |

## Setup

### Prerequisites

1. **Install vLLM** (if not already installed):
   ```bash
   conda activate captionqa
   pip install vllm
   ```

2. **Verify model access**:
   ```bash
   # Check if model is cached locally
   ls ~/.cache/huggingface/hub/ | grep Qwen3-VL
   ```

### File Structure

```
Tiger-VL-Rethinker/
├── start_vllm_server.sh          # Start vLLM server
├── captioning/
│   ├── caption_qwen3vl_server.py # Server-based captioning script
│   └── caption_server_simple.sh  # Example SLURM script
└── QA/
    └── qa_qwen3vl_server.py      # Server-based QA script
```

## Usage

### 1. Start vLLM Server

The server must be running before you can use the client scripts.

#### Start the server:
```bash
cd CAPTIONING/Tiger-VL-Rethinker
sbatch start_vllm_server.sh
```

#### Check server status:
```bash
# View server logs
tail -f slurm-*-vllm-server.out

# Expected output when ready:
# INFO:     Started server process [12345]
# INFO:     Waiting for application startup.
# INFO:     Application startup complete.
# INFO:     Uvicorn running on http://0.0.0.0:8000
```

#### Test the server:
```bash
# Check available models
curl http://localhost:8000/v1/models

# Expected output:
# {
#   "object": "list",
#   "data": [
#     {
#       "id": "Qwen/Qwen3-VL-4B-Instruct",
#       "object": "model",
#       ...
#     }
#   ]
# }
```

#### Get server job ID:
```bash
squeue -u $USER | grep vllm_qwen3vl
```

**Important**: Keep the server running throughout your captioning/QA jobs. You can run multiple client jobs against the same server.

---

### 2. Run Captioning

#### Quick Start (SIMPLE prompt):
```bash
sbatch captioning/caption_server_simple.sh
```

#### Custom Configuration:
```bash
python captioning/caption_qwen3vl_server.py \
  --dataset-path "/path/to/images" \
  --output-dir "./output" \
  --server-url "http://localhost:8000" \
  --prompt-style "SIMPLE" \
  --max-tokens 256
```

#### Available Prompt Styles:
- **SIMPLE**: "Describe this image in detail."
- **SHORT**: "Write a very short caption for the given image."
- **LONG**: "Write a very long and detailed caption describing the given image as comprehensively as possible."

#### Arguments:
- `--dataset-path`: Directory containing images to caption (required)
- `--server-url`: vLLM server URL (required, e.g., http://localhost:8000)
- `--output-dir`: Where to save results (default: ./output)
- `--prompt-style`: SIMPLE, SHORT, or LONG (default: SIMPLE)
- `--model-name`: Model name for output folders (default: Qwen3-VL-4B-Instruct)
- `--max-tokens`: Maximum tokens to generate (default: 256)
- `--overwrite`: Overwrite existing captions

#### Output Structure:
```
output/
└── captions/
    └── Qwen3-VL-4B-Instruct/
        └── Qwen3-VL-4B-Instruct_simple.json
```

#### Output Format:
```json
{
  "image1.jpg": "A detailed caption describing the first image...",
  "image2.jpg": "A detailed caption describing the second image...",
  ...
}
```

#### Example SLURM Script:
```bash
#!/bin/bash
#SBATCH --job-name=caption_server
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=2-00:00:00

# Activate environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate captionqa

# Run captioning
python caption_qwen3vl_server.py \
  --dataset-path "/mnt/data-alpha-sg-02/team-agent/ai_glasses/datasets/ViRL39K/images" \
  --output-dir "./output" \
  --server-url "http://localhost:8000" \
  --prompt-style "SIMPLE" \
  --max-tokens 256
```

---

### 3. Run QA Evaluation

#### Basic Usage:
```bash
python QA/qa_qwen3vl_server.py \
  --question-path "/path/to/questions.parquet" \
  --dataset-path "/path/to/dataset" \
  --server-url "http://localhost:8000"
```

#### Full Example:
```bash
python QA/qa_qwen3vl_server.py \
  --question-path "/mnt/data/questions.parquet" \
  --dataset-path "/mnt/data/ViRL39K" \
  --server-url "http://localhost:8000" \
  --output-dir "./output" \
  --model-name "Qwen3-VL-4B-Instruct" \
  --max-tokens 128
```

#### Arguments:
- `--question-path`: Path to parquet file with questions (required)
- `--dataset-path`: Base path for dataset images (required)
- `--server-url`: vLLM server URL (required)
- `--output-dir`: Directory to save results (default: ./output)
- `--model-name`: Model name for output folders (default: Qwen3-VL-4B-Instruct)
- `--max-tokens`: Maximum tokens to generate (default: 128)
- `--overwrite`: Overwrite existing results

#### Question Parquet Format:
The parquet file should contain:
- `qid`: Question ID
- `question`: Question text with choices
- `answer`: Ground truth answer (e.g., `\boxed{A}`)
- `image`: Image path(s) - can be string, list, or numpy array
- `category`: Question category (optional)
- `source`: Question source (optional)

#### Multiple Image Support:
The QA script automatically handles questions with multiple images:
```python
# Single image
row['image'] = array(['images/img1.png'], dtype=object)

# Multiple images
row['image'] = array(['images/img1.png', 'images/img2.png'], dtype=object)
```

#### Output Structure:
```
output/
└── qa/
    └── Qwen3-VL-4B-Instruct/
        ├── Qwen3-VL-4B-Instruct_qa_results.json
        └── Qwen3-VL-4B-Instruct_qa_metrics.json
```

#### Results JSON Format:
```json
{
  "qid_001": {
    "qid": "qid_001",
    "question": "What color is the sky?",
    "choices": ["blue", "red", "green"],
    "gt_answer": "\\boxed{A}",
    "gt_letter": "A",
    "predicted_letter": "A",
    "model_output": "The answer is A.",
    "is_correct": true,
    "category": "Color Recognition",
    "source": "Custom",
    "image_paths": ["images/sky.jpg"],
    "num_images": 1
  }
}
```

#### Metrics JSON Format:
```json
{
  "model": "Qwen3-VL-4B-Instruct",
  "total_questions": 1000,
  "correct_answers": 850,
  "overall_accuracy": 0.8500,
  "category_metrics": {
    "Color Recognition": {
      "total": 200,
      "correct": 180,
      "accuracy": 0.9000
    },
    "Object Detection": {
      "total": 300,
      "correct": 250,
      "accuracy": 0.8333
    }
  },
  "question_path": "/path/to/questions.parquet",
  "dataset_path": "/path/to/dataset",
  "server_url": "http://localhost:8000"
}
```

#### Example SLURM Script:
```bash
#!/bin/bash
#SBATCH --job-name=qa_server
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=2-00:00:00

# Activate environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate captionqa

# Run QA evaluation
python QA/qa_qwen3vl_server.py \
  --question-path "/mnt/data/questions.parquet" \
  --dataset-path "/mnt/data/ViRL39K" \
  --server-url "http://localhost:8000" \
  --output-dir "./output" \
  --max-tokens 128
```

---

## Configuration

### Server Configuration

Edit `start_vllm_server.sh` to customize:

```bash
# Model path
MODEL_PATH="Qwen/Qwen3-VL-4B-Instruct"

# Network settings
HOST="0.0.0.0"  # Listen on all interfaces
PORT=8000       # Server port

# GPU settings
TENSOR_PARALLEL_SIZE=1  # Number of GPUs (increase for multi-GPU)

# Model settings
--max-model-len 8192    # Maximum sequence length
--dtype auto            # Automatic dtype selection
```

### For Multi-GPU:
```bash
#SBATCH --gres=gpu:2
TENSOR_PARALLEL_SIZE=2
```

### For Different Port:
```bash
PORT=8001

# Then use in clients:
--server-url "http://localhost:8001"
```

### Client Configuration

Both client scripts support:
- **Incremental saving**: Results saved after each successful inference
- **Resume capability**: Automatically skip already-processed items
- **Error handling**: Continue on failures, log errors

---

## Troubleshooting

### Server Issues

#### Problem: Server won't start
```bash
# Check if port is already in use
lsof -i :8000

# Kill existing process if needed
kill <PID>
```

#### Problem: Out of memory
```bash
# Reduce max model length in start_vllm_server.sh
--max-model-len 4096  # Instead of 8192

# Or request more memory
#SBATCH --mem=96G
```

#### Problem: Server crashes during inference
```bash
# Check server logs
tail -100 slurm-*-vllm-server.err

# Common causes:
# - Input too long (reduce --max-tokens)
# - Image too large (resize images)
# - Memory pressure (reduce concurrent requests)
```

### Client Issues

#### Problem: Connection refused
```bash
# Verify server is running
squeue -u $USER | grep vllm

# Test server endpoint
curl http://localhost:8000/v1/models

# Check server is on the same node (for localhost)
# Or use the node's hostname: http://node123:8000
```

#### Problem: Slow inference
```bash
# Possible causes:
# 1. Server overloaded - wait for other jobs to finish
# 2. Large images - resize before encoding
# 3. Long prompts - reduce --max-tokens

# Monitor server CPU/GPU usage
nvidia-smi -l 1  # On GPU node
```

#### Problem: Image not found errors
```bash
# Verify dataset path
ls /path/to/dataset/images/

# Check image paths in parquet
python -c "
import pandas as pd
df = pd.read_parquet('questions.parquet')
print(df['image'].head())
"

# Ensure dataset-path joins correctly with image paths
# e.g., dataset-path=/data/ViRL39K + image=images/img.jpg
# -> /data/ViRL39K/images/img.jpg
```

#### Problem: Empty or corrupted output
```bash
# Check SLURM error logs
cat slurm-*-caption-server.err

# Verify server is responding
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen/Qwen3-VL-4B-Instruct", "messages": [{"role": "user", "content": "test"}]}'
```

### Performance Optimization

#### For Large Datasets:
1. **Batch Processing**: Run multiple client jobs in parallel
2. **Resume Support**: Use incremental saving to resume interrupted jobs
3. **Resource Allocation**: Balance CPU clients vs GPU server capacity

#### Optimal Setup:
```bash
# 1 GPU node running vLLM server
#SBATCH --gres=gpu:1
#SBATCH --mem=64G

# 3-5 CPU nodes running clients
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
```

---

## Advanced Usage

### Running Multiple Clients

You can run multiple captioning/QA jobs simultaneously:

```bash
# Start server once
sbatch start_vllm_server.sh

# Run multiple captioning jobs
sbatch captioning/caption_server_simple.sh
sbatch captioning/caption_server_short.sh
sbatch captioning/caption_server_long.sh

# Run QA jobs
sbatch qa_job1.sh
sbatch qa_job2.sh
```

### Using Remote Server

If the server is on a different node:

```bash
# Find server node
squeue -u $USER | grep vllm_qwen3vl
# Example output: node042

# Use node hostname in clients
--server-url "http://node042:8000"
```

### Monitoring

Monitor server and client jobs:

```bash
# Check all jobs
squeue -u $USER

# Watch server logs
watch -n 5 tail -20 slurm-*-vllm-server.out

# Check client progress
tail -f slurm-*-caption-server.out
```

---

## Comparison with Local Inference

### When to use Server Mode:
- ✅ Processing large datasets (>1000 images)
- ✅ Running multiple experiments/configurations
- ✅ Need fast turnaround (skip model loading)
- ✅ Running concurrent jobs
- ✅ Limited GPU resources (share one GPU)

### When to use Local Mode:
- ✅ Single small task (<100 images)
- ✅ Development/debugging
- ✅ Dedicated GPU per job
- ✅ No network setup needed
- ✅ Simpler setup

---

## Summary

### Workflow Checklist:

- [ ] Install vLLM: `pip install vllm`
- [ ] Start server: `sbatch start_vllm_server.sh`
- [ ] Verify server: `curl http://localhost:8000/v1/models`
- [ ] Run captioning: `sbatch captioning/caption_server_simple.sh`
- [ ] Run QA: `python QA/qa_qwen3vl_server.py --question-path ... --dataset-path ... --server-url ...`
- [ ] Check results in `output/` directory
- [ ] Shut down server when done: `scancel <job_id>`

### Quick Reference:

| Task | Command |
|------|---------|
| Start server | `sbatch start_vllm_server.sh` |
| Check server | `curl http://localhost:8000/v1/models` |
| Caption images | `sbatch captioning/caption_server_simple.sh` |
| Run QA | `python QA/qa_qwen3vl_server.py --question-path <path> --dataset-path <path> --server-url http://localhost:8000` |
| View server logs | `tail -f slurm-*-vllm-server.out` |
| Stop server | `scancel <job_id>` |

---

## Support

For issues or questions:
1. Check server logs: `slurm-*-vllm-server.{out,err}`
2. Check client logs: `slurm-*-caption-server.{out,err}`
3. Verify API endpoint: `curl http://localhost:8000/v1/models`
4. Review this guide's [Troubleshooting](#troubleshooting) section
