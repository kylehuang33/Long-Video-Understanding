# vLLM Server Scripts (CaptionQA)

This folder contains all vLLM server-related scripts for the CaptionQA project.

## Server Setup

### Start vLLM Server

```bash
cd server
sbatch start_vllm_server.sh

# Or use simple command
vllm serve Qwen/Qwen3-VL-8B-Instruct --port 8000
```

### Verify Server

```bash
curl http://localhost:8000/v1/models
```

## Image-Based QA with Server

Two server-based implementations are available:

### Option 1: qa_image.py (Recommended)

Follows the same pattern as `caption.py` and `qa.py`.

```bash
cd server

# Direct command
python qa_image.py \
  --split natural \
  --vllm-server-url http://localhost:8000

# Or with SLURM
sbatch run_qa_image.sh
```

### Option 2: qa_image_qwen3vl_server.py

Alternative implementation with different interface.

```bash
cd server

# Direct command
python qa_image_qwen3vl_server.py \
  --dataset "Borise/CaptionQA" \
  --split natural \
  --server-url http://localhost:8000

# Or with SLURM
sbatch run_qa_image_server.sh
```

## Local Inference (No Server)

For local inference without vLLM server, use:

```bash
cd ..  # Go back to CaptionQA root

# Local inference with batch processing
python qa_image_local.py \
  --split natural \
  --model Qwen/Qwen3-VL-8B-Instruct \
  --batch-size 4
```

## Documentation

- **QA_IMAGE_USAGE.md**: Documentation for `qa_image.py`
- **IMAGE_QA_README.md**: Documentation for `qa_image_qwen3vl_server.py`
- **../QA_IMAGE_LOCAL_USAGE.md**: Documentation for local inference (in parent directory)

## File Structure

```
server/
├── README.md                       # This file
├── start_vllm_server.sh            # Start vLLM server
├── qa_image.py                     # Image QA via server (recommended)
├── qa_image_qwen3vl_server.py      # Image QA via server (alternative)
├── run_qa_image.sh                 # SLURM script for qa_image.py
├── run_qa_image_server.sh          # SLURM script for qa_image_qwen3vl_server.py
├── QA_IMAGE_USAGE.md              # Documentation for qa_image.py
└── IMAGE_QA_README.md             # Documentation for qa_image_qwen3vl_server.py
```

## Quick Reference

| Task | Command |
|------|---------|
| Start server | `sbatch start_vllm_server.sh` |
| QA (recommended) | `python qa_image.py --vllm-server-url http://localhost:8000` |
| QA (alternative) | `python qa_image_qwen3vl_server.py --server-url http://localhost:8000` |
| Check server | `curl http://localhost:8000/v1/models` |

## Server vs Local

| Aspect | Server (this folder) | Local (parent folder) |
|--------|---------------------|----------------------|
| **Setup** | Start server first | Direct execution |
| **Batch** | Sequential | Batched (4-8 at once) |
| **Speed** | ~3 q/s | ~6-8 q/s |
| **Best For** | Multiple concurrent jobs | Single dedicated job |

## Workflow

### For Server-Based:
```bash
# Step 1: Start server
cd server
sbatch start_vllm_server.sh

# Step 2: Run QA
sbatch run_qa_image.sh
```

### For Local-Based:
```bash
# Single command (no server)
cd ..  # CaptionQA root
sbatch run_qa_image_local.sh
```
