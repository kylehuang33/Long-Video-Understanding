# CaptionQA Project Structure

This document explains the organization of scripts in this project.

## Directory Structure

```
CaptionQA/
├── caption.py                   # Generate captions (supports multiple backends)
├── qa.py                        # Caption-based QA (uses captions for evaluation)
├── qa_image_local.py            # Image-based QA with local inference ✅
├── run_qa_image_local.sh        # SLURM script for local inference
├── QA_IMAGE_LOCAL_USAGE.md     # Documentation for local inference
├── server/                      # vLLM server scripts
│   ├── start_vllm_server.sh            # Start vLLM server
│   ├── qa_image.py                     # Image QA via server (recommended)
│   ├── qa_image_qwen3vl_server.py      # Image QA via server (alternative)
│   ├── run_qa_image.sh                 # SLURM for qa_image.py
│   ├── run_qa_image_server.sh          # SLURM for qa_image_qwen3vl_server.py
│   ├── QA_IMAGE_USAGE.md              # Documentation for qa_image.py
│   └── IMAGE_QA_README.md             # Documentation for qa_image_qwen3vl_server.py
├── pipeline/                    # API wrappers for different backends
├── output/                      # Output directory
├── README.md                    # Main documentation
└── AZURE_OPENAI_IMPLEMENTATION_FLOW.md  # Azure OpenAI integration docs
```

## Evaluation Approaches

### 1. Caption-Based QA (Traditional)

Use `caption.py` + `qa.py` for two-step evaluation:

```bash
# Step 1: Generate captions
python caption.py \
  --dataset "Borise/CaptionQA" \
  --split natural \
  --model gpt-4o \
  --prompt SIMPLE

# Step 2: Answer questions using captions
python qa.py \
  --caption-path outputs/captions/gpt-4o/gpt-4o_simple.json \
  --split natural
```

**Advantages:**
- Tests text-only LLMs
- Compares caption quality
- Can reuse captions for multiple experiments

### 2. Image-Based QA with Local Inference ✅ (Recommended)

Use `qa_image_local.py` for direct image evaluation:

```bash
python qa_image_local.py \
  --split natural \
  --model Qwen/Qwen3-VL-8B-Instruct \
  --batch-size 4
```

**Advantages:**
- ✅ No server setup needed
- ✅ Batch processing (2x faster)
- ✅ Self-contained
- ✅ Full GPU utilization

### 3. Image-Based QA with Server (server/)

Use server-based scripts for multiple concurrent jobs:

```bash
# Step 1: Start server
cd server
sbatch start_vllm_server.sh

# Step 2: Run QA
python qa_image.py --vllm-server-url http://localhost:8000
```

**Advantages:**
- One-time model loading
- Multiple concurrent jobs
- Efficient resource sharing

## When to Use What

| Scenario | Use This | Why |
|----------|----------|-----|
| Single QA evaluation | `qa_image_local.py` | Fastest, self-contained |
| Multiple experiments | `server/qa_image.py` | Reuse loaded model |
| Testing text LLMs | `caption.py` + `qa.py` | Caption-based evaluation |
| Caption quality comparison | `caption.py` + `qa.py` | Tests captioning |
| Maximum speed | `qa_image_local.py` --batch-size 8 | Batch processing |
| Shared GPU | `server/` scripts | Efficient sharing |

## Quick Start

### Local Inference (Recommended)

```bash
# Single command!
python qa_image_local.py \
  --split natural \
  --model Qwen/Qwen3-VL-8B-Instruct \
  --batch-size 4

# Or with SLURM
sbatch run_qa_image_local.sh
```

### Server-Based Inference

```bash
# Step 1: Start server
cd server
sbatch start_vllm_server.sh

# Step 2: Run QA
python qa_image.py \
  --split natural \
  --vllm-server-url http://localhost:8000
```

### Caption-Based QA

```bash
# Step 1: Generate captions
python caption.py \
  --dataset "Borise/CaptionQA" \
  --split natural \
  --model gpt-4o

# Step 2: Run QA
python qa.py \
  --caption-path outputs/captions/gpt-4o/gpt-4o_simple.json \
  --split natural
```

## File Naming Convention

| Pattern | Type | Example |
|---------|------|---------|
| `qa_image_local.py` | Local inference | Direct model loading |
| `server/qa_image.py` | Server-based | Requires vLLM server |
| `qa.py` | Caption-based | Uses text captions |
| `*.sh` | SLURM script | Batch job |

## Performance Comparison

| Method | Setup | Speed | Batch | Best For |
|--------|-------|-------|-------|----------|
| `qa_image_local.py` | None | 6-8 q/s | Yes (4-8) | Single job ✅ |
| `server/qa_image.py` | Start server | 3 q/s | No | Multiple jobs |
| `qa.py` (caption-based) | Generate captions first | ~10 q/s (text-only) | Yes | Caption testing |

**Recommendation**: Use `qa_image_local.py` for most cases!

## Documentation

| File | Description |
|------|-------------|
| **README.md** | Main project documentation |
| **QA_IMAGE_LOCAL_USAGE.md** | Local inference guide ✅ |
| **server/QA_IMAGE_USAGE.md** | Server-based qa_image.py guide |
| **server/IMAGE_QA_README.md** | Server-based qa_image_qwen3vl_server.py guide |
| **AZURE_OPENAI_IMPLEMENTATION_FLOW.md** | Azure OpenAI integration |

## Workflow Examples

### Example 1: Quick Evaluation

```bash
# Fastest way - local inference with batching
python qa_image_local.py \
  --split natural \
  --batch-size 8
```

### Example 2: Multiple Experiments

```bash
# Start server once
cd server
sbatch start_vllm_server.sh

# Run multiple experiments (all use same server)
python qa_image.py --split natural --vllm-server-url http://localhost:8000
python qa_image.py --split document --vllm-server-url http://localhost:8000
python qa_image.py --split ecommerce --vllm-server-url http://localhost:8000
```

### Example 3: Compare Caption Quality

```bash
# Generate captions with different models
python caption.py --model gpt-4o --prompt SIMPLE
python caption.py --model gpt-4o --prompt LONG
python caption.py --model claude-3-5-sonnet --prompt SIMPLE

# Evaluate each
python qa.py --caption-path outputs/captions/gpt-4o/gpt-4o_simple.json
python qa.py --caption-path outputs/captions/gpt-4o/gpt-4o_long.json
python qa.py --caption-path outputs/captions/claude-3-5-sonnet/claude-3-5-sonnet_simple.json
```

## Migration from Old Structure

### Old Files (Before Organization)
```
CaptionQA/
├── qa_image.py                      # Server-based
├── qa_image_qwen3vl_server.py       # Server-based
├── run_qa_image.sh                  # Server-based
├── run_qa_image_server.sh           # Server-based
├── start_vllm_server.sh             # Server
```

### New Structure (After Organization)
```
CaptionQA/
├── qa_image_local.py                # Local (recommended) ✅
├── run_qa_image_local.sh            # Local SLURM
└── server/                          # All server files
    ├── qa_image.py                  # Server-based
    ├── qa_image_qwen3vl_server.py   # Server-based
    ├── run_qa_image.sh              # Server SLURM
    ├── run_qa_image_server.sh       # Server SLURM
    └── start_vllm_server.sh         # Server startup
```

## Choosing the Right Script

```
Need to evaluate images?
│
├─ Single job, dedicated GPU?
│  └─> Use qa_image_local.py (fastest!) ✅
│
├─ Multiple concurrent jobs?
│  └─> Use server/qa_image.py
│
└─ Testing captions or text-only LLMs?
   └─> Use caption.py + qa.py
```

## Advanced Usage

### Batch Size Optimization

```bash
# Start small
python qa_image_local.py --batch-size 4

# Monitor GPU memory
nvidia-smi -l 1

# If GPU has headroom, increase
python qa_image_local.py --batch-size 8
```

### All Splits Evaluation

```bash
# Local inference
for split in natural document ecommerce embodiedai; do
  python qa_image_local.py --split $split --batch-size 4
done

# Or all at once
python qa_image_local.py --split all --batch-size 8
```

## Summary

- **qa_image_local.py**: Best for most use cases (fast, self-contained) ✅
- **server/**: Use when running multiple concurrent experiments
- **qa.py**: Use for caption-based evaluation only

**Default recommendation**: Start with `qa_image_local.py --batch-size 4`
