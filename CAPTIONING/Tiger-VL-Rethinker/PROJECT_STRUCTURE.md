# Tiger-VL-Rethinker Project Structure

This document explains the organization of scripts in this project.

## Directory Structure

```
Tiger-VL-Rethinker/
├── captioning/                  # Local captioning scripts
│   ├── caption_qwen3vl.py      # Caption with local model
│   ├── caption_simple.sh        # SLURM script (simple)
│   ├── caption_short.sh         # SLURM script (short)
│   └── caption_long.sh          # SLURM script (long)
├── QA/                          # Local QA scripts
│   ├── qa_qwen3vl.py           # QA with local model
│   ├── run_qa.sh                # SLURM script
│   └── test_qa.py               # Test script
├── server/                      # vLLM server scripts
│   ├── start_vllm_server.sh            # Start server (legacy)
│   ├── start_vllm_server_simple.sh     # Start server (recommended)
│   ├── VLLM_SERVER_GUIDE.md            # Complete server documentation
│   ├── captioning/
│   │   ├── caption_qwen3vl_server.py   # Caption via server
│   │   └── caption_server_simple.sh    # SLURM script
│   └── QA/
│       └── qa_qwen3vl_server.py        # QA via server
├── output/                      # Output directory
├── README.md                    # Main documentation
└── RUN_INSTRUCTIONS.md          # Usage instructions
```

## When to Use What

### Local Inference (captioning/ and QA/)

Use when:
- ✅ Running single dedicated jobs
- ✅ You have exclusive GPU access
- ✅ No need for concurrent jobs

**Advantages:**
- Self-contained (no server management)
- Simple setup
- Full GPU utilization

**Commands:**
```bash
# Captioning
sbatch captioning/caption_simple.sh

# QA
sbatch QA/run_qa.sh
```

### Server-Based Inference (server/)

Use when:
- ✅ Running multiple experiments
- ✅ Multiple people sharing GPU
- ✅ Want to keep model loaded between jobs
- ✅ Concurrent processing needed

**Advantages:**
- One-time model loading
- Efficient resource sharing
- Can run multiple client jobs

**Commands:**
```bash
# 1. Start server (once)
sbatch server/start_vllm_server_simple.sh

# 2. Run jobs (multiple times)
sbatch server/captioning/caption_server_simple.sh
python server/QA/qa_qwen3vl_server.py --server-url http://localhost:8000
```

## Quick Start

### Local Inference

```bash
# Captioning
sbatch captioning/caption_simple.sh

# QA
python QA/qa_qwen3vl.py \
  --question-path /path/to/questions.parquet \
  --dataset-path /path/to/dataset
```

### Server-Based Inference

```bash
# Step 1: Start server
cd server
sbatch start_vllm_server_simple.sh

# Step 2: Run captioning
sbatch captioning/caption_server_simple.sh

# Step 3: Run QA
python QA/qa_qwen3vl_server.py \
  --question-path /path/to/questions.parquet \
  --dataset-path /path/to/dataset \
  --server-url http://localhost:8000
```

## File Naming Convention

| Pattern | Type | Example |
|---------|------|---------|
| `*_server.py` | Server-based script | `caption_qwen3vl_server.py` |
| `*.py` (no suffix) | Local inference | `caption_qwen3vl.py` |
| `*.sh` | SLURM batch script | `caption_simple.sh` |

## Documentation

- **README.md**: Main project documentation
- **RUN_INSTRUCTIONS.md**: Step-by-step usage guide
- **server/VLLM_SERVER_GUIDE.md**: Complete vLLM server documentation
- **server/README.md**: Server directory guide

## Choosing Between Local and Server

| Criterion | Use Local | Use Server |
|-----------|-----------|------------|
| Number of jobs | 1-2 | 3+ |
| Job frequency | One-time | Repeated |
| GPU sharing | No | Yes |
| Setup complexity | Simple | Moderate |
| Resource efficiency | Good | Excellent (when running multiple jobs) |

## Migration Guide

If you have old scripts:

### Old Structure
```
├── caption_qwen3vl.py           # Local
├── caption_qwen3vl_server.py    # Server
├── qa_qwen3vl.py                # Local
├── qa_qwen3vl_server.py         # Server
├── start_vllm_server.sh         # Server
```

### New Structure
```
├── captioning/
│   └── caption_qwen3vl.py           # Local
├── QA/
│   └── qa_qwen3vl.py                # Local
└── server/
    ├── start_vllm_server_simple.sh  # Server
    ├── captioning/
    │   └── caption_qwen3vl_server.py    # Server
    └── QA/
        └── qa_qwen3vl_server.py         # Server
```

## Summary

- **captioning/** and **QA/**: Local model inference (self-contained)
- **server/**: All vLLM server-related code (requires server setup)
- Choose based on your workflow and resource constraints
