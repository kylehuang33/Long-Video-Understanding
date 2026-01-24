# vLLM Server Scripts

This folder contains all vLLM server-related scripts for the Tiger-VL-Rethinker project.

## Server Setup

### Start vLLM Server

```bash
# Simple command (recommended)
cd server
sbatch start_vllm_server_simple.sh

# Or legacy command
sbatch start_vllm_server.sh
```

### Verify Server

```bash
curl http://localhost:8000/v1/models
```

## Captioning with Server

```bash
cd server/captioning

# Run captioning
sbatch caption_server_simple.sh

# Or with custom settings
python caption_qwen3vl_server.py \
  --dataset-path /path/to/images \
  --server-url "http://localhost:8000" \
  --prompt-style SIMPLE
```

## QA with Server

```bash
cd server/QA

# Run QA evaluation
python qa_qwen3vl_server.py \
  --question-path /path/to/questions.parquet \
  --dataset-path /path/to/dataset \
  --server-url "http://localhost:8000"
```

## Local Inference (No Server)

For local inference without vLLM server, use the scripts in the parent directories:

- **Captioning**: `../captioning/caption_qwen3vl.py`
- **QA**: `../QA/qa_qwen3vl.py`

## Documentation

See `VLLM_SERVER_GUIDE.md` for complete documentation on:
- Server setup and configuration
- Usage examples
- Troubleshooting
- Performance optimization

## File Structure

```
server/
├── README.md                        # This file
├── VLLM_SERVER_GUIDE.md            # Complete documentation
├── start_vllm_server.sh            # Start server (legacy)
├── start_vllm_server_simple.sh     # Start server (recommended)
├── captioning/
│   ├── caption_qwen3vl_server.py   # Captioning via server
│   └── caption_server_simple.sh    # SLURM script for captioning
└── QA/
    └── qa_qwen3vl_server.py        # QA evaluation via server
```

## Quick Reference

| Task | Command |
|------|---------|
| Start server | `sbatch start_vllm_server_simple.sh` |
| Caption images | `python captioning/caption_qwen3vl_server.py --server-url http://localhost:8000` |
| Run QA | `python QA/qa_qwen3vl_server.py --server-url http://localhost:8000` |
| Check server | `curl http://localhost:8000/v1/models` |
