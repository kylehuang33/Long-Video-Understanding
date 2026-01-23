# Simple vLLM Server Guide

This guide shows the simplest way to use vLLM server for image captioning and QA using the `vllm serve` command.

## Quick Start

### 1. Start vLLM Server (One Command!)

```bash
# Basic usage
vllm serve Qwen/Qwen3-VL-8B-Instruct

# With custom port
vllm serve Qwen/Qwen3-VL-8B-Instruct --port 8000

# With multiple GPUs
vllm serve Qwen/Qwen3-VL-8B-Instruct --tensor-parallel-size 2

# Full configuration
vllm serve Qwen/Qwen3-VL-8B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 1 \
  --trust-remote-code \
  --dtype auto \
  --max-model-len 8192
```

### 2. Use SLURM (Recommended)

```bash
# Tiger-VL-Rethinker
cd CAPTIONING/Tiger-VL-Rethinker
sbatch start_vllm_server_simple.sh

# Or CaptionQA
cd CAPTIONING/CaptionQA
sbatch start_vllm_server.sh
```

### 3. Verify Server is Running

```bash
# Check server status
curl http://localhost:8000/v1/models

# Expected output:
{
  "object": "list",
  "data": [
    {
      "id": "Qwen/Qwen3-VL-8B-Instruct",
      "object": "model",
      "created": 1234567890,
      "owned_by": "vllm"
    }
  ]
}
```

### 4. Use the API

Now all your existing scripts work automatically! The server exposes an OpenAI-compatible API.

## Usage with Existing Scripts

### Tiger-VL-Rethinker

```bash
# 1. Start server
cd CAPTIONING/Tiger-VL-Rethinker
sbatch start_vllm_server_simple.sh

# 2. Run captioning (uses server API)
sbatch captioning/caption_server_simple.sh

# 3. Run QA (uses server API)
python QA/qa_qwen3vl_server.py \
  --question-path /path/to/questions.parquet \
  --dataset-path /path/to/dataset \
  --server-url "http://localhost:8000"
```

### CaptionQA

```bash
# 1. Start server
cd CAPTIONING/CaptionQA
sbatch start_vllm_server.sh

# 2. Run image-based QA (uses server API)
python qa_image_qwen3vl_server.py \
  --dataset "Borise/CaptionQA" \
  --split "natural" \
  --server-url "http://localhost:8000"
```

## Command Comparison

### Old Way (Complex)
```bash
python -m vllm.entrypoints.openai.api_server \
  --model "Qwen/Qwen3-VL-8B-Instruct" \
  --host "0.0.0.0" \
  --port "8000" \
  --tensor-parallel-size "1" \
  --trust-remote-code \
  --dtype auto \
  --max-model-len 8192
```

### New Way (Simple) ✅
```bash
vllm serve Qwen/Qwen3-VL-8B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 1 \
  --trust-remote-code \
  --dtype auto \
  --max-model-len 8192
```

Both are equivalent! The `vllm serve` command is just a cleaner interface.

## Common Options

| Option | Description | Default |
|--------|-------------|---------|
| `--host` | Host to bind to | 127.0.0.1 |
| `--port` | Port to listen on | 8000 |
| `--tensor-parallel-size` | Number of GPUs | 1 |
| `--trust-remote-code` | Allow custom model code | False |
| `--dtype` | Data type (auto, float16, bfloat16) | auto |
| `--max-model-len` | Max sequence length | Model default |
| `--gpu-memory-utilization` | GPU memory fraction | 0.9 |

## Complete Workflow Example

### Scenario: Caption 10,000 images + Run QA

```bash
# =============================================
# Step 1: Start vLLM Server (once)
# =============================================
cd CAPTIONING/Tiger-VL-Rethinker
sbatch start_vllm_server_simple.sh

# Wait for server to start
tail -f slurm-*-vllm-server.out
# Look for: "Uvicorn running on http://0.0.0.0:8000"

# =============================================
# Step 2: Run Captioning (3 different prompts)
# =============================================
# All use the same server!

# Simple captions
python captioning/caption_qwen3vl_server.py \
  --dataset-path /path/to/images \
  --server-url "http://localhost:8000" \
  --prompt-style SIMPLE

# Short captions
python captioning/caption_qwen3vl_server.py \
  --dataset-path /path/to/images \
  --server-url "http://localhost:8000" \
  --prompt-style SHORT

# Long captions
python captioning/caption_qwen3vl_server.py \
  --dataset-path /path/to/images \
  --server-url "http://localhost:8000" \
  --prompt-style LONG

# =============================================
# Step 3: Run QA Evaluation
# =============================================
python QA/qa_qwen3vl_server.py \
  --question-path /path/to/questions.parquet \
  --dataset-path /path/to/dataset \
  --server-url "http://localhost:8000"

# =============================================
# Step 4: Stop Server (when done)
# =============================================
squeue -u $USER  # Find job ID
scancel <job_id>
```

## Direct Command Line Usage (No SLURM)

If you want to run directly without SLURM:

```bash
# Terminal 1: Start server
vllm serve Qwen/Qwen3-VL-8B-Instruct --port 8000

# Terminal 2: Run captioning
python caption_qwen3vl_server.py \
  --dataset-path /path/to/images \
  --server-url "http://localhost:8000"

# Terminal 3: Run QA
python qa_qwen3vl_server.py \
  --question-path /path/to/questions.parquet \
  --dataset-path /path/to/dataset \
  --server-url "http://localhost:8000"
```

## API Endpoints

The vLLM server exposes OpenAI-compatible endpoints:

### List Models
```bash
curl http://localhost:8000/v1/models
```

### Chat Completions (for multimodal)
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-VL-8B-Instruct",
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "image_url",
            "image_url": {"url": "data:image/jpeg;base64,<BASE64_IMAGE>"}
          },
          {
            "type": "text",
            "text": "Describe this image."
          }
        ]
      }
    ],
    "max_tokens": 256
  }'
```

### Health Check
```bash
curl http://localhost:8000/health
```

## Client Scripts That Work with This Server

All these scripts automatically work with the vLLM server:

### Tiger-VL-Rethinker
- ✅ `captioning/caption_qwen3vl_server.py`
- ✅ `QA/qa_qwen3vl_server.py`

### CaptionQA
- ✅ `qa_image_qwen3vl_server.py`

They all use the same pattern:
```python
import requests

response = requests.post(
    f"{server_url}/v1/chat/completions",
    json={
        "model": "Qwen/Qwen3-VL-8B-Instruct",
        "messages": messages,
        "max_tokens": max_tokens
    }
)
```

## Environment Variables (Optional)

You can also use environment variables:

```bash
# Set server URL
export VLLM_SERVER_URL="http://localhost:8000"

# Then scripts can use it
python caption_qwen3vl_server.py \
  --dataset-path /path/to/images \
  --server-url "$VLLM_SERVER_URL"
```

## Troubleshooting

### Issue: Command not found
```bash
# Install vLLM
pip install vllm

# Or update
pip install --upgrade vllm
```

### Issue: Port already in use
```bash
# Use different port
vllm serve Qwen/Qwen3-VL-8B-Instruct --port 8001

# Then update scripts
--server-url "http://localhost:8001"
```

### Issue: Out of memory
```bash
# Reduce max sequence length
vllm serve Qwen/Qwen3-VL-8B-Instruct --max-model-len 4096

# Or adjust GPU memory
vllm serve Qwen/Qwen3-VL-8B-Instruct --gpu-memory-utilization 0.8
```

### Issue: Model not loading
```bash
# Check if model exists locally
ls ~/.cache/huggingface/hub/ | grep Qwen3-VL

# Or download first
huggingface-cli download Qwen/Qwen3-VL-8B-Instruct
```

## Advanced Configuration

### Multi-GPU Setup
```bash
# Use 2 GPUs
vllm serve Qwen/Qwen3-VL-8B-Instruct --tensor-parallel-size 2

# Update SLURM script
#SBATCH --gres=gpu:2
TENSOR_PARALLEL_SIZE=2
```

### Custom API Keys (Optional)
```bash
# Start server with API key
vllm serve Qwen/Qwen3-VL-8B-Instruct --api-key "your-secret-key"

# Client side
curl http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer your-secret-key" \
  -H "Content-Type: application/json" \
  -d '{...}'
```

### Monitoring
```bash
# Watch GPU usage
watch -n 1 nvidia-smi

# Check server logs
tail -f slurm-*-vllm-server.out

# Monitor requests
curl http://localhost:8000/metrics
```

## Comparison: Server vs Local

| Aspect | vLLM Server | Local Loading |
|--------|-------------|---------------|
| **Command** | `vllm serve MODEL` | Load in Python script |
| **Startup** | Once (30-60s) | Every job (30-60s each) |
| **Memory** | Shared across jobs | Per-job overhead |
| **Scalability** | Many clients | One at a time |
| **Setup** | Start server first | Self-contained |
| **Best For** | Production, multiple jobs | Quick tests, single task |

## Summary

**Simple Workflow:**
1. `vllm serve Qwen/Qwen3-VL-8B-Instruct` → Start server
2. `python script.py --server-url http://localhost:8000` → Run any script
3. All scripts use the same server automatically!

**Key Benefits:**
- ✅ One simple command to start server
- ✅ OpenAI-compatible API
- ✅ All existing scripts work without changes
- ✅ Efficient resource sharing
- ✅ Easy to monitor and debug

That's it! Just use `vllm serve` and your scripts handle the rest.
