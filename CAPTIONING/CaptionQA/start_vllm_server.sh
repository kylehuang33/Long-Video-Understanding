#!/bin/bash
#SBATCH --job-name=vllm_qwen3vl
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --time=7-00:00:00
#SBATCH --output=slurm-%j-vllm-server.out
#SBATCH --error=slurm-%j-vllm-server.err

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate captionqa

# Configuration
MODEL_NAME="Qwen/Qwen3-VL-8B-Instruct"
HOST="0.0.0.0"
PORT=8000
TENSOR_PARALLEL_SIZE=1

echo "========================================"
echo "Starting vLLM Server for CaptionQA"
echo "========================================"
echo "Model: $MODEL_NAME"
echo "Host: $HOST"
echo "Port: $PORT"
echo "Tensor Parallel Size: $TENSOR_PARALLEL_SIZE"
echo "========================================"
echo ""

# Start vLLM server using the simple 'vllm serve' command
vllm serve "$MODEL_NAME" \
  --host "$HOST" \
  --port "$PORT" \
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
  --trust-remote-code \
  --dtype auto \
  --max-model-len 8192

echo ""
echo "========================================"
echo "vLLM server stopped"
echo "========================================"
