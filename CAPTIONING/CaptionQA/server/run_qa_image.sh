#!/bin/bash
#SBATCH --job-name=qa_image
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --nodes=1
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm-%j-qa-image.out
#SBATCH --error=slurm-%j-qa-image.err

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate captionqa

# Configuration
SPLIT="all"  # Options: natural, document, ecommerce, embodiedai, all
VLLM_SERVER_URL="http://localhost:8000"
OUTPUT_DIR="./outputs"
MODEL="Qwen/Qwen3-VL-8B-Instruct"
MAX_TOKENS=128

echo "========================================"
echo "Image-Based QA Evaluation"
echo "========================================"
echo "Split: $SPLIT"
echo "vLLM Server: $VLLM_SERVER_URL"
echo "Output Dir: $OUTPUT_DIR"
echo "Model: $MODEL"
echo "========================================"
echo ""

# Run QA evaluation
python qa_image.py \
  --split "$SPLIT" \
  --vllm-server-url "$VLLM_SERVER_URL" \
  --output-dir "$OUTPUT_DIR" \
  --model "$MODEL" \
  --max-tokens "$MAX_TOKENS"

echo ""
echo "========================================"
echo "QA evaluation complete!"
echo "========================================"
