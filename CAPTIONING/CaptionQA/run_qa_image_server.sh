#!/bin/bash
#SBATCH --job-name=qa_image_server
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --nodes=1
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm-%j-qa-image-server.out
#SBATCH --error=slurm-%j-qa-image-server.err

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate captionqa

# Configuration
DATASET="Borise/CaptionQA"
SPLIT="all"  # Options: natural, document, ecommerce, embodiedai, all
SERVER_URL="http://localhost:8000"  # Change this to your vLLM server URL
OUTPUT_DIR="./outputs"
MODEL_NAME="Qwen3-VL-4B-Instruct"
MAX_TOKENS=128
TEMPERATURE=0.0

echo "========================================"
echo "Image-Based QA with Qwen3VL (Server Mode)"
echo "========================================"
echo "Dataset: $DATASET"
echo "Split: $SPLIT"
echo "Output: $OUTPUT_DIR"
echo "Server: $SERVER_URL"
echo "Model: $MODEL_NAME"
echo "========================================"
echo ""

# Run QA evaluation
python qa_image_qwen3vl_server.py \
  --dataset "$DATASET" \
  --split "$SPLIT" \
  --server-url "$SERVER_URL" \
  --output-dir "$OUTPUT_DIR" \
  --model-name "$MODEL_NAME" \
  --max-tokens "$MAX_TOKENS" \
  --temperature "$TEMPERATURE"

echo ""
echo "========================================"
echo "QA evaluation complete!"
echo "========================================"
