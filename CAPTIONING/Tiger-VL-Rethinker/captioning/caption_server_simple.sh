#!/bin/bash
#SBATCH --job-name=caption_server
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --nodes=1
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm-%j-caption-server.out
#SBATCH --error=slurm-%j-caption-server.err

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate captionqa

# Configuration
DATASET_PATH="/mnt/data-alpha-sg-02/team-agent/ai_glasses/datasets/ViRL39K/images"
OUTPUT_DIR="./output"
SERVER_URL="http://localhost:8000"  # Change this to your vLLM server URL
PROMPT_STYLE="SIMPLE"
MAX_TOKENS=256

echo "========================================"
echo "Qwen3VL Image Captioning (Server Mode)"
echo "========================================"
echo "Dataset: $DATASET_PATH"
echo "Output: $OUTPUT_DIR"
echo "Server: $SERVER_URL"
echo "Prompt Style: $PROMPT_STYLE"
echo "========================================"
echo ""

# Run captioning
python caption_qwen3vl_server.py \
  --dataset-path "$DATASET_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --server-url "$SERVER_URL" \
  --prompt-style "$PROMPT_STYLE" \
  --max-tokens "$MAX_TOKENS"

echo ""
echo "========================================"
echo "Captioning complete!"
echo "========================================"
