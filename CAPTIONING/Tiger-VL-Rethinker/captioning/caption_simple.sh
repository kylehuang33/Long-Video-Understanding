#!/bin/bash
#SBATCH --job-name=qwen3vl_simple
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm-%j-simple.out
#SBATCH --error=slurm-%j-simple.err

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate captionqa

# Configuration
DATASET_PATH="/mnt/data-alpha-sg-02/team-agent/ai_glasses/datasets/ViRL39K/images"
OUTPUT_PATH="../output/captions/"
PROMPT_STYLE="SIMPLE"
MODEL_PATH="Qwen/Qwen3-VL-4B-Instruct"
MAX_NEW_TOKENS=1024

echo "========================================"
echo "Qwen3VL Image Captioning - SIMPLE"
echo "========================================"
echo "Dataset: $DATASET_PATH"
echo "Output: $OUTPUT_PATH"
echo "Prompt Style: $PROMPT_STYLE"
echo "Model: $MODEL_PATH"
echo "========================================"
echo ""

# Run captioning
python caption_qwen3vl.py \
  --dataset-path "$DATASET_PATH" \
  --output-path "$OUTPUT_PATH" \
  --prompt-style "$PROMPT_STYLE" \
  --model-path "$MODEL_PATH" \
  --max-new-tokens "$MAX_NEW_TOKENS"

echo ""
echo "========================================"
echo "Captioning complete!"
echo "Results saved to: $OUTPUT_PATH"
echo "========================================"
