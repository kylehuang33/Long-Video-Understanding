#!/bin/bash
#SBATCH --job-name=qwen3vl_short
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm-%j-short.out
#SBATCH --error=slurm-%j-short.err

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate captionqa

# Configuration
DATASET_PATH="/mnt/data-alpha-sg-02/team-agent/ai_glasses/datasets/ViRL39K/images"
OUTPUT_PATH="./captions_short.json"
PROMPT_STYLE="SHORT"
MODEL_PATH="Qwen/Qwen3-VL-4B-Instruct"
MAX_NEW_TOKENS=128

echo "========================================"
echo "Qwen3VL Image Captioning - SHORT"
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
