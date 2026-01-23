#!/bin/bash
#SBATCH --job-name=qwen3vl_caption
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --time=1-00:00:00

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate captionqa

# Set dataset path and output path
DATASET_PATH="/mnt/data-alpha-sg-02/team-agent/ai_glasses/datasets/ViRL39K/images"
OUTPUT_PATH="./captions_simple.json"
PROMPT_STYLE="SIMPLE"

# Run captioning
python caption_qwen3vl.py \
  --dataset-path "$DATASET_PATH" \
  --output-path "$OUTPUT_PATH" \
  --prompt-style "$PROMPT_STYLE" \
  --model-path "Qwen/Qwen3-VL-4B-Instruct" \
  --max-new-tokens 256

echo "Captioning complete! Results saved to $OUTPUT_PATH"
