#!/bin/bash
#SBATCH --job-name=qa_image_local
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm-%j-qa-image-local.out
#SBATCH --error=slurm-%j-qa-image-local.err

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate captionqa

# Configuration
SPLIT="all"  # Options: natural, document, ecommerce, embodiedai, all
MODEL="Qwen/Qwen3-VL-8B-Instruct"
BATCH_SIZE=4
OUTPUT_DIR="./outputs"
MAX_TOKENS=128
DATASET="/mnt/data-alpha-sg-02/team-agent/ai_glasses/datasets/CaptionQA"

echo "========================================"
echo "Image-Based QA Evaluation (Local)"
echo "========================================"
echo "Split: $SPLIT"
echo "Model: $MODEL"
echo "Batch Size: $BATCH_SIZE"
echo "Output Dir: $OUTPUT_DIR"
echo "========================================"
echo ""

# Run QA evaluation
python qa_image_local.py \
  --split "$SPLIT" \
  --model "$MODEL" \
  --dataset $DATASET \
  --batch-size "$BATCH_SIZE" \
  --output-dir "$OUTPUT_DIR" \
  --max-tokens "$MAX_TOKENS"

echo ""
echo "========================================"
echo "QA evaluation complete!"
echo "========================================"
