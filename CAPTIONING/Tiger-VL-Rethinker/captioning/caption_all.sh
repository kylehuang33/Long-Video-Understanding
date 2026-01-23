#!/bin/bash
#SBATCH --job-name=qwen3vl_all
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --time=3-00:00:00
#SBATCH --output=slurm-%j-all.out
#SBATCH --error=slurm-%j-all.err

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate captionqa

# Configuration
DATASET_PATH="/mnt/data-alpha-sg-02/team-agent/ai_glasses/datasets/ViRL39K/images"
MODEL_PATH="Qwen/Qwen3-VL-4B-Instruct"

echo "========================================"
echo "Qwen3VL Image Captioning - ALL STYLES"
echo "========================================"
echo "Dataset: $DATASET_PATH"
echo "Model: $MODEL_PATH"
echo "========================================"
echo ""

# Run SIMPLE captioning
echo "========================================"
echo "Running SIMPLE captioning..."
echo "========================================"
python caption_qwen3vl.py \
  --dataset-path "$DATASET_PATH" \
  --output-path "./captions_simple.json" \
  --prompt-style "SIMPLE" \
  --model-path "$MODEL_PATH" \
  --max-new-tokens 256

echo ""
echo "✓ SIMPLE captioning complete!"
echo ""

# Run SHORT captioning
echo "========================================"
echo "Running SHORT captioning..."
echo "========================================"
python caption_qwen3vl.py \
  --dataset-path "$DATASET_PATH" \
  --output-path "./captions_short.json" \
  --prompt-style "SHORT" \
  --model-path "$MODEL_PATH" \
  --max-new-tokens 128

echo ""
echo "✓ SHORT captioning complete!"
echo ""

# Run LONG captioning
echo "========================================"
echo "Running LONG captioning..."
echo "========================================"
python caption_qwen3vl.py \
  --dataset-path "$DATASET_PATH" \
  --output-path "./captions_long.json" \
  --prompt-style "LONG" \
  --model-path "$MODEL_PATH" \
  --max-new-tokens 512

echo ""
echo "✓ LONG captioning complete!"
echo ""

echo "========================================"
echo "All captioning tasks complete!"
echo "========================================"
echo "Output files:"
echo "  - captions_simple.json"
echo "  - captions_short.json"
echo "  - captions_long.json"
echo "========================================"
