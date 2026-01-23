#!/bin/bash
#SBATCH --job-name=qwen3vl_qa
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --time=3-00:00:00
#SBATCH --output=slurm-%j-qa.out
#SBATCH --error=slurm-%j-qa.err

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate captionqa

# Configuration
QUESTION_PATH="/mnt/data-alpha-sg-02/team-agent/ai_glasses/datasets/ViRL39K/39Krelease.parquet"
DATASET_PATH="/mnt/data-alpha-sg-02/team-agent/ai_glasses/datasets/ViRL39K"
OUTPUT_DIR="./output/qa"
MODEL_PATH="Qwen/Qwen3-VL-4B-Instruct"
MAX_NEW_TOKENS=128

echo "========================================"
echo "Qwen3VL Question Answering Evaluation"
echo "========================================"
echo "Questions: $QUESTION_PATH"
echo "Dataset: $DATASET_PATH"
echo "Output Directory: $OUTPUT_DIR"
echo "Model: $MODEL_PATH"
echo "========================================"
echo ""

# Run QA evaluation
python qa_qwen3vl.py \
  --question-path "$QUESTION_PATH" \
  --dataset-path "$DATASET_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --model-path "$MODEL_PATH" \
  --max-new-tokens "$MAX_NEW_TOKENS"

echo ""
echo "========================================"
echo "Evaluation complete!"
echo "Check output directory: $OUTPUT_DIR"
echo "========================================"
