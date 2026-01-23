#!/bin/bash
#SBATCH --job-name=caption_5.1        # Sets a name for your job
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --nodes=1                             # Requests 1 node
#SBATCH --time=1-12:00:00                     # Sets the estimated runtime



source ~/miniconda3/etc/profile.d/conda.sh
conda activate captionqa

python caption.py \
  --output-dir ./output/captions \
  --split all \
  --model "gpt-5.1-chat" \
  --backend azure_openai \
  --prompt SIMPLE



