#!/bin/bash

# Caption images using vLLM server
# Make sure vLLM server is running before executing this script!
# Start vLLM server with: vllm serve Qwen/Qwen3-VL-4B-Instruct --port 8000

# Dependencies: pip install pandas pyarrow requests tqdm

PARQUET_PATH="/mnt/data-alpha-sg-02/team-agent/ai_glasses/datasets/ViRL39K/39Krelease.parquet"
DATASET_ROOT="/mnt/data-alpha-sg-02/team-agent/ai_glasses/datasets/ViRL39K"
OUTPUT_DIR="./results_caption"
VLLM_URL="http://localhost:8000/v1"
MODEL="Qwen/Qwen3-VL-4B-Instruct"
PROMPT_STYLE="SIMPLE"  # Options: SIMPLE, SHORT, LONG

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run captioning
# Output will be saved to: {OUTPUT_DIR}/captions/{model}/{model}_{prompt_style}.json
# Example: ./results_caption/captions/Qwen3-VL-4B-Instruct/Qwen3-VL-4B-Instruct_simple.json
python caption_images_vllm.py \
    --parquet-path "$PARQUET_PATH" \
    --dataset-root "$DATASET_ROOT" \
    --output-dir "$OUTPUT_DIR" \
    --vllm-url "$VLLM_URL" \
    --model "$MODEL" \
    --prompt-style "$PROMPT_STYLE" \
    --max-tokens 512

echo "Done! Check $OUTPUT_DIR/captions for caption files"
