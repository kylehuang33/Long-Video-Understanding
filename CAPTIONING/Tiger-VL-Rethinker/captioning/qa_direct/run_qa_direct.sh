#!/bin/bash

# Direct QA using vLLM server
# Make sure vLLM server is running before executing this script!
# Start vLLM server with: vllm serve Qwen/Qwen3-VL-4B-Instruct --port 8000

PARQUET_PATH="/mnt/data-alpha-sg-02/team-agent/ai_glasses/datasets/ViRL39K/39Krelease.parquet"
DATASET_ROOT="/mnt/data-alpha-sg-02/team-agent/ai_glasses/datasets/ViRL39K"
OUTPUT_DIR="./results_direct"
VLLM_URL="http://localhost:8000/v1"
MODEL="Qwen/Qwen3-VL-4B-Instruct"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run QA
# Output will be saved to: {OUTPUT_DIR}/{model_safe}/{model_safe}_results.json
# Example: ./results_direct/Qwen3-VL-4B-Instruct/Qwen3-VL-4B-Instruct_results.json
python qa_direct_vllm.py \
    --parquet-path "$PARQUET_PATH" \
    --dataset-root "$DATASET_ROOT" \
    --output-dir "$OUTPUT_DIR" \
    --vllm-url "$VLLM_URL" \
    --model "$MODEL" \
    --max-tokens 512 \
    --temperature 0.0

echo "Done! Check $OUTPUT_DIR for results"
