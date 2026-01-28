#!/bin/bash

# Caption-based QA using vLLM server (all-in-one: caption + QA)
# Make sure vLLM server is running before executing this script!
# Start vLLM server with: vllm serve Qwen/Qwen3-VL-4B-Instruct --port 8000

# Dependencies: pip install pandas pyarrow requests tqdm math-verify

PARQUET_PATH="/mnt/data-alpha-sg-02/team-agent/ai_glasses/datasets/ViRL39K/39Krelease.parquet"
DATASET_ROOT="/mnt/data-alpha-sg-02/team-agent/ai_glasses/datasets/ViRL39K"
OUTPUT_DIR="./results_caption"
VLLM_URL="http://localhost:8000/v1"
CAPTION_MODEL="Qwen/Qwen3-VL-4B-Instruct"
QA_MODEL="Qwen/Qwen3-VL-4B-Instruct"
PROMPT_STYLE="SIMPLE"  # Options: SIMPLE, SHORT, LONG

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run caption-based QA
# Output will be saved to:
#   Captions: {OUTPUT_DIR}/captions/{caption_model}/{caption_model}_{prompt_style}.json
#   Results:  {OUTPUT_DIR}/qa_results/{qa_model}/{qa_model}_with_{caption_model}_{prompt_style}.json
#   Metrics:  {OUTPUT_DIR}/qa_results/{qa_model}/{qa_model}_with_{caption_model}_{prompt_style}_metrics.json
# Example:
#   ./results_caption/captions/Qwen3-VL-4B-Instruct/Qwen3-VL-4B-Instruct_simple.json
#   ./results_caption/qa_results/Qwen3-VL-4B-Instruct/Qwen3-VL-4B-Instruct_with_Qwen3-VL-4B-Instruct_simple.json
#   ./results_caption/qa_results/Qwen3-VL-4B-Instruct/Qwen3-VL-4B-Instruct_with_Qwen3-VL-4B-Instruct_simple_metrics.json
python qa_caption_vllm.py \
    --parquet-path "$PARQUET_PATH" \
    --dataset-root "$DATASET_ROOT" \
    --output-dir "$OUTPUT_DIR" \
    --vllm-url "$VLLM_URL" \
    --caption-model "$CAPTION_MODEL" \
    --qa-model "$QA_MODEL" \
    --prompt-style "$PROMPT_STYLE" \
    --max-tokens 512 \
    --temperature 0.0

echo "Done! Check $OUTPUT_DIR for captions, results, and metrics"
