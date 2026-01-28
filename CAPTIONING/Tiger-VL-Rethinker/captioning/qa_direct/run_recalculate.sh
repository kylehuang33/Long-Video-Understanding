#!/bin/bash

# Recalculate metrics from existing results JSON using math_verify
# This script re-processes results to use math_verify for answer extraction and comparison

# Dependencies: pip install math-verify

# Input results file (change this to your results file)
INPUT_FILE="./results_direct/Qwen3-VL-4B-Instruct/Qwen3-VL-4B-Instruct_results.json"

# Optional: specify output paths (defaults: overwrite input, metrics in same dir)
# OUTPUT_FILE="./results_direct/Qwen3-VL-4B-Instruct/Qwen3-VL-4B-Instruct_results_recalculated.json"
# METRICS_FILE="./results_direct/Qwen3-VL-4B-Instruct/Qwen3-VL-4B-Instruct_metrics_recalculated.json"

# Check if input file exists
if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: Input file not found: $INPUT_FILE"
    echo "Please update INPUT_FILE variable to point to your results JSON"
    exit 1
fi

# Run recalculation
# By default, this will:
#   - Overwrite the input file with updated results
#   - Create metrics file at: {input_dir}/{input_name}_metrics.json
python recalculate_metrics.py \
    --input "$INPUT_FILE"

# To save to different files, uncomment and use:
# python recalculate_metrics.py \
#     --input "$INPUT_FILE" \
#     --output "$OUTPUT_FILE" \
#     --metrics "$METRICS_FILE"

echo "Done! Check the output files for recalculated results and metrics"
