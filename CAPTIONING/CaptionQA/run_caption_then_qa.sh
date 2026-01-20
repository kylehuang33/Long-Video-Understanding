#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
  cat <<'EOF'
Run captioning first, then QA evaluation using the generated caption file.

Usage:
  bash run_caption_then_qa.sh [options]

Options:
  --dataset DATASET             CaptionQA dataset name/path (default: Borise/CaptionQA)
  --split SPLIT                 Split: natural|document|ecommerce|embodiedai|all (default: all)
  --prompt PROMPT               Prompt name (default: SIMPLE)
  --output-dir DIR              Output dir for captions (default: ./example_captions)
  --caption-model MODEL         Captioning model id/path
  --caption-backend BACKEND     Caption backend (openai|gemini|claude|vllm|vllm_server|qwenvl)
  --caption-temperature FLOAT   Captioning temperature
  --caption-max-tokens INT      Captioning max tokens
  --vllm-server-url URL         vLLM server URL for captioning (optional)
  --overwrite                   Overwrite existing captions
  --qa-model MODEL              QA model id/path
  --qa-backend BACKEND          QA backend (vllm|transformers)
  --qa-max-tokens INT           QA max tokens (default: 4)
  --tp-size INT                 vLLM tensor parallel size (default: 1)
  -h, --help                    Show this help
EOF
}

: "${DATASET:=Borise/CaptionQA}"
: "${SPLIT:=all}"
: "${PROMPT:=SIMPLE}"
: "${OUTPUT_DIR:=${SCRIPT_DIR}/example_captions}"
: "${CAPTION_MODEL:=Qwen/Qwen2.5-VL-3B-Instruct}"
: "${CAPTION_BACKEND:=}"
: "${CAPTION_TEMPERATURE:=}"
: "${CAPTION_MAX_TOKENS:=}"
: "${VLLM_SERVER_URL:=}"
: "${OVERWRITE:=0}"
: "${QA_MODEL:=Qwen/Qwen2.5-72B-Instruct}"
: "${QA_BACKEND:=vllm}"
: "${QA_MAX_TOKENS:=4}"
: "${TP_SIZE:=1}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset) DATASET="$2"; shift 2 ;;
    --split) SPLIT="$2"; shift 2 ;;
    --prompt) PROMPT="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --caption-model) CAPTION_MODEL="$2"; shift 2 ;;
    --caption-backend) CAPTION_BACKEND="$2"; shift 2 ;;
    --caption-temperature) CAPTION_TEMPERATURE="$2"; shift 2 ;;
    --caption-max-tokens) CAPTION_MAX_TOKENS="$2"; shift 2 ;;
    --vllm-server-url) VLLM_SERVER_URL="$2"; shift 2 ;;
    --overwrite) OVERWRITE=1; shift ;;
    --qa-model) QA_MODEL="$2"; shift 2 ;;
    --qa-backend) QA_BACKEND="$2"; shift 2 ;;
    --qa-max-tokens) QA_MAX_TOKENS="$2"; shift 2 ;;
    --tp-size) TP_SIZE="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
  esac
done

caption_args=(
  "--output-dir" "$OUTPUT_DIR"
  "--split" "$SPLIT"
  "--prompt" "$PROMPT"
  "--model" "$CAPTION_MODEL"
  "--dataset" "$DATASET"
)
if [[ -n "$CAPTION_BACKEND" ]]; then
  caption_args+=("--backend" "$CAPTION_BACKEND")
fi
if [[ -n "$CAPTION_TEMPERATURE" ]]; then
  caption_args+=("--temperature" "$CAPTION_TEMPERATURE")
fi
if [[ -n "$CAPTION_MAX_TOKENS" ]]; then
  caption_args+=("--max-tokens" "$CAPTION_MAX_TOKENS")
fi
if [[ -n "$VLLM_SERVER_URL" ]]; then
  caption_args+=("--vllm-server-url" "$VLLM_SERVER_URL")
fi
if [[ "$OVERWRITE" -eq 1 ]]; then
  caption_args+=("--overwrite")
fi

echo "==> Captioning..."
python "$SCRIPT_DIR/caption.py" "${caption_args[@]}"

prompt_lower="$(printf '%s' "$PROMPT" | tr '[:upper:]' '[:lower:]')"
model_safe="${CAPTION_MODEL//\//_}"
caption_path="${OUTPUT_DIR}/${prompt_lower}/${model_safe}.json"
echo "==> Caption path: ${caption_path}"

if [[ ! -f "$caption_path" ]]; then
  echo "Caption file not found at: ${caption_path}" >&2
  exit 1
fi

qa_args=(
  "--caption-path" "$caption_path"
  "--split" "$SPLIT"
  "--dataset" "$DATASET"
  "--backend" "$QA_BACKEND"
  "--model" "$QA_MODEL"
  "--max-tokens" "$QA_MAX_TOKENS"
  "--tp-size" "$TP_SIZE"
)

echo "==> QA..."
python "$SCRIPT_DIR/qa.py" "${qa_args[@]}"
