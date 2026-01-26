# ✅ FINAL - All Scripts Updated to Use requests.post()

## What Changed

All scripts now use **`requests.post()`** to call vLLM server API directly, matching the pattern in `@CAPTIONING/CaptionQA/server/qa_image.py`.

### Before (OpenAI Client)
```python
from openai import OpenAI

client = OpenAI(
    api_key="EMPTY",
    base_url="http://localhost:8000/v1"
)

response = client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": content}],
    max_tokens=max_tokens
)
```

### After (Direct requests.post())
```python
import requests

response = requests.post(
    f"{server_url}/v1/chat/completions",
    json={
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "temperature": temperature
    },
    headers={"Content-Type": "application/json"}
)

if response.status_code != 200:
    raise Exception(f"Server error: {response.status_code} - {response.text}")

result = response.json()
answer = result['choices'][0]['message']['content'].strip()
```

## Updated Files

### qa_direct/
- ✅ `qa_direct_vllm.py` - Now uses `requests.post()`

### qa_caption/
- ✅ `caption_images_vllm.py` - Now uses `requests.post()`
- ✅ `qa_with_captions_vllm.py` - Now uses `requests.post()`
- ✅ `qa_caption_vllm.py` - Now uses `requests.post()`

## Dependencies Updated

### Old
```bash
pip install pandas pyarrow openai tqdm
```

### New
```bash
pip install pandas pyarrow requests tqdm
```

## Usage (Unchanged)

### 1. Start vLLM Server
```bash
vllm serve Qwen/Qwen3-VL-4B-Instruct --port 8000
```

### 2. Run Scripts
```bash
# Direct QA
cd qa_direct
./run_qa_direct.sh

# Caption-based QA (separate steps)
cd qa_caption
./run_caption.sh
./run_qa_with_captions.sh

# Caption-based QA (all-in-one)
cd qa_caption
./run_qa_caption.sh
```

## Key Features (Unchanged)

1. ✅ **Separated Outputs**
   - Captions: `results_caption/captions/{model}/{model}_{style}.json`
   - QA results: `results_caption/qa_results/{model}/{model}_with_{caption_model}_{style}.json`

2. ✅ **Caption Reusability**
   - Caption once, use for multiple QA experiments

3. ✅ **Flexible Prompt Styles**
   - SIMPLE, SHORT, LONG

4. ✅ **Model Name Sanitization**
   - "Qwen/Qwen3-VL-4B-Instruct" → "Qwen3-VL-4B-Instruct"

5. ✅ **Direct vLLM Server API Calls**
   - Uses `requests.post()` like `@CAPTIONING/CaptionQA/server/qa_image.py`

## Verification

```bash
# Check no OpenAI imports remain
grep -r "from openai import OpenAI" qa_direct/ qa_caption/
# Should return nothing

# Check requests imports are present
grep -r "import requests" qa_direct/ qa_caption/
# Should show all 4 files
```

## All Requirements Met

1. ✅ Caption and QA outputs separated
2. ✅ Captioning and QA with captions - separate scripts with bash files
3. ✅ Uses vLLM server with `requests.post()` (not OpenAI client)
4. ✅ Follows pattern from `@CAPTIONING/CaptionQA/server/qa_image.py`

## Ready to Use!

All scripts are now ready to use with your vLLM server. Just start the server and run the bash scripts!
