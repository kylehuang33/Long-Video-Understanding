# Azure OpenAI Implementation Flow

## Overview
This document describes the complete flow of how Azure OpenAI integration works in the caption.py script for the CaptionQA dataset.

---

## 1. Initialization Phase

### 1.1 Command Line Execution
When the user runs the script:
```bash
python CAPTIONING/CaptionQA/caption.py \
  --dataset "/path/to/CaptionQA" \
  --split "natural" \
  --output-dir "./outputs" \
  --model "gpt-5.1-chat" \
  --backend azure_openai \
  --prompt "SIMPLE"
```

### 1.2 Argument Parsing (caption.py:418-476)
- The `argparse` parser processes all command-line arguments
- Key arguments for Azure OpenAI:
  - `--model`: The Azure OpenAI deployment name (e.g., "gpt-5.1-chat", "gpt-4o")
  - `--backend`: Explicitly set to "azure_openai" or auto-detected
  - `--dataset`: Path to CaptionQA dataset
  - `--split`: Which domain split to use (natural, document, ecommerce, embodiedai, all)
  - `--prompt`: Which prompt template to use (SIMPLE, DETAILED, etc.)

### 1.3 Output Path Construction (caption.py:471-475)
```python
model_safe = make_model_safe(args.model)  # "gpt-5.1-chat" → "gpt-5.1-chat"
out_dir = os.path.join(args.output_dir, (args.prompt or "").lower())  # "./outputs/simple"
args.output_path = os.path.join(out_dir, f"{model_safe}.json")  # "./outputs/simple/gpt-5.1-chat.json"
```

---

## 2. Backend Detection Phase

### 2.1 Automatic Backend Detection (caption.py:33-51)
If `--backend` is not explicitly specified:

```python
def detect_model_backend(model: str) -> str:
    model_lower = model.lower()

    # Check for Azure OpenAI (explicit "azure" mention)
    if 'azure' in model_lower:
        return 'azure_openai'

    # Check other backends...
    # Falls back to 'openai' for gpt-* models
```

**Detection Logic:**
1. If model name contains "azure" → `azure_openai`
2. If model name contains "qwen3-vl" → `qwenvl`
3. If model name contains "gemini" → `gemini`
4. If model name contains "claude" → `claude`
5. Default → `openai`

### 2.2 Backend Override (caption.py:283)
If `--backend azure_openai` is explicitly provided:
```python
backend = args.backend if getattr(args, 'backend', None) else detect_model_backend(args.model)
```

This overrides auto-detection and forces Azure OpenAI backend.

---

## 3. Dataset Loading Phase

### 3.1 Load CaptionQA Dataset (caption.py:278)
```python
dataset = load_captionqa_dataset(args.dataset, args.split)
```

**What happens:**
- Loads HuggingFace dataset from local path or HF hub
- Each entry contains:
  - `id`: Image identifier (e.g., "nat_001", "doc_042")
  - `images`: List of PIL Image objects
  - Other metadata fields

**Example dataset entry:**
```python
{
    "id": "nat_001",
    "images": [<PIL.Image.Image>, <PIL.Image.Image>],  # Can be multiple images
    "domain": "natural",
    # ... other fields
}
```

---

## 4. Client Initialization Phase

### 4.1 Azure OpenAI Client Creation (caption.py:288-300)

When `backend == 'azure_openai'`:
```python
if backend == 'azure_openai':
    client = AMD_azure_openai_client()
```

### 4.2 Inside AMD_azure_openai_client() (pipeline/api.py:105-137)

**Step 1: Check OpenAI SDK availability**
```python
if _openai is None:
    raise OptionalDependencyError("openai is not installed. pip install 'openai>=1.0.0'")
```

**Step 2: Read environment variables**
```python
api_key = os.getenv("AZURE_OPENAI_API_KEY")
if not api_key:
    raise ValueError("AZURE_OPENAI_API_KEY environment variable is not set.")

endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "").rstrip("/")
if not endpoint:
    raise ValueError("AZURE_OPENAI_ENDPOINT environment variable is not set.")
```

**Step 3: Create OpenAI client with Azure configuration**
```python
client = _openai.OpenAI(
    api_key=api_key,
    base_url=f"{endpoint}/openai/v1/",  # e.g., "https://YOUR-RESOURCE.openai.azure.com/openai/v1/"
)
return client
```

**Key Point:** Azure OpenAI uses the standard OpenAI SDK, just with a different `base_url` and `api_key`.

---

## 5. Prompt Loading Phase

### 5.1 Determine Prompt (caption.py:311-328)

**Option A: Using taxonomy file**
```python
if args.taxonomy:
    taxonomy = load_json(args.taxonomy)
    tax_prompts = create_taxonomy_prompts(taxonomy, prompt_name="default")
```

**Option B: Using named prompt**
```python
else:
    prompt_text = get_prompt(args.prompt)  # e.g., get_prompt("SIMPLE")
```

**Example prompts:**
- `SIMPLE`: "Write a short caption for this image."
- `DETAILED`: "Write a detailed, comprehensive caption for this image..."
- `TECHNICAL`: "Provide a technical description of this image..."

---

## 6. Image Processing Loop

### 6.1 Iterate Through Dataset (caption.py:334-395)

For each entry in the dataset:

**Step 1: Get image ID and check if already processed**
```python
for entry in tqdm(dataset, desc="Captioning"):
    image_id = entry.get('id')  # e.g., "nat_001"
    image_key = str(image_id)

    # Skip if already processed
    if image_key in results and not args.overwrite:
        continue
```

**Step 2: Extract PIL images from dataset**
```python
images = entry.get('images', [])  # List of PIL Image objects
```

**Step 3: Convert PIL images to temporary files**
```python
image_paths = []
for img in images:
    if isinstance(img, Image.Image):
        temp_path = save_pil_image_to_temp(img)  # Saves to /tmp/tmp*.jpg
        image_paths.append(temp_path)
        temp_files.append(temp_path)  # Track for cleanup
```

**Why temporary files?**
- APIs expect file paths or URLs, not PIL objects
- Temporary files are deleted after processing

**Step 4: Adjust prompt for multi-image entries**
```python
current_prompt = prompt_text
if len(image_paths) > 1:
    if 'MULTIVIEW' in CAPTION_PROMPTS:
        current_prompt = get_prompt('MULTIVIEW')
    else:
        current_prompt = f"You are viewing {len(image_paths)} related images. {prompt_text}"
```

---

## 7. Caption Generation Phase

### 7.1 Call generate_caption() (caption.py:375-384)

```python
caption = generate_caption(
    client=client,
    model=args.model,
    image_paths=image_paths,
    prompt=current_prompt,
    temperature=args.temperature,
    max_tokens=args.max_tokens,
    retries=args.retries,
    backend=backend
)
```

### 7.2 Inside generate_caption() (caption.py:123-270)

**Retry Loop:**
```python
for attempt in range(retries + 1):
    try:
        # ... generate caption
    except openai.OpenAIError as e:
        print(f"[api_error] attempt {attempt + 1}: {e}")
        if attempt < retries:
            _sleep_backoff(attempt)  # Exponential backoff
        continue
```

**For Azure OpenAI backend (caption.py:227-257):**

Since Azure OpenAI uses the same API as standard OpenAI, it goes through the `else` branch:

**Step 1: Encode all images to base64**
```python
encoded_images = [encode_image(img_path) for img_path in image_paths]
```

**What encode_image() does (pipeline/utils.py:14-23):**
```python
def encode_image(image):
    if isinstance(image, str):  # File path
        with open(image, 'rb') as image_file:
            byte_data = image_file.read()
    else:  # PIL Image
        output_buffer = BytesIO()
        image.save(output_buffer, format="PNG")
        byte_data = output_buffer.getvalue()

    base64_str = base64.b64encode(byte_data).decode("utf-8")
    return base64_str
```

**Step 2: Build content items with images**
```python
content_items = []
for encoded_image in encoded_images:
    content_items.append({
        "type": "image_url",
        "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}
    })
```

**Example content_items:**
```python
[
    {
        "type": "image_url",
        "image_url": {
            "url": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEAYABgAAD..."
        }
    },
    {
        "type": "image_url",
        "image_url": {
            "url": "data:image/jpeg;base64,iVBORw0KGgoAAAANSUhEUgAAA..."
        }
    }
]
```

**Step 3: Build messages in OpenAI format**
```python
messages = [
    {
        "role": "user",
        "content": content_items,  # All images
    },
    {
        "role": "user",
        "content": prompt  # Text prompt
    },
]
```

**Example messages:**
```json
[
    {
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
        ]
    },
    {
        "role": "user",
        "content": "Write a short caption for this image."
    }
]
```

**Step 4: Call OpenAI API via AMD_openai_call()**
```python
completion = AMD_openai_call(
    client,
    model,
    messages=messages,
    temperature=temperature,
    stream=False,
    max_tokens=max_tokens
)
```

### 7.3 Inside AMD_openai_call() (pipeline/api.py:126-152)

**The actual API call:**
```python
def AMD_openai_call(client: Any, model_id: str, messages: Union[str, List[dict]], **kwargs: Any) -> Any:
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]

    try:
        return client.chat.completions.create(
            model=model_id,
            messages=messages,
            **kwargs  # temperature, max_tokens, stream, etc.
        )
    except _BadRequestError as e:
        # Handle edge cases (e.g., reasoning_effort not supported)
        # ...retry logic
```

**What happens:**
1. The Azure OpenAI client (created with Azure endpoint) makes an HTTP POST request to:
   - URL: `https://YOUR-RESOURCE.openai.azure.com/openai/v1/chat/completions`
   - Headers: `{"Authorization": "Bearer {AZURE_API_KEY}"}`
   - Body: JSON with model, messages, temperature, max_tokens

2. Azure OpenAI processes the request:
   - Decodes the base64 images
   - Analyzes the images with the vision model (e.g., GPT-4o, GPT-5.1)
   - Generates a caption based on the prompt

3. Returns a completion object:
```python
{
    "id": "chatcmpl-...",
    "choices": [
        {
            "message": {
                "role": "assistant",
                "content": "A sunny beach with clear blue water and white sand..."
            },
            "finish_reason": "stop"
        }
    ],
    "model": "gpt-5.1-chat",
    "usage": {...}
}
```

**Step 5: Extract caption text**
```python
caption = completion.choices[0].message.content.strip()
return caption
```

---

## 8. Results Storage Phase

### 8.1 Save Caption (caption.py:386-393)

After successful caption generation:

```python
if caption:
    # Save with dataset ID as key
    results[image_key] = caption  # {"nat_001": "A sunny beach with..."}

    # Save after each successful caption (incremental saving)
    with open(args.output_path, 'w') as f:
        json.dump(results, f, indent=2)
else:
    print(f"Failed to generate caption for {image_key}")
```

**Output JSON format:**
```json
{
  "nat_001": "A sunny beach with clear blue water and white sand...",
  "nat_002": "A mountain landscape with snow-capped peaks...",
  "nat_003": "An urban street scene with tall buildings...",
  ...
}
```

**Key feature:** Incremental saving ensures data is not lost if the script crashes mid-execution.

---

## 9. Cleanup Phase

### 9.1 Delete Temporary Files (caption.py:396-401)

After processing all images:

```python
for temp_file in temp_files:
    try:
        os.unlink(temp_file)  # Delete /tmp/tmp*.jpg files
    except Exception:
        pass
```

### 9.2 Final Save (caption.py:403-408)

Ensure all results are written:

```python
try:
    with open(args.output_path, 'w') as f:
        json.dump(results, f, indent=2)
except Exception as e:
    print(f"Error writing results to {args.output_path}: {e}")
```

### 9.3 Print Summary (caption.py:410-415)

```python
print(f"Captioning complete! Results saved to {args.output_path}")
print(f"Total captions: {len(results)}")

elapsed = time.time() - start_time
_mins, _secs = divmod(int(elapsed), 60)
_hours, _mins = divmod(_mins, 60)
print(f"Total caption time: {_hours:02d}:{_mins:02d}:{_secs:02d} ({elapsed:.2f}s)")
```

**Example output:**
```
Captioning complete! Results saved to ./outputs/simple/gpt-5.1-chat.json
Total captions: 500
Total caption time: 00:15:42 (942.35s)
```

---

## 10. Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. User runs caption.py with --backend azure_openai            │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│ 2. Backend Detection                                            │
│    - detect_model_backend() or explicit --backend flag          │
│    - Result: backend = "azure_openai"                           │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│ 3. Load CaptionQA Dataset from HuggingFace                      │
│    - Returns list of entries with PIL images                    │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│ 4. Initialize Azure OpenAI Client                               │
│    - Read AZURE_OPENAI_API_KEY from env                         │
│    - Read AZURE_OPENAI_ENDPOINT from env                        │
│    - Create OpenAI client with Azure base_url                   │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│ 5. Load Prompt Template                                         │
│    - get_prompt("SIMPLE") → "Write a short caption..."          │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│ 6. FOR EACH image in dataset:                                   │
│    ┌─────────────────────────────────────────────────────────┐ │
│    │ a. Extract PIL images from entry                        │ │
│    │ b. Save PIL images to temp files (/tmp/tmp*.jpg)        │ │
│    │ c. Encode images to base64                              │ │
│    │ d. Build OpenAI messages format:                        │ │
│    │    - User message with image_url (base64 data URLs)     │ │
│    │    - User message with text prompt                      │ │
│    │ e. Call Azure OpenAI API via AMD_openai_call()          │ │
│    │    ┌────────────────────────────────────────────────┐   │ │
│    │    │ - POST to Azure endpoint with images + prompt │   │ │
│    │    │ - Azure GPT model analyzes images             │   │ │
│    │    │ - Returns caption text                        │   │ │
│    │    └────────────────────────────────────────────────┘   │ │
│    │ f. Extract caption from response                        │ │
│    │ g. Save to results dict {"nat_001": "caption..."}       │ │
│    │ h. Write to JSON file (incremental save)                │ │
│    └─────────────────────────────────────────────────────────┘ │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│ 7. Cleanup                                                      │
│    - Delete temporary image files                               │
│    - Final save to JSON                                         │
│    - Print summary statistics                                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 11. Error Handling

### 11.1 Environment Variable Errors
```python
# If AZURE_OPENAI_API_KEY is not set
ValueError: AZURE_OPENAI_API_KEY environment variable is not set.

# If AZURE_OPENAI_ENDPOINT is not set
ValueError: AZURE_OPENAI_ENDPOINT environment variable is not set.
```

### 11.2 API Errors with Retry Logic
```python
# Retry up to 2 times (configurable via --retries)
[api_error] attempt 1: Rate limit exceeded
# Exponential backoff: sleep 0.5s, 1s, 2s...
[api_error] attempt 2: Rate limit exceeded
Failed to generate caption for nat_042
```

### 11.3 Image Processing Errors
```python
# If PIL image cannot be saved
Skipping nat_042 (could not process images)
```

---

## 12. Key Design Decisions

### 12.1 Why Reuse AMD_openai_call()?
- Azure OpenAI API is 100% compatible with standard OpenAI API
- Only difference is the endpoint URL (handled in client initialization)
- Reduces code duplication
- Easier maintenance

### 12.2 Why Use Temporary Files?
- HuggingFace datasets return PIL Image objects
- APIs expect file paths or base64-encoded strings
- Temporary files bridge this gap
- Automatic cleanup prevents disk space issues

### 12.3 Why Incremental Saving?
- Long-running jobs may crash or timeout
- Incremental saving preserves progress
- Can resume from where it left off (with --overwrite flag control)

### 12.4 Why Separate Backend Detection?
- Allows flexible model naming (e.g., "azure-gpt-4o", "gpt-5.1-chat")
- Explicit override via --backend flag
- Falls back to sensible defaults

---

## 13. Complete Example End-to-End

**Step-by-step example:**

1. **User sets environment variables:**
```bash
export AZURE_OPENAI_API_KEY="sk-..."
export AZURE_OPENAI_ENDPOINT="https://my-resource.openai.azure.com"
```

2. **User runs script:**
```bash
python CAPTIONING/CaptionQA/caption.py \
  --dataset "/data/CaptionQA" \
  --split "natural" \
  --output-dir "./outputs" \
  --model "gpt-5.1-chat" \
  --backend azure_openai \
  --prompt "SIMPLE"
```

3. **Script execution:**
```
Loading dataset /data/CaptionQA (split: natural)...
Loaded 500 entries
Using azure_openai backend for model gpt-5.1-chat
Using prompt: SIMPLE
Write a short caption for this image.
Saving outputs to ./outputs/simple/gpt-5.1-chat.json...
Processing 500 images from /data/CaptionQA (natural split)
Captioning: 100%|██████████| 500/500 [15:42<00:00, 1.88s/it]
Captioning complete! Results saved to ./outputs/simple/gpt-5.1-chat.json
Total captions: 500
Total caption time: 00:15:42 (942.35s)
```

4. **Output file (./outputs/simple/gpt-5.1-chat.json):**
```json
{
  "nat_001": "A scenic beach view with turquoise water and palm trees swaying in the breeze.",
  "nat_002": "A mountain landscape featuring snow-capped peaks and a clear blue sky.",
  ...
}
```

---

## 14. Comparison with Other Backends

| Aspect | Azure OpenAI | Standard OpenAI | Claude | Gemini |
|--------|-------------|-----------------|--------|--------|
| Client Init | `AMD_azure_openai_client()` | `AMD_openai_client()` | `AMD_claude_client()` | `AMD_gemini_client()` |
| Image Format | Base64 data URL | Base64 data URL | Base64 with media_type | File paths directly |
| API Call | `AMD_openai_call()` (shared) | `AMD_openai_call()` (shared) | `AMD_claude_call()` | `AMD_gemini_call()` |
| Message Format | OpenAI format | OpenAI format | Anthropic format | Google GenAI format |
| Env Variables | `AZURE_OPENAI_*` | `OPENAI_API_KEY` | N/A (AMD gateway) | N/A (AMD gateway) |

---

## 15. Testing Checklist

- [ ] Environment variables are set correctly
- [ ] Azure OpenAI client initializes without errors
- [ ] Backend detection works for "azure" keyword in model name
- [ ] Backend can be explicitly set with --backend flag
- [ ] Images are encoded to base64 correctly
- [ ] Multi-image entries work correctly
- [ ] API calls succeed and return captions
- [ ] Results are saved to correct JSON path
- [ ] Incremental saving works (file updates after each caption)
- [ ] Temporary files are cleaned up
- [ ] Error handling works for API failures
- [ ] Retry logic works with exponential backoff

---

## Summary

The Azure OpenAI integration seamlessly fits into the existing caption.py architecture by:

1. **Adding a new client function** that configures the OpenAI SDK for Azure
2. **Reusing existing API call logic** since Azure OpenAI is API-compatible
3. **Following established patterns** from other backend integrations
4. **Supporting the same dataset and prompts** as all other backends
5. **Providing flexible backend selection** via auto-detection or explicit flags

The implementation is minimal, maintainable, and consistent with the codebase's existing design patterns.
