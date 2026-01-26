# Modifications Summary - Final Version

## What Was Changed

### 1. File Organization - Now with Separate Caption and QA Outputs

#### Before
```
qa_caption/
├── qa_caption_vllm.py
├── run_qa_caption.sh
└── results_caption/
    └── Qwen3-VL-4B-Instruct/
        ├── Qwen3-VL-4B-Instruct_simple.json (captions)
        └── Qwen3-VL-4B-Instruct_with_Qwen3-VL-4B-Instruct_simple.json (QA)
```

#### After
```
qa_caption/
├── caption_images_vllm.py          # NEW: Caption only
├── qa_with_captions_vllm.py        # NEW: QA with existing captions
├── qa_caption_vllm.py              # KEPT: All-in-one version
├── run_caption.sh                  # NEW: Run captioning only
├── run_qa_with_captions.sh         # NEW: Run QA with captions
├── run_qa_caption.sh               # KEPT: Run both together
└── results_caption/
    ├── captions/                   # NEW: Separate caption directory
    │   └── Qwen3-VL-4B-Instruct/
    │       └── Qwen3-VL-4B-Instruct_simple.json
    └── qa_results/                 # NEW: Separate QA directory
        └── Qwen3-VL-4B-Instruct/
            └── Qwen3-VL-4B-Instruct_with_Qwen3-VL-4B-Instruct_simple.json
```

### 2. New Scripts Created

#### `caption_images_vllm.py`
- **Purpose**: Caption images only, no QA
- **Input**: Parquet file with image paths
- **Output**: `captions/{model}/{model}_{style}.json`
- **Benefits**:
  - Caption once, reuse multiple times
  - Faster for multiple QA experiments
  - Easier to debug captioning issues

#### `qa_with_captions_vllm.py`
- **Purpose**: QA using existing caption files
- **Input**:
  - Parquet file with questions
  - Existing caption JSON file
- **Output**: `qa_results/{qa_model}/{qa_model}_with_{caption_model}_{style}.json`
- **Benefits**:
  - No need to recaption for each QA run
  - Can use different QA models with same captions
  - Faster experimentation

#### Bash Scripts
- **`run_caption.sh`**: Run captioning only
- **`run_qa_with_captions.sh`**: Run QA with existing captions

### 3. File Naming Convention - Now Separated

#### Caption Files
- **Location**: `results_caption/captions/{caption_model}/`
- **Format**: `{caption_model}_{prompt_style}.json`
- **Example**: `captions/Qwen3-VL-4B-Instruct/Qwen3-VL-4B-Instruct_simple.json`

#### QA Results Files
- **Location**: `results_caption/qa_results/{qa_model}/`
- **Format**: `{qa_model}_with_{caption_model}_{prompt_style}.json`
- **Example**: `qa_results/Qwen3-VL-4B-Instruct/Qwen3-VL-4B-Instruct_with_Qwen3-VL-4B-Instruct_simple.json`

#### Direct QA Files
- **Location**: `results_direct/{qa_model}/`
- **Format**: `{qa_model}_results.json`
- **Example**: `results_direct/Qwen3-VL-4B-Instruct/Qwen3-VL-4B-Instruct_results.json`

### 4. Three Ways to Do Caption-Based QA

#### Option 1: Separate Steps (NEW - Recommended for Experiments)
```bash
# Step 1: Caption
./run_caption.sh

# Step 2: QA
./run_qa_with_captions.sh
```

**Benefits**:
- Caption once, QA multiple times
- Use different QA models with same captions
- Faster for multiple experiments

#### Option 2: All-in-One (KEPT - Quick Testing)
```bash
./run_qa_caption.sh
```

**Benefits**:
- Simple one-step process
- Good for first-time runs

#### Option 3: Manual (Advanced)
```bash
# Caption with model A
python caption_images_vllm.py --model ModelA --prompt-style SIMPLE

# QA with model B using ModelA's captions
python qa_with_captions_vllm.py \
    --caption-file captions/ModelA/ModelA_simple.json \
    --model ModelB
```

**Benefits**:
- Maximum flexibility
- Mix and match models

## Key Benefits of Separation

### 1. Caption and QA Accuracy Separated
✅ **Captions**: `captions/{model}/{model}_{style}.json`
✅ **QA Results**: `qa_results/{model}/{model}_with_{caption_model}_{style}.json`

Now you can:
- Evaluate caption quality independently
- Evaluate QA accuracy independently
- Compare different caption models with same QA model
- Compare different QA models with same captions

### 2. Reusability
```bash
# Caption once
./run_caption.sh  # Creates: captions/ModelA/ModelA_simple.json

# Use with different QA models
python qa_with_captions_vllm.py --caption-file captions/ModelA/ModelA_simple.json --model ModelB
python qa_with_captions_vllm.py --caption-file captions/ModelA/ModelA_simple.json --model ModelC
python qa_with_captions_vllm.py --caption-file captions/ModelA/ModelA_simple.json --model ModelD
```

### 3. Flexibility
```bash
# Generate captions with different styles
./run_caption.sh  # PROMPT_STYLE="SIMPLE"
./run_caption.sh  # PROMPT_STYLE="LONG"

# Compare QA performance with different caption styles
python qa_with_captions_vllm.py --caption-file captions/Model/Model_simple.json
python qa_with_captions_vllm.py --caption-file captions/Model/Model_long.json
```

## Answer to Your Requirements

### Q1: Caption and output accuracy have to be separate?
✅ **YES!** Now completely separated:
- **Captions**: `results_caption/captions/{model}/{model}_{style}.json`
- **QA Results**: `results_caption/qa_results/{model}/{model}_with_{caption_model}_{style}.json`

You can now:
- Evaluate caption quality independently
- Evaluate QA accuracy independently
- Mix and match different caption and QA models

### Q2: Captioning and questioning with caption separately with bash files?
✅ **YES!** Three new files created:

1. **`caption_images_vllm.py`** + **`run_caption.sh`**
   - Caption images only
   - Output: `captions/{model}/{model}_{style}.json`

2. **`qa_with_captions_vllm.py`** + **`run_qa_with_captions.sh`**
   - QA using existing captions
   - Input: Caption file from step 1
   - Output: `qa_results/{model}/{model}_with_{caption_model}_{style}.json`

3. **`qa_caption_vllm.py`** + **`run_qa_caption.sh`** (kept for convenience)
   - All-in-one version
   - Does both captioning and QA in one run

## File Structure Comparison

### Before (Mixed)
```
results_caption/
└── Qwen3-VL-4B-Instruct/
    ├── Qwen3-VL-4B-Instruct_simple.json                    # Captions
    └── Qwen3-VL-4B-Instruct_with_..._simple.json          # QA results
```

### After (Separated)
```
results_caption/
├── captions/                                               # Caption outputs only
│   └── Qwen3-VL-4B-Instruct/
│       ├── Qwen3-VL-4B-Instruct_simple.json
│       ├── Qwen3-VL-4B-Instruct_short.json
│       └── Qwen3-VL-4B-Instruct_long.json
│
└── qa_results/                                             # QA outputs only
    └── Qwen3-VL-4B-Instruct/
        ├── Qwen3-VL-4B-Instruct_with_Qwen3-VL-4B-Instruct_simple.json
        ├── Qwen3-VL-4B-Instruct_with_Qwen3-VL-4B-Instruct_short.json
        └── Qwen3-VL-4B-Instruct_with_Qwen3-VL-4B-Instruct_long.json
```

## Usage Examples

### Example 1: Caption Once, QA Multiple Times
```bash
cd qa_caption

# Step 1: Caption images (do this once)
./run_caption.sh
# Output: captions/Qwen3-VL-4B-Instruct/Qwen3-VL-4B-Instruct_simple.json

# Step 2a: QA with model A
# Edit run_qa_with_captions.sh: QA_MODEL="ModelA"
./run_qa_with_captions.sh
# Output: qa_results/ModelA/ModelA_with_Qwen3-VL-4B-Instruct_simple.json

# Step 2b: QA with model B (reusing same captions!)
# Edit run_qa_with_captions.sh: QA_MODEL="ModelB"
./run_qa_with_captions.sh
# Output: qa_results/ModelB/ModelB_with_Qwen3-VL-4B-Instruct_simple.json
```

### Example 2: Compare Caption Styles
```bash
cd qa_caption

# Caption with SIMPLE
# Edit run_caption.sh: PROMPT_STYLE="SIMPLE"
./run_caption.sh

# Caption with LONG
# Edit run_caption.sh: PROMPT_STYLE="LONG"
./run_caption.sh

# QA with SIMPLE captions
# Edit run_qa_with_captions.sh: CAPTION_FILE points to simple.json
./run_qa_with_captions.sh

# QA with LONG captions
# Edit run_qa_with_captions.sh: CAPTION_FILE points to long.json
./run_qa_with_captions.sh

# Now compare QA accuracy with different caption styles!
```

### Example 3: Quick All-in-One
```bash
cd qa_caption
./run_qa_caption.sh
# Does both captioning and QA in one run
```

## Summary of All Scripts

| Script | Purpose | Input | Output |
|--------|---------|-------|--------|
| `qa_direct_vllm.py` | Direct QA | Images + Questions | `results_direct/{model}/{model}_results.json` |
| `caption_images_vllm.py` | Caption only | Images | `captions/{model}/{model}_{style}.json` |
| `qa_with_captions_vllm.py` | QA with captions | Captions + Questions | `qa_results/{model}/{model}_with_{caption_model}_{style}.json` |
| `qa_caption_vllm.py` | All-in-one | Images + Questions | Both caption and QA files |

## Bash Scripts

| Bash Script | Python Script | Purpose |
|-------------|---------------|---------|
| `run_qa_direct.sh` | `qa_direct_vllm.py` | Direct QA |
| `run_caption.sh` | `caption_images_vllm.py` | Caption only |
| `run_qa_with_captions.sh` | `qa_with_captions_vllm.py` | QA with existing captions |
| `run_qa_caption.sh` | `qa_caption_vllm.py` | All-in-one |
