# ✅ UPDATED - Handles Both Multiple Choice and Open-Ended Questions

## Problem Identified

The dataset has **two types of questions**:

### Type 1: Multiple Choice (with options)
```python
{
  'question': '<image>\nIs the number of flowers even or odd?\nChoices:\n(A) odd\n(B) even',
  'answer': '\\boxed{B}'
}
```

### Type 2: Open-Ended (counting/numeric)
```python
{
  'question': '<image>\nHow many cans are there?',
  'answer': '\\boxed{100}'
}
```

## Solution

Updated all QA scripts to automatically detect question type and handle both formats.

## New Functions

### 1. `has_choices(question: str) -> bool`
Detects if question has multiple choice options:
- Looks for patterns like `(A)`, `(B)`, `(C)`
- Looks for "Choices:" keyword
- Returns `True` for multiple choice, `False` for open-ended

### 2. `extract_answer(answer_text: str, question: str) -> Optional[str]`
Extracts answer based on question type:

**For Multiple Choice:**
- Extracts letter (A, B, C, etc.)
- Validates letter is within valid range
- Returns: `"A"`, `"B"`, `"C"`, etc.

**For Open-Ended:**
- Extracts number from response
- Returns: `"100"`, `"20"`, etc.
- Falls back to text (up to 50 chars) if no number found

### 3. `extract_ground_truth(answer: str, question: str) -> Optional[str]`
Extracts ground truth based on question type:

**For Multiple Choice:**
- Parses `\boxed{B}` → returns `"B"`

**For Open-Ended:**
- Parses `\boxed{100}` → returns `"100"`

## Output Format Changes

### Old Field Names (removed):
- ~~`ground_truth_letter`~~
- ~~`predicted_letter`~~

### New Field Names:
- `ground_truth_answer` - Works for both letters and numbers
- `predicted_answer` - Works for both letters and numbers
- `question_type` - Either `"multiple_choice"` or `"open_ended"`

## Example Outputs

### Multiple Choice Question

**Input:**
```json
{
  "question": "<image>\nIs the number of flowers even or odd?\nChoices:\n(A) odd\n(B) even",
  "answer": "\\boxed{B}"
}
```

**Output:**
```json
{
  "qid": "xxx",
  "question": "<image>\nIs the number of flowers even or odd?\nChoices:\n(A) odd\n(B) even",
  "ground_truth": "\\boxed{B}",
  "ground_truth_answer": "B",
  "raw_output": "Looking at the image, I count an even number of flowers. The answer is B.",
  "predicted_answer": "B",
  "is_correct": true,
  "question_type": "multiple_choice",
  "category": "(GradeSchool) Non-Geo Math",
  "source": "Processed",
  "image_paths": ["images/xxx.jpg"]
}
```

### Open-Ended Question

**Input:**
```json
{
  "question": "<image>\nHow many cans are there?",
  "answer": "\\boxed{100}"
}
```

**Output:**
```json
{
  "qid": "yyy",
  "question": "<image>\nHow many cans are there?",
  "ground_truth": "\\boxed{100}",
  "ground_truth_answer": "100",
  "raw_output": "I can see 100 cans arranged in rows in the image.",
  "predicted_answer": "100",
  "is_correct": true,
  "question_type": "open_ended",
  "category": "Spatial Reasoning",
  "source": "Processed",
  "image_paths": ["images/yyy.jpg"]
}
```

### Wrong Answer Example

**Input:**
```json
{
  "question": "<image>\nHow many squares are there?",
  "answer": "\\boxed{100}"
}
```

**Output:**
```json
{
  "qid": "zzz",
  "question": "<image>\nHow many squares are there?",
  "ground_truth": "\\boxed{100}",
  "ground_truth_answer": "100",
  "raw_output": "I count approximately 95 squares in the image.",
  "predicted_answer": "95",
  "is_correct": false,
  "question_type": "open_ended",
  "category": "(GradeSchool) Geometric",
  "source": "Processed",
  "image_paths": ["images/zzz.jpg"]
}
```

## Answer Extraction Logic

### Multiple Choice
```python
# Model output: "The answer is B because..."
# Extraction: Finds "B" → returns "B"

# Model output: "I choose option 2"
# Extraction: Finds "2" → converts to "B" (2nd option)

# Model output: "Based on the image, the answer is (A)"
# Extraction: Finds "A" → returns "A"
```

### Open-Ended
```python
# Model output: "I count 100 cans in the image."
# Extraction: Finds "100" → returns "100"

# Model output: "There are twenty items."
# Extraction: Finds "20" → returns "20"

# Model output: "The answer is 95 squares."
# Extraction: Finds "95" → returns "95"
```

## Comparison Logic

Both types use simple string comparison:
```python
is_correct = (predicted_answer == ground_truth_answer)

# Multiple choice: "B" == "B" → True
# Open-ended: "100" == "100" → True
# Wrong answer: "95" == "100" → False
```

## Metrics Computation

Metrics work the same for both question types:
- Overall accuracy: correct / total
- Per-category accuracy: correct in category / total in category

The `question_type` field allows you to compute separate metrics:
```python
# You can filter by question_type in post-processing
multiple_choice_results = {k: v for k, v in results.items() if v['question_type'] == 'multiple_choice'}
open_ended_results = {k: v for k, v in results.items() if v['question_type'] == 'open_ended'}
```

## Updated Files

All 4 scripts updated:
- ✅ `qa_direct/qa_direct_vllm.py`
- ✅ `qa_caption/caption_images_vllm.py` (no changes needed)
- ✅ `qa_caption/qa_with_captions_vllm.py`
- ✅ `qa_caption/qa_caption_vllm.py`

## Verification

```bash
python3 -m py_compile qa_direct/qa_direct_vllm.py
python3 -m py_compile qa_caption/qa_with_captions_vllm.py
python3 -m py_compile qa_caption/qa_caption_vllm.py
# ✅ All pass
```

## Ready to Use!

The scripts now automatically handle both question types without any configuration needed. Just run them as before:

```bash
# Direct QA
cd qa_direct && ./run_qa_direct.sh

# Caption-based QA
cd qa_caption && ./run_caption.sh && ./run_qa_with_captions.sh
# OR
cd qa_caption && ./run_qa_caption.sh
```

The output will include `question_type` field so you can analyze performance separately for multiple choice vs open-ended questions.
