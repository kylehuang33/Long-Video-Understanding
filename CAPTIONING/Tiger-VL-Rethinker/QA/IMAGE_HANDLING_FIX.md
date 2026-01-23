# Image Handling Fix for QA Script

## Problem
The script was failing to load images because:
1. The `image` column in the parquet file is a **numpy array**, not a regular list
2. When converted to string, it became `"['images/ai2d-24027-0.png']"` instead of `"images/ai2d-24027-0.png"`
3. This created invalid paths like: `/path/to/dataset/['images/ai2d-24027-0.png']`
4. Multiple images were not supported

## Solution

### 1. Proper Numpy Array Handling
Added code to detect and convert numpy arrays to regular lists:

```python
# Convert numpy array to list if needed
if hasattr(image_rel_paths, 'tolist'):
    image_rel_paths = image_rel_paths.tolist()
```

### 2. Multiple Image Support
Changed `generate_answer()` to accept a **list of image paths**:

**Before:**
```python
def generate_answer(image_path: str, ...)
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": str(image_path)},
            {"type": "text", "text": question_text},
        ],
    }]
```

**After:**
```python
def generate_answer(image_paths: List[str], ...)
    content = []
    for img_path in image_paths:
        content.append({"type": "image", "image": str(img_path)})
    content.append({"type": "text", "text": question_text})

    messages = [{
        "role": "user",
        "content": content,
    }]
```

### 3. Robust Path Extraction
```python
# Get image paths (handle numpy arrays, lists, tuples)
image_rel_paths = row['image']

# Convert numpy array to list if needed
if hasattr(image_rel_paths, 'tolist'):
    image_rel_paths = image_rel_paths.tolist()
elif not isinstance(image_rel_paths, (list, tuple)):
    image_rel_paths = [image_rel_paths]

# Convert to list and ensure strings
if isinstance(image_rel_paths, (list, tuple)):
    image_rel_paths = [str(p) for p in image_rel_paths]
else:
    image_rel_paths = [str(image_rel_paths)]

# Build full paths
image_paths = [os.path.join(dataset_path, p) for p in image_rel_paths]
```

### 4. Check All Images Exist
```python
# Check if all images exist
missing_images = [p for p in image_paths if not os.path.exists(p)]
if missing_images:
    print(f"\nWarning: Image(s) not found: {missing_images}")
    # Handle error...
```

## Examples

### Single Image Question
Input:
```python
row['image'] = array(['images/ai2d-24027-0.png'], dtype=object)
```

Processing:
```python
image_rel_paths = ['images/ai2d-24027-0.png']
image_paths = ['/path/to/dataset/images/ai2d-24027-0.png']
```

Model receives:
```python
messages = [{
    "role": "user",
    "content": [
        {"type": "image", "image": "/path/to/dataset/images/ai2d-24027-0.png"},
        {"type": "text", "text": "What is the answer?\nOptions:\nA. yes\nB. no"},
    ],
}]
```

### Multiple Image Question
Input:
```python
row['image'] = array(['images/img1.png', 'images/img2.png'], dtype=object)
```

Processing:
```python
image_rel_paths = ['images/img1.png', 'images/img2.png']
image_paths = [
    '/path/to/dataset/images/img1.png',
    '/path/to/dataset/images/img2.png'
]
```

Model receives:
```python
messages = [{
    "role": "user",
    "content": [
        {"type": "image", "image": "/path/to/dataset/images/img1.png"},
        {"type": "image", "image": "/path/to/dataset/images/img2.png"},
        {"type": "text", "text": "Compare these images.\nOptions:\nA. same\nB. different"},
    ],
}]
```

## Result Format

The output JSON now includes:
```json
{
  "qid_001": {
    "qid": "qid_001",
    "question": "Is the dotted line a line of symmetry?",
    "choices": ["yes", "no"],
    "gt_answer": "\\boxed{A}",
    "gt_letter": "A",
    "predicted_letter": "A",
    "model_output": "The answer is A.",
    "is_correct": true,
    "category": "(GradeSchool) Geometric",
    "source": "Processed",
    "image_paths": ["images/ai2d-24027-0.png"],
    "num_images": 1
  }
}
```

## Data Types Handled

The script now properly handles:
- ✅ Numpy arrays: `array(['img.png'], dtype=object)`
- ✅ Lists: `['img.png']`
- ✅ Tuples: `('img.png',)`
- ✅ Single strings: `'img.png'`
- ✅ Multiple images: `['img1.png', 'img2.png', 'img3.png']`

## Testing

Test with a sample:
```python
import pandas as pd
import numpy as np

# Simulate the data format
data = {
    'qid': ['q1', 'q2', 'q3'],
    'image': [
        np.array(['images/img1.png'], dtype=object),  # Single image
        np.array(['images/img2.png', 'images/img3.png'], dtype=object),  # Multiple images
        ['images/img4.png'],  # Regular list
    ]
}

df = pd.DataFrame(data)

# Test extraction
for idx, row in df.iterrows():
    image_rel_paths = row['image']

    if hasattr(image_rel_paths, 'tolist'):
        image_rel_paths = image_rel_paths.tolist()
    elif not isinstance(image_rel_paths, (list, tuple)):
        image_rel_paths = [image_rel_paths]

    print(f"Row {idx}: {image_rel_paths}")
```

Expected output:
```
Row 0: ['images/img1.png']
Row 1: ['images/img2.png', 'images/img3.png']
Row 2: ['images/img4.png']
```

## Benefits

1. **Fixes image not found errors**: Properly extracts paths from numpy arrays
2. **Supports multiple images**: Handles questions with multiple images
3. **More robust**: Works with various data types (numpy, list, tuple, string)
4. **Better error messages**: Shows which specific images are missing
5. **Tracks image count**: Saves `num_images` in results for analysis

## Migration

No changes needed to existing scripts - the fix is backward compatible!

Just run:
```bash
sbatch run_qa.sh
```

The script will now properly handle all image formats.
