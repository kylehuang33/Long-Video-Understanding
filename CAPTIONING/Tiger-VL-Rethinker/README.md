# Qwen3VL Image Captioning

This directory contains scripts for captioning images using the Qwen3VL-4B-Instruct model.

## Files

- `caption_qwen3vl.py` - Main captioning script
- `run_caption.sh` - SLURM batch script for running captioning jobs
- `README.md` - This file

## Installation

Make sure you have the required dependencies installed:

```bash
pip install torch transformers qwen-vl-utils tqdm
```

## Usage

### Basic Usage

Caption all images in a directory:

```bash
python caption_qwen3vl.py \
  --dataset-path /path/to/images \
  --output-path ./captions.json \
  --prompt-style SIMPLE
```

### Prompt Styles

Three prompt styles are available:

- **SIMPLE**: "Describe this image in detail."
- **SHORT**: "Write a very short caption for the given image."
- **LONG**: "Write a very long and detailed caption describing the given image as comprehensively as possible."

### Command-Line Arguments

**Required:**
- `--dataset-path` - Path to directory containing images to caption
- `--output-path` - Path to save output JSON file with captions

**Optional:**
- `--prompt-style` - Style of prompt to use: SIMPLE, SHORT, or LONG (default: SIMPLE)
- `--model-path` - Path or HuggingFace model ID (default: Qwen/Qwen3-VL-4B-Instruct)
- `--max-new-tokens` - Maximum tokens to generate (default: 256)
- `--overwrite` - Overwrite existing captions (default: False)

### Examples

#### Example 1: Simple captions
```bash
python caption_qwen3vl.py \
  --dataset-path /mnt/data-alpha-sg-02/team-agent/ai_glasses/datasets/ViRL39K/images \
  --output-path ./captions_simple.json \
  --prompt-style SIMPLE
```

#### Example 2: Short captions
```bash
python caption_qwen3vl.py \
  --dataset-path /mnt/data-alpha-sg-02/team-agent/ai_glasses/datasets/ViRL39K/images \
  --output-path ./captions_short.json \
  --prompt-style SHORT
```

#### Example 3: Long detailed captions
```bash
python caption_qwen3vl.py \
  --dataset-path /mnt/data-alpha-sg-02/team-agent/ai_glasses/datasets/ViRL39K/images \
  --output-path ./captions_long.json \
  --prompt-style LONG \
  --max-new-tokens 512
```

#### Example 4: Using local model
```bash
python caption_qwen3vl.py \
  --dataset-path /path/to/images \
  --output-path ./captions.json \
  --prompt-style SIMPLE \
  --model-path /mnt/data-alpha-sg-02/team-agent/ai_glasses/models/Qwen3-VL-4B-Instruct
```

### Using SLURM

To run as a batch job:

1. Edit `run_caption.sh` to set your desired parameters:
   - `DATASET_PATH` - Path to your images
   - `OUTPUT_PATH` - Where to save captions
   - `PROMPT_STYLE` - Which prompt style to use

2. Submit the job:
```bash
sbatch run_caption.sh
```

3. Monitor the job:
```bash
# Check job status
squeue -u $USER

# View log output (replace JOBID with your job ID)
tail -f slurm-JOBID.out
```

## Output Format

The script generates a JSON file with the following format:

```json
{
  "image1.jpg": "A detailed caption describing the first image.",
  "image2.png": "A detailed caption describing the second image.",
  "image3.jpg": "A detailed caption describing the third image."
}
```

## Features

- **Incremental Saving**: Captions are saved after each image is processed, so progress is not lost if the script is interrupted
- **Resume Support**: By default, the script skips images that already have captions (use `--overwrite` to regenerate)
- **Progress Bar**: Shows real-time progress with `tqdm`
- **Error Handling**: Continues processing even if individual images fail
- **Supported Formats**: JPG, JPEG, PNG, BMP, GIF, WEBP, TIFF, TIF

## Performance Notes

- **GPU Required**: This script requires a GPU for efficient inference
- **Memory**: Approximately 16GB GPU memory is recommended for Qwen3VL-4B
- **Speed**: Typical processing time is 5-10 seconds per image depending on GPU

## Troubleshooting

### Out of Memory
If you get OOM errors, try:
- Using a GPU with more memory
- Reducing `--max-new-tokens`
- Processing images in smaller batches

### Model Not Found
If the model download fails:
- Check your internet connection
- Use a local model with `--model-path /path/to/local/model`
- Set HuggingFace cache: `export HF_HOME=/path/to/cache`

### Image Loading Errors
If specific images fail to load:
- Check image file integrity
- Ensure images are in supported formats
- Check file permissions

## Example SLURM Output

```
Using prompt style: SIMPLE
Prompt: Describe this image in detail.

Scanning directory: /mnt/data-alpha-sg-02/team-agent/ai_glasses/datasets/ViRL39K/images
Found 1000 images

Loading model from Qwen/Qwen3-VL-4B-Instruct...
Model loaded successfully!

Processing images...
Captioning: 100%|██████████| 1000/1000 [1:23:45<00:00, 5.03s/it]

============================================================
Captioning complete!
============================================================
Total images: 1000
Successfully captioned: 1000
Errors: 0
Results saved to: ./captions_simple.json
Total time: 5025.42s (83.76m)
Average time per image: 5.03s
```

## Citation

If you use Qwen3VL in your research, please cite:

```bibtex
@misc{qwen3vl,
  title={Qwen3-VL: Vision-Language Models},
  author={Qwen Team},
  year={2024},
  url={https://github.com/QwenLM/Qwen3-VL}
}
```
