# Running Qwen3VL Captioning Scripts

This guide explains how to use the bash scripts for image captioning.

## Available Scripts

| Script | Description | Output File | Max Tokens |
|--------|-------------|-------------|------------|
| `caption_simple.sh` | Standard detailed captions | `captions_simple.json` | 256 |
| `caption_short.sh` | Short, concise captions | `captions_short.json` | 128 |
| `caption_long.sh` | Long, comprehensive captions | `captions_long.json` | 512 |
| `caption_all.sh` | Run all three styles sequentially | All three files | Varies |

## Quick Start

### 1. Submit a Single Job

To generate captions with a specific style:

```bash
# For SIMPLE captions
sbatch caption_simple.sh

# For SHORT captions
sbatch caption_short.sh

# For LONG captions
sbatch caption_long.sh

# For ALL styles (runs all three sequentially)
sbatch caption_all.sh
```

### 2. Monitor Job Progress

Check job status:
```bash
squeue -u $USER
```

View real-time output:
```bash
# Replace JOBID with your actual job ID
tail -f slurm-JOBID-simple.out
tail -f slurm-JOBID-short.out
tail -f slurm-JOBID-long.out
tail -f slurm-JOBID-all.out
```

### 3. Check Results

Once complete, check the output files:
```bash
ls -lh captions_*.json

# View sample of captions
head -20 captions_simple.json
```

## Customizing the Scripts

If you need to change parameters, edit the script before submitting:

### Change Dataset Path

Edit the script and modify:
```bash
DATASET_PATH="/your/custom/path/to/images"
```

### Change Output Location

Edit the script and modify:
```bash
OUTPUT_PATH="./your_custom_output.json"
```

### Change Model Path

To use a local model instead of downloading:
```bash
MODEL_PATH="/mnt/data-alpha-sg-02/team-agent/ai_glasses/models/Qwen3-VL-4B-Instruct"
```

### Change GPU/Memory Requirements

Edit the SBATCH parameters at the top of the script:
```bash
#SBATCH --gres=gpu:2          # Use 2 GPUs
#SBATCH --mem=64G             # Use 64GB memory
#SBATCH --time=3-00:00:00     # Set 3 day time limit
```

## Example Workflows

### Workflow 1: Generate Simple Captions
```bash
cd /mnt/data-alpha-sg-01/team-agent/home/y84401399/DOING_PROJECTS/long_video_understanding/CAPTIONING/Tiger-VL-Rethinker

# Submit job
sbatch caption_simple.sh

# Monitor progress
tail -f slurm-*-simple.out

# Check results when done
cat captions_simple.json | jq '.' | head -50
```

### Workflow 2: Generate All Styles
```bash
cd /mnt/data-alpha-sg-01/team-agent/home/y84401399/DOING_PROJECTS/long_video_understanding/CAPTIONING/Tiger-VL-Rethinker

# Submit job for all styles
sbatch caption_all.sh

# This will generate:
# - captions_simple.json
# - captions_short.json
# - captions_long.json
```

### Workflow 3: Resume Interrupted Job

If a job gets interrupted, simply resubmit:
```bash
# The script will automatically skip already-captioned images
sbatch caption_simple.sh
```

To force regeneration of all captions, edit the script and add `--overwrite`:
```bash
python caption_qwen3vl.py \
  --dataset-path "$DATASET_PATH" \
  --output-path "$OUTPUT_PATH" \
  --prompt-style "$PROMPT_STYLE" \
  --model-path "$MODEL_PATH" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --overwrite
```

### Workflow 4: Run Multiple Jobs in Parallel

Submit multiple jobs for different styles simultaneously:
```bash
# Submit all three at once
sbatch caption_simple.sh
sbatch caption_short.sh
sbatch caption_long.sh

# Check all running jobs
squeue -u $USER
```

## Troubleshooting

### Job Not Starting

If job stays in queue:
```bash
# Check partition availability
sinfo

# Check your job details
scontrol show job JOBID
```

### Out of Memory

If you get OOM errors, edit the script:
```bash
# Increase memory allocation
#SBATCH --mem=64G

# Or reduce max tokens
MAX_NEW_TOKENS=128
```

### Job Gets Cancelled

Check time limit:
```bash
# Increase time limit in script
#SBATCH --time=4-00:00:00  # 4 days
```

### Model Download Issues

If model download fails, use a local model:
```bash
# In the script, change:
MODEL_PATH="/path/to/local/model"
```

Or set cache directory:
```bash
# Add to script before python command:
export HF_HOME="/your/cache/directory"
```

### GPU Issues

Check GPU availability:
```bash
# View GPU partitions
sinfo -p gpu

# Request specific GPU type
#SBATCH --gres=gpu:a100:1
```

## Output Format

All scripts generate JSON files with this format:

```json
{
  "image001.jpg": "A scenic photograph showing...",
  "image002.png": "The image depicts a modern...",
  "image003.jpg": "This is a close-up view of..."
}
```

## Performance Notes

### Expected Processing Times

For ViRL39K dataset (~39,000 images):

| Style | Avg Time/Image | Total Time (1 GPU) |
|-------|----------------|-------------------|
| SHORT | ~3-5 seconds | ~40-55 hours |
| SIMPLE | ~5-8 seconds | ~55-90 hours |
| LONG | ~8-12 seconds | ~90-130 hours |

*Times are approximate and vary based on GPU model and image resolution*

### Optimization Tips

1. **Use multiple GPUs**: Edit `--gres=gpu:2` for 2x speedup
2. **Reduce max_tokens**: Lower values = faster generation
3. **Run styles in parallel**: Submit all three jobs at once
4. **Use local model**: Avoid download time on first run

## Monitoring Disk Space

Check available space:
```bash
df -h .
```

Check output file sizes:
```bash
du -sh captions_*.json
```

## Example Output

```bash
$ tail -f slurm-12345-simple.out

Using prompt style: SIMPLE
Prompt: Describe this image in detail.

Scanning directory: /mnt/data-alpha-sg-02/team-agent/ai_glasses/datasets/ViRL39K/images
Found 39000 images

Loading model from Qwen/Qwen3-VL-4B-Instruct...
Model loaded successfully!

Processing images...
Captioning:  25%|████▌              | 9750/39000 [13:45:22<41:16:03, 5.08s/it]
✓ Generated caption for img_9750.jpg: A modern office space with multiple desks, computers...
✓ Saved 9750 captions to ./captions_simple.json
```

## Getting Help

If you encounter issues:

1. Check the SLURM output file: `slurm-JOBID-*.out`
2. Check the error file: `slurm-JOBID-*.err`
3. Test with a small dataset first
4. Run `python test_caption.py` to verify setup

## Cancelling Jobs

To cancel a running job:
```bash
# Cancel specific job
scancel JOBID

# Cancel all your jobs
scancel -u $USER

# Cancel jobs by name
scancel --name=qwen3vl_simple
```

## Cleaning Up

Remove SLURM log files:
```bash
# Remove all SLURM logs
rm slurm-*.out slurm-*.err

# Keep only recent logs
ls -t slurm-*.out | tail -n +6 | xargs rm
```
