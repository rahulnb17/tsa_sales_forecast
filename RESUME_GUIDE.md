# Checkpoint & Resume Guide

## Overview

The training system now supports **checkpoint/resume functionality**. If training is interrupted (wifi disconnects, computer crashes, etc.), you can resume from where it stopped instead of starting over.

## How It Works

### Automatic Checkpoint Saving
- After **each state completes**, a checkpoint is saved to `results/training_checkpoint.json`
- Contains: completed states list, results, best model info
- **Incremental saves** - no need to wait for all states to finish

### Automatic Resume
- When you run `python train.py` again, it automatically:
  1. Checks for existing checkpoint
  2. Loads completed states
  3. **Skips already completed states**
  4. Continues with remaining states

## Usage

### Normal Training (with resume enabled by default)
```bash
python train.py
```

If interrupted, just run the same command again:
```bash
python train.py
```

It will automatically resume from where it stopped!

### Start Fresh (disable resume)
If you want to retrain everything from scratch:
```bash
python train.py --no-resume
```

This will overwrite existing models and start fresh.

## Example Scenario

**Day 1:**
```bash
python train.py
# Processes 20 states, then wifi disconnects
```

**Day 2:**
```bash
python train.py
# Automatically detects 20 states completed
# Resumes from state 21
# Only processes remaining 23 states
```

## Checkpoint File

Location: `results/training_checkpoint.json`

Contains:
- `completed_states`: List of states already trained
- `results`: Evaluation results for completed states
- `best_models`: Best model info for each state

**Note**: Model files are saved separately in `models/` directory, so even if checkpoint is lost, you can still see which models exist.

## Benefits

✅ **No lost progress** - Each state saves immediately  
✅ **Resume anytime** - Just run the same command  
✅ **Safe interruptions** - Wifi drops, crashes, etc. won't lose work  
✅ **Incremental results** - Check progress anytime by looking at checkpoint

## Troubleshooting

**Q: Checkpoint file corrupted?**  
A: Delete `results/training_checkpoint.json` and run with `--no-resume` to start fresh

**Q: Want to retrain specific states?**  
A: Delete their model files from `models/` directory, then run normally (they'll be retrained)

**Q: Checkpoint says state completed but model file missing?**  
A: The system will try to reload, but if file is missing, it will retrain that state

## Best Practices

1. **Let it run** - Even if interrupted, progress is saved
2. **Check checkpoint** - Look at `results/training_checkpoint.json` to see progress
3. **Don't delete models** - Model files in `models/` are needed for resume
4. **Use `--no-resume` sparingly** - Only if you want to completely retrain

