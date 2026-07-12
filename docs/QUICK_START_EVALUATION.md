# Quick Start: Running Pedal-Aware Evaluation

## Prerequisites
- Conda or Miniconda available on Windows
- Conda environment `onsvel` created from the repository root `environment.yml`
- Trained model checkpoint in `out/model_snapshots/`, or a documented release/download checkpoint copied there
- Datasets in `datasets/` directory

> **Checkpoint policy:** `.torch` checkpoints are generated binary artifacts and are ignored by Git. Do not commit new checkpoints or move them to Git LFS for normal development. Share selected models as versioned release/download artifacts and document the URL, filename, expected local path, and checksum/metric metadata.

## Step 1: Create and Validate Environment
```bash
conda env create -f environment.yml
conda activate onsvel
python -m pytest tests -q
```

If the environment already exists, use `conda env update -f environment.yml --prune` instead of `conda env create -f environment.yml`.

For a clean reproducible reset, especially if you previously installed packages with pip inside `onsvel`, remove and recreate the environment:

```bash
conda env remove -n onsvel
conda env create -f environment.yml
conda activate onsvel
python -m pytest tests -q
```

Run the commands above from the repository root. `environment.yml` installs the project in editable mode (`-e .`) after the pinned Conda packages are present, so direct commands such as `python scripts/03_evaluate_pedal_model.py` can import `ov_piano`. If you only need to refresh that editable install later, run `python -m pip install --no-deps --no-build-isolation -e .` after activating `onsvel`. Avoid ad-hoc `pip install --upgrade ...` commands inside `onsvel`; use the clean reset commands above to return to the pinned environment.

## Step 2: Run Evaluation

Evaluation now has named presets so diagnostic runs are not confused with final metrics:

| Preset | Purpose | Validation split | Report as final? |
|--------|---------|------------------|------------------|
| `quick` | Smoke test that the checkpoint, data paths, and pedal metrics work | Highly shortened (`XV_TAKE_ONE_EVERY=50`) | **No** |
| `low_memory` | Default memory-safe run for 8GB-class GPUs | Shortened (`XV_TAKE_ONE_EVERY=20`) | **No** |
| `full` | Final benchmark/reporting run | Full validation split (`XV_TAKE_ONE_EVERY=1`) | **Yes** |

```bash
# Fast smoke test only. Do not report these metrics as final.
python scripts/03_evaluate_pedal_model.py EVALUATION_PRESET=quick

# Memory-safe diagnostic/default run. Do not report these metrics as final.
python scripts/03_evaluate_pedal_model.py EVALUATION_PRESET=low_memory

# Final/reportable metrics: uses the full validation split for threshold search.
python scripts/03_evaluate_pedal_model.py EVALUATION_PRESET=full

# Or specify a specific checkpoint:
python scripts/03_evaluate_pedal_model.py EVALUATION_PRESET=full SNAPSHOT_INPATH="out/model_snapshots/YOUR_MODEL.torch"
```

> **Important:** `quick` and `low_memory` still evaluate the test split, but their thresholds are selected from a shortened validation split. Treat those results as diagnostic only. Rerun with `EVALUATION_PRESET=full` before publishing or comparing final metrics.

### Resuming Interrupted Evaluation

Evaluation checkpoints are enabled by default. The scripts save resumable stage
checkpoints under `out/eval_checkpoints/` after each successfully processed file
or completed grid-search combination:

- `03_evaluate_pedal_model.py` checkpoints validation inference, note grid search,
  pedal grid search, and final test metrics.
- `04_evaluate_test_split.py` checkpoints test inference and test grid summaries.

If a run is interrupted, rerun the same command and matching checkpoints will be
loaded automatically. Checkpoints are fingerprinted from the model snapshot, data
paths, split/file list, decoder settings, thresholds, and tolerance settings, so
changing evaluation-relevant options creates a separate checkpoint file instead
of mixing stale results.

Useful overrides:

```bash
# Disable evaluation resume/checkpointing for one run
python scripts/03_evaluate_pedal_model.py EVALUATION_CHECKPOINTS_ENABLED=false

# Force recomputation for the current command/fingerprint
python scripts/03_evaluate_pedal_model.py RESET_EVALUATION_CHECKPOINTS=true

# Store checkpoint files somewhere else
python scripts/03_evaluate_pedal_model.py EVALUATION_CHECKPOINT_DIR="out/eval_checkpoints_full_run"
```

## Step 3: Check Results
Results will be printed to console and saved to `out/txt_logs/`

Expected output format:
```
XV HYPERPARAMETER SEARCH:
Summary (without velocity):
   threshold   shift      P      R      F1
0       0.70  -0.01  0.xxx  0.xxx  0.xxx
...

TEST RESULTS:
ONSETS:
                           P      R      F1
2004/...                0.xxx  0.xxx  0.xxx
...
AVERAGES (t=0.74, s=-0.01)  0.xxx  0.xxx  0.xxx

ONSETS+VELOCITIES:
[Similar format]
```

## Troubleshooting

### Error: "ModuleNotFoundError: No module named 'omegaconf'"
**Solution:** Make sure you activated the correct conda environment:
```bash
conda activate onsvel
```

### Error: "No module named 'pyaudioop'" when reading audio
**Cause:** You are likely running the app or CLI with Python 3.13+ outside the supported `onsvel` Conda environment. Python 3.13 removed the stdlib `audioop` module that `pydub` expects.

**Preferred solution:** Use the project environment:
```bash
conda activate onsvel
```

**Pip-only workaround:** Install the compatibility package, then make sure `ffmpeg` is also installed and available on `PATH` for MP3/non-WAV decoding:
```bash
python -m pip install audioop-lts
```

### Error: "ValueError: too many values to unpack"
**Solution:** This was the main issue - FIXED in [03_evaluate_pedal_model.py](03_evaluate_pedal_model.py:218)

### Error: "FileNotFoundError" for model checkpoint
**Solution:** Check that your model file exists:
```bash
ls out/model_snapshots/
```

### Out of Memory (OOM)
**Solution:** Start from the `low_memory` preset, then reduce chunk size if needed:
```bash
python scripts/03_evaluate_pedal_model.py EVALUATION_PRESET=low_memory INFERENCE_CHUNK_SIZE=30.0
```

## Configuration Options

You can override any configuration parameter via command line:

```bash
# Select a named preset
python scripts/03_evaluate_pedal_model.py EVALUATION_PRESET=quick
python scripts/03_evaluate_pedal_model.py EVALUATION_PRESET=low_memory
python scripts/03_evaluate_pedal_model.py EVALUATION_PRESET=full

# Use smaller chunks (for limited memory)
python scripts/03_evaluate_pedal_model.py INFERENCE_CHUNK_SIZE=100

# Test fewer thresholds (faster evaluation)
python scripts/03_evaluate_pedal_model.py SEARCH_THRESHOLDS="[0.7, 0.75, 0.8]"

# Use CPU instead of GPU
python scripts/03_evaluate_pedal_model.py DEVICE="cpu"

# Process only every Nth validation file (faster)
python scripts/03_evaluate_pedal_model.py XV_TAKE_ONE_EVERY=10
```

If `XV_TAKE_ONE_EVERY` is not `1`, the validation search is shortened and the script will warn that the resulting validation-selected thresholds and test metrics are not final/reportable.

## Alternative Evaluation Scripts

### 1. Memory-Efficient Evaluation
```bash
python simple_eval.py
```
- Uses smaller chunks (60s)
- Fewer threshold tests
- Better for systems with limited memory

### 2. Pedal-Specific Evaluation
```bash
python eval_with_pedals.py
```
- Evaluates sustain pedal detection
- Reports pedal onset/offset metrics
- Includes onset/velocity evaluation

## Understanding the Results

### Precision (P)
- Percentage of predicted notes that are correct
- Higher is better (fewer false alarms)

### Recall (R)
- Percentage of actual notes that were detected
- Higher is better (fewer missed notes)

### F1 Score
- Harmonic mean of Precision and Recall
- Balanced measure of overall performance
- **This is the primary metric to optimize**

### Threshold (t)
- Probability threshold for onset detection
- Lower threshold → more detections (higher recall, lower precision)
- Higher threshold → fewer detections (higher precision, lower recall)
- Optimal threshold found via cross-validation. Use `EVALUATION_PRESET=full` when the threshold will be used for final metrics.

### Shift (s)
- Time offset applied to predictions (in seconds)
- Accounts for systematic timing bias in the model
- Typically small (-0.01 to 0.01 seconds)

## Expected Performance

Based on the original paper (without pedal):
- **Onsets F1:** ~0.967 (96.7%)
- **Onsets+Velocities F1:** ~0.945 (94.5%)

Your model (with pedal support) may have slightly different metrics depending on training configuration.

## Next Steps After Evaluation

1. **Analyze training logs:**
   ```bash
python scripts/05_analyze_training_logs.py LOG_PATH="out/txt_logs/YOUR_LOG.json"
   ```

2. **Generate visualization plots:**
   ```bash
python scripts/06_visualize_pedal_predictions.py SNAPSHOT_INPATH="out/model_snapshots/YOUR_MODEL.torch"
   ```

3. **Continue training:**
   ```bash
   python scripts/02_train_pedal_model.py
   # Will auto-resume from latest checkpoint
   ```

---

**All fixes have been applied. Evaluation should now run without errors.** ✅
