# Quick Start: Pedal-Aware Evaluation

This guide explains how to run the maintained evaluation scripts for a trained pedal-aware checkpoint.

## Prerequisites

- Conda or Miniconda on Windows
- The `onsvel` environment created from `environment.yml`
- MAESTRO/MAPS data prepared under `datasets/`
- A checkpoint copied to `out/model_snapshots/`

`.torch` checkpoints are local binary artifacts and are ignored by Git. If a selected model needs to be shared, keep it as a separate release/download artifact and record the filename, expected local path, metric summary, and checksum if available.

## Step 1: create or refresh the environment

```bash
conda env create -f environment.yml
conda activate onsvel
python -m pytest tests -q
```

If the environment already exists:

```bash
conda env update -f environment.yml --prune
conda activate onsvel
python -m pytest tests -q
```

For a clean reset after local pip experiments:

```bash
conda env remove -n onsvel
conda env create -f environment.yml
conda activate onsvel
python -m pytest tests -q
```

`environment.yml` installs this project in editable mode, so commands such as `python scripts/03_evaluate_pedal_model.py` can import `ov_piano` from the repository root.

## Step 2: choose an evaluation preset

| Preset | Purpose | Validation split | Use for final report? |
|--------|---------|------------------|-----------------------|
| `quick` | Smoke test for checkpoint and data paths | Highly shortened (`XV_TAKE_ONE_EVERY=50`) | No |
| `low_memory` | Safer run for limited GPU memory | Shortened (`XV_TAKE_ONE_EVERY=20`) | No |
| `full` | Final benchmark run | Full validation split (`XV_TAKE_ONE_EVERY=1`) | Yes |

```bash
# Fast smoke test only
python scripts/03_evaluate_pedal_model.py EVALUATION_PRESET=quick

# Memory-safer diagnostic run
python scripts/03_evaluate_pedal_model.py EVALUATION_PRESET=low_memory

# Final metrics using the full validation split
python scripts/03_evaluate_pedal_model.py EVALUATION_PRESET=full

# Final metrics for a specific checkpoint
python scripts/03_evaluate_pedal_model.py EVALUATION_PRESET=full SNAPSHOT_INPATH="out/model_snapshots/YOUR_MODEL.torch"
```

The `quick` and `low_memory` presets still run test evaluation, but their thresholds are selected from shortened validation splits. Use `EVALUATION_PRESET=full` before reporting final metrics.

## Resuming interrupted evaluation

Evaluation resume files are stored under `out/eval_checkpoints/`. They are fingerprinted from the checkpoint, data paths, split list, decoder settings, thresholds, and tolerance settings.

Useful overrides:

```bash
# Disable evaluation resume/checkpointing for one run
python scripts/03_evaluate_pedal_model.py EVALUATION_CHECKPOINTS_ENABLED=false

# Recompute the current command/fingerprint
python scripts/03_evaluate_pedal_model.py RESET_EVALUATION_CHECKPOINTS=true

# Store checkpoint files somewhere else
python scripts/03_evaluate_pedal_model.py EVALUATION_CHECKPOINT_DIR="out/eval_checkpoints_full_run"
```

## Step 3: check the output

Results are printed to the console and written to `out/txt_logs/`.

Expected format:

```text
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
```

## Troubleshooting

### `ModuleNotFoundError: No module named 'omegaconf'`

Activate the project environment:

```bash
conda activate onsvel
```

### `No module named 'pyaudioop'` when reading audio

This usually means the app or CLI is running on Python 3.13+ outside the supported Conda environment. Use:

```bash
conda activate onsvel
```

For pip-only setups, install the compatibility package and make sure `ffmpeg` is available on `PATH`:

```bash
python -m pip install audioop-lts
```

### `ValueError: too many values to unpack`

Use the current evaluation script. The pedal-aware model returns onset, velocity, and pedal outputs, so older two-output evaluation code will fail.

### `FileNotFoundError` for a checkpoint

Check that the checkpoint exists:

```bash
ls out/model_snapshots/
```

On Windows cmd:

```cmd
dir out\model_snapshots
```

### Out of memory

Start with the lower-memory preset, then reduce chunk size if required:

```bash
python scripts/03_evaluate_pedal_model.py EVALUATION_PRESET=low_memory INFERENCE_CHUNK_SIZE=30.0
```

## Common overrides

```bash
# Smaller chunks for limited memory
python scripts/03_evaluate_pedal_model.py INFERENCE_CHUNK_SIZE=100

# Fewer thresholds for a faster diagnostic run
python scripts/03_evaluate_pedal_model.py SEARCH_THRESHOLDS="[0.7, 0.75, 0.8]"

# CPU evaluation
python scripts/03_evaluate_pedal_model.py DEVICE="cpu"

# Process every Nth validation file only; not final/reportable
python scripts/03_evaluate_pedal_model.py XV_TAKE_ONE_EVERY=10
```

If `XV_TAKE_ONE_EVERY` is not `1`, the validation search is shortened and the script will warn that the metrics are diagnostic only.

## Maintained evaluation scripts

```bash
# Main pedal-aware evaluation
python scripts/03_evaluate_pedal_model.py EVALUATION_PRESET=full

# Test-split evaluation helper
python scripts/04_evaluate_test_split.py
```

## Reading the metrics

- Precision: the proportion of predicted notes/events that are correct.
- Recall: the proportion of ground-truth notes/events that were detected.
- F1 score: harmonic mean of precision and recall.
- Threshold: probability cutoff used by the decoder.
- Shift: small timing offset applied when matching predictions to ground truth.

For final reporting, use the full preset so thresholds are selected using the full validation split.

## After evaluation

```bash
# Analyze training logs
python scripts/05_analyze_training_logs.py LOG_PATH="out/txt_logs/YOUR_LOG.json"

# Generate a qualitative plot
python scripts/06_visualize_pedal_predictions.py SNAPSHOT_INPATH="out/model_snapshots/YOUR_MODEL.torch"
```
