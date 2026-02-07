# Quick Start: Running Evaluation

## Prerequisites
- Conda environment `onsvel` activated
- Trained model checkpoint in `out/model_snapshots/`
- Datasets in `datasets/` directory

## Step 1: Activate Environment
```bash
conda activate onsvel
```

## Step 2: Run Evaluation
```bash
# Using the default checkpoint (latest in out/model_snapshots/)
python 2_eval_onsets_velocities.py

# Or specify a specific checkpoint:
python 2_eval_onsets_velocities.py SNAPSHOT_INPATH="out/model_snapshots/YOUR_MODEL.torch"
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

### Error: "ValueError: too many values to unpack"
**Solution:** This was the main issue - FIXED in [2_eval_onsets_velocities.py](2_eval_onsets_velocities.py:218)

### Error: "FileNotFoundError" for model checkpoint
**Solution:** Check that your model file exists:
```bash
ls out/model_snapshots/
```

### Out of Memory (OOM)
**Solution:** Reduce chunk size in evaluation:
```bash
python 2_eval_onsets_velocities.py INFERENCE_CHUNK_SIZE=60.0
```

## Configuration Options

You can override any configuration parameter via command line:

```bash
# Use smaller chunks (for limited memory)
python 2_eval_onsets_velocities.py INFERENCE_CHUNK_SIZE=100

# Test fewer thresholds (faster evaluation)
python 2_eval_onsets_velocities.py SEARCH_THRESHOLDS="[0.7, 0.75, 0.8]"

# Use CPU instead of GPU
python 2_eval_onsets_velocities.py DEVICE="cpu"

# Process only every Nth validation file (faster)
python 2_eval_onsets_velocities.py XV_TAKE_ONE_EVERY=10
```

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
- Optimal threshold found via cross-validation

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
   python 3_analyze_logs.py LOG_PATH="out/txt_logs/YOUR_LOG.json"
   ```

2. **Generate visualization plots:**
   ```bash
   python 4_qualitative_plots.py SNAPSHOT_INPATH="out/model_snapshots/YOUR_MODEL.torch"
   ```

3. **Continue training:**
   ```bash
   python 1_train_onsets_velocities.py
   # Will auto-resume from latest checkpoint
   ```

---

**All fixes have been applied. Evaluation should now run without errors.** ✅
