# Evaluation Memory Notes

This note explains an out-of-memory problem observed during evaluation and the safeguards used in the current scripts.

## Observed error

```text
numpy.core._exceptions._ArrayMemoryError: Unable to allocate 10.0 GiB for an array
with shape (4019, 334343) and data type float64
```

## Cause

The evaluation step compares predicted note intervals against ground-truth intervals. In the failing case, one file produced 334,343 predictions, which is far above a normal note count. `mir_eval` then attempted to compare every prediction with every ground-truth note:

```text
4,019 ground-truth notes x 334,343 predictions = about 1.3 billion comparisons
```

At float64 precision, that comparison matrix requires roughly 10 GB of memory.

Likely causes for this situation include:

- threshold set too low;
- a weak or unfinished checkpoint producing noisy outputs;
- an unusually long or problematic file;
- duplicate detections from decoder settings.

## Safeguards in the evaluation scripts

### Higher diagnostic thresholds

Diagnostic runs can use a higher threshold to reduce low-confidence predictions:

```bash
python scripts/03_evaluate_pedal_model.py EVALUATION_PRESET=low_memory SEARCH_THRESHOLDS="[0.80]"
```

### Prediction count limit

The scripts support `MAX_PREDICTIONS_PER_FILE`. Files above the limit are skipped before calling the metric code.

```bash
python scripts/03_evaluate_pedal_model.py EVALUATION_PRESET=low_memory MAX_PREDICTIONS_PER_FILE=50000
```

A rough guide:

| Prediction count | Meaning | Action |
|------------------|---------|--------|
| 1,000-5,000 | Normal for many pieces | Process normally |
| 5,000-20,000 | High but possible for long/complex pieces | Check logs |
| 20,000-50,000 | Very high | Use caution |
| Above 50,000 | Excessive for this setup | Skip or inspect manually |

### Logging before metric computation

The evaluation logs both ground-truth and prediction counts. This makes it easier to identify files that would create very large comparison matrices.

Example:

```text
GT: 2,847 notes, Pred: 3,102 notes
```

For an excessive case:

```text
SKIPPING file.midi: too many predictions (334,343) vs 4,019 ground truth
```

### Cleanup between files

The scripts clear temporary variables and GPU cache between files where possible. This does not solve excessive prediction counts by itself, but it helps keep long evaluation runs stable.

## Recommended workflow

Start with a diagnostic run:

```bash
python scripts/03_evaluate_pedal_model.py EVALUATION_PRESET=low_memory
```

If it completes and the prediction counts look reasonable, run the full preset for reportable metrics:

```bash
python scripts/03_evaluate_pedal_model.py EVALUATION_PRESET=full
```

`quick` and `low_memory` shorten the validation split for speed or memory safety. Final metrics should come from `EVALUATION_PRESET=full`, or another run where `XV_TAKE_ONE_EVERY=1`.

## If memory problems continue

Try the following in order:

```bash
# Raise the threshold for a diagnostic run
python scripts/03_evaluate_pedal_model.py EVALUATION_PRESET=low_memory SEARCH_THRESHOLDS="[0.85]"

# Use the shortest preset to check paths and checkpoint loading
python scripts/03_evaluate_pedal_model.py EVALUATION_PRESET=quick

# Lower the prediction safety limit
python scripts/03_evaluate_pedal_model.py EVALUATION_PRESET=low_memory MAX_PREDICTIONS_PER_FILE=20000

# Reduce inference chunk size
python scripts/03_evaluate_pedal_model.py EVALUATION_PRESET=low_memory INFERENCE_CHUNK_SIZE=30.0
```

If a specific file repeatedly produces excessive predictions, inspect that file separately before including it in final evaluation.
