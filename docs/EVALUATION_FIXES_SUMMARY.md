# Evaluation Notes for the Pedal-Aware Model

This note records the main changes made while adapting the original onset/velocity evaluation workflow to the pedal-aware model.

## Model output handling

The original evaluation path expected two outputs from the model: onsets and velocities. The pedal-aware model returns three outputs: onsets, velocities, and sustain-pedal probabilities.

Previous two-output assumption:

```python
probs, vels = model(x)
```

Current pedal-aware unpacking:

```python
probs, vels, pedals = model(x)
```

The maintained evaluation entry point is:

```bash
python scripts/03_evaluate_pedal_model.py
```

## Strided inference shape checks

`ov_piano/inference.py` now validates the batch and time dimensions for every model output, not only for the note outputs. This keeps the same chunking code usable for both two-output and three-output models.

The important check is:

```python
assert all(o.shape[0] == chunk.shape[0] for o in outputs)
assert all(o.shape[-1] == chunk.shape[-1] for o in outputs)
```

## Model outputs

`OnsetsAndVelocities` now returns:

1. Onset probabilities with shape `(batch, 88, time)`
2. Velocity predictions with shape `(batch, 88, time)`
3. Sustain-pedal probabilities with shape `(batch, 1, time)`

The pedal output is decoded separately from the note events.

## Data layout used for training

The piano-roll HDF5 file stores note and pedal information in this order:

```text
[0:128]     onsets for MIDI notes
[128:256]   note frames for MIDI notes
[256]       sustain pedal
[257]       soft pedal
[258]       sostenuto/tenuto pedal where available
```

The current training code uses the sustain-pedal row as the pedal target.

## Evaluation presets

The main evaluation script has three presets:

| Preset | Intended use | Final/reportable? |
|--------|--------------|-------------------|
| `quick` | Fast path check | No |
| `low_memory` | Diagnostic run for limited memory | No |
| `full` | Full validation threshold search and final metrics | Yes |

Use this command for final metrics:

```bash
python scripts/03_evaluate_pedal_model.py EVALUATION_PRESET=full SNAPSHOT_INPATH="out/model_snapshots/YOUR_MODEL.torch"
```

## Memory-related safeguards

The evaluation code includes safeguards for large prediction sets:

- chunked inference for long audio files;
- configurable `MAX_PREDICTIONS_PER_FILE`;
- logging of ground-truth and prediction counts;
- cleanup between files;
- resumable evaluation checkpoints under `out/eval_checkpoints/`.

These checks are especially useful on 8 GB GPUs and Windows systems where long validation files can otherwise cause out-of-memory errors.

## Current workflow

1. Train or copy a checkpoint into `out/model_snapshots/`.
2. Run a quick diagnostic:

   ```bash
   python scripts/03_evaluate_pedal_model.py EVALUATION_PRESET=quick
   ```

3. Run final evaluation:

   ```bash
   python scripts/03_evaluate_pedal_model.py EVALUATION_PRESET=full SNAPSHOT_INPATH="out/model_snapshots/YOUR_MODEL.torch"
   ```

4. Review logs in `out/txt_logs/` and generated plots under `out/`.

## Files most relevant to evaluation

- `scripts/03_evaluate_pedal_model.py` - main pedal-aware evaluation script
- `scripts/04_evaluate_test_split.py` - test-split helper
- `ov_piano/inference.py` - strided inference and decoders
- `ov_piano/eval.py` - metric helpers and ground-truth loading
- `ov_piano/models/ov.py` - model forward pass
