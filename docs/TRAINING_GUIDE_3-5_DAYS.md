# Training Guide: Pedal Head Fine-Tuning

This guide describes the training workflow used for the sustain-pedal extension. The starting point is the existing onset/velocity checkpoint. During the recommended run, the note backbone is kept frozen and only the pedal head is trained.

## Training objective

The training setup is meant to:

- keep onset and velocity performance close to the loaded note checkpoint;
- learn a usable sustain-pedal output from the MAESTRO pedal annotations;
- avoid updating shared note layers during pedal-specialized training;
- keep memory use within an 8 GB GPU / 16 GB RAM workstation.

## Recommended command

Run this from the repository root:

```bash
conda activate onsvel

# Optional: remove the local resume pointer before starting a fresh pedal run.
# On Windows cmd, use: del out\model_snapshots\.resume_state.json
rm -f out/model_snapshots/.resume_state.json

python scripts/02_train_pedal_model.py \
  PEDAL_ONLY_FINETUNE=true \
  RESUME_FROM_LATEST=false \
  PEDAL_LR_MAX=0.0003 \
  TRAIN_BATCH_SECS=4.0 \
  NUM_EPOCHS=8
```

In pedal-only mode:

- `specnorm`, `stem`, `onset_stages`, and `velocity_stage` are frozen;
- pedal features are detached before the pedal loss is applied;
- onset/velocity losses are logged under `note_monitor_losses` for checking only;
- the optimizer uses `PEDAL_LR_MAX`, which is smaller than the full-model learning rate;
- training stops early if no valid note checkpoint is available, because freezing an untrained note model would not be useful.

Avoid `RESUME_FROM_LATEST=true` unless the latest checkpoint was produced by the same pedal-only workflow. If an older experimental checkpoint has lower note F1, start from the bundled baseline by leaving `SNAPSHOT_INPATH` unset and using `RESUME_FROM_LATEST=false`.

## Important configuration values

| Parameter | Earlier test value | Current value | Reason |
|-----------|-------------------|---------------|--------|
| `TRAIN_BATCH_SECS` | 0.05s | 4.0s or longer | Gives the pedal head more musical context |
| `NUM_EPOCHS` | 2 | 8 | Allows more passes over the training data |
| `ONSET_POSITIVES_WEIGHT` | 2.0 | 3.0 | Keeps sparse onset monitoring comparable |
| `PEDAL_LOSS_LAMBDA` | 0.5 | 1.0 | Gives pedal learning enough weight |
| `PEDAL_POSITIVES_WEIGHT` | 2.0 | 2.0 | Keeps pedal-active frames balanced |
| `PEDAL_LR_MAX` | N/A | 0.0003 | Stable pedal-head fine-tuning |
| `TRAINABLE_ONSETS` | `true` | `false` | Protects the note layers during pedal training |
| `TRAIN_LOG_EVERY` | 10 | 50 | Reduces logging overhead |

## Approximate training time

The estimate below is based on the local RTX 2070 SUPER 8 GB / 16 GB RAM setup used during testing.

- MAESTRO training set: about 1,200 files
- 4-second chunks: roughly 90,000 chunks per epoch
- Effective batch size: 1 or 2 physical samples with gradient accumulation up to 16
- Expected speed: around 1.5-2 steps per second, depending on disk and GPU load
- 8 epochs: usually around 10-12 hours, with extra time for checkpointing and logging

For planning, allow 1-2 days so there is time to evaluate the checkpoint and rerun if needed.

## Monitoring training

A typical console section looks like this:

```text
[TRAIN] epoch: 1, step: 50, global_step: 50
  losses: {pedal: 0.18}
  note_monitor_losses: {vel: ..., ons: ...}
  loss_mode: pedal_only_note_losses_are_monitoring_only
  LR: 0.00027
```

During a healthy pedal-only run:

- pedal loss should gradually decrease;
- onset/velocity monitor losses should stay roughly stable because those layers are frozen;
- a sudden drop in note F1 after this workflow usually points to checkpoint selection or thresholding, not to updated note weights.

If `note_monitor_losses.ons` is very high from the start, check that the baseline note checkpoint loaded correctly. The default baseline path is:

```text
out/model_snapshots/OnsetsAndVelocities_2023_03_04_09_53_53.289step=43500_f1=0.9675__0.9480.torch
```

## Why the settings were chosen

### Longer chunks

Very short chunks, such as 50 ms, do not contain enough context for sustain-pedal changes. A 4-second chunk normally includes several note onsets and at least some local phrasing information. This is more useful for learning pedal state transitions.

### More epochs

Two epochs were enough for early checks but not enough for the pedal head to see the data repeatedly. Eight epochs is a practical middle ground for the available hardware.

### Frozen note backbone

The pedal objective is added on top of a checkpoint that already performs well for note detection. Freezing the note backbone keeps the training target narrow: preserve note transcription while learning sustain-pedal output.

The main settings are:

- `PEDAL_ONLY_FINETUNE=true` freezes note-related layers;
- `DETACH_PEDAL_FEATURES=true` prevents pedal gradients from changing shared features;
- `PEDAL_LR_MAX=0.0003` keeps updates small;
- `MAX_GRAD_NORM=1.0` clips large gradients near pedal transitions.

## Checkpoints

The training script saves checkpoints under:

```text
out/model_snapshots/
```

Use the latest checkpoint only if it came from the pedal-only run you intend to evaluate. If the folder contains mixed experiments, pass the exact checkpoint path to the evaluation command.

Checkpoint policy:

- `.torch` checkpoints are local binary artifacts and are ignored by Git;
- do not commit new checkpoints directly;
- share selected models separately as release/download artifacts with filename, path, metric, and checksum notes;
- keep local training outputs under `out/model_snapshots/`.

## Resume after interruption

Automatic resume is only used when explicitly enabled:

```bash
python scripts/02_train_pedal_model.py RESUME_FROM_LATEST=true PEDAL_ONLY_FINETUNE=true
```

The resume pointer is stored at:

```text
out/model_snapshots/.resume_state.json
```

Before using it, confirm that the checkpoint listed there belongs to the intended pedal-only run.

## After training

Evaluate the selected checkpoint:

```bash
python scripts/03_evaluate_pedal_model.py EVALUATION_PRESET=full SNAPSHOT_INPATH="out/model_snapshots/YOUR_MODEL.torch"
```

Generate a qualitative plot if needed:

```bash
python scripts/06_visualize_pedal_predictions.py SNAPSHOT_INPATH="out/model_snapshots/YOUR_MODEL.torch"
```

Inspect logs:

```bash
python scripts/05_analyze_training_logs.py LOG_PATH="out/txt_logs/YOUR_LOG.json"
```

## Memory notes

With the settings above, expected resource use is approximately:

- VRAM: 6-7 GB;
- RAM: 10-12 GB;
- disk: depends on checkpoint frequency, but several GB should be kept free.

If memory is tight, reduce `TRAIN_BATCH_SECS` slightly or keep `DATALOADER_WORKERS=0` on Windows.
