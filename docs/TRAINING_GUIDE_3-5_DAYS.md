# Training Guide: Pedal-Specialized Training Without Hurting Note F1

This project now treats sustain-pedal learning as a **pedal-head fine-tuning**
task by default. The bundled note checkpoint already has high onset/velocity
performance, so the safest workflow is to freeze the note backbone and train
only the newly added pedal head.

## Recommended Command

From the repository root:

```bash
conda activate onsvel

# Optional but recommended if recent experimental checkpoints degraded note F1.
# On Windows cmd, use: del out\model_snapshots\.resume_state.json
rm -f out/model_snapshots/.resume_state.json

python scripts/02_train_pedal_model.py \
  PEDAL_ONLY_FINETUNE=true \
  RESUME_FROM_LATEST=false \
  PEDAL_LR_MAX=0.0003 \
  TRAIN_BATCH_SECS=4.0 \
  NUM_EPOCHS=8
```

Expected safety properties:

- `specnorm`, `stem`, `onset_stages`, and `velocity_stage` are frozen.
- Pedal features are detached before the pedal loss.
- Onset/velocity losses appear under `note_monitor_losses`, not optimized `losses`, in pedal-only mode.
- The optimizer uses `PEDAL_LR_MAX`, not the larger full-model `LR_MAX`.
- The script refuses to run pedal-only fine-tuning without a valid note checkpoint; this prevents freezing a random note model.

Avoid `RESUME_FROM_LATEST=true` unless you are certain the latest checkpoint was
also produced by the safe pedal-only workflow. If recent agentic/code-assisted
runs lowered note F1, start again from the bundled baseline by leaving
`SNAPSHOT_INPATH` unset and `RESUME_FROM_LATEST=false`.

## Configuration Changes

### ✅ **What Changed (Prototype → Production)**

| Parameter | Old (Broken) | New (Optimized) | Why? |
|-----------|--------------|-----------------|------|
| `TRAIN_BATCH_SECS` | **0.05s** (50ms) | **4.0s** | Model needs sufficient context to learn patterns |
| `NUM_EPOCHS` | **2** | **8** | More epochs = better learning |
| `ONSET_POSITIVES_WEIGHT` | 2.0 | **3.0** | Onsets are sparse, need higher weight |
| `PEDAL_LOSS_LAMBDA` | 0.5 | **1.0** | Pedal detection is important |
| `PEDAL_POSITIVES_WEIGHT` | 2.0 | **2.0** | Pedal-active frames are weighted without overpowering training |
| `PEDAL_LR_MAX` | N/A | **0.0003** | Stable pedal-head-only fine-tuning |
| `TRAINABLE_ONSETS` | `true` | **`false`** | Do not update note layers during pedal specialization |
| `TRAIN_LOG_EVERY` | 10 | **50** | Reduce logging overhead |

---

## 📊 **Training Time Estimate**

### Hardware: RTX 2070 SUPER (8GB VRAM), 16GB RAM

**Calculations:**
- MAESTRO training set: ~1,200 files × 5 min avg = 6,000 minutes of audio
- 4-second chunks: ~90,000 chunks per epoch
- Effective batch size: 2 × 8 (gradient accum) = 16
- Steps per epoch: 90,000 ÷ 16 = **5,625 steps**
- Training speed: ~1.5-2 steps/second (realistic estimate)
- Time per epoch: 5,625 ÷ 1.75 ≈ **3,200 seconds = ~53 minutes**
- **8 epochs: ~7 hours**

**Conservative estimate with overhead: 10-12 hours**

### Timeline:
- ✅ **Best case:** 10-12 hours (< 1 day)
- ✅ **Typical:** 1-2 days
- ✅ **Conservative:** 3 days maximum

**Result: Well within your 3-5 day target!** 🎉

---

## 🚀 **How to Start Training**

### Step 1: Clean Start (Recommended)
```bash
# Remove resume state so the run starts from the bundled high-F1 note baseline.
rm -f out/model_snapshots/.resume_state.json

# Activate environment
conda activate onsvel

# Start safe pedal-only training
python scripts/02_train_pedal_model.py RESUME_FROM_LATEST=false PEDAL_ONLY_FINETUNE=true
```

### Step 2: Monitor Progress
Watch the console output:
```
[TRAIN] epoch: 1, step: 50, global_step: 50
  losses: {pedal: 0.18}
  note_monitor_losses: {vel: ..., ons: ...}
  loss_mode: pedal_only_note_losses_are_monitoring_only
  LR: 0.00027
```

**What to look for:**
- **Pedal loss** should gradually decrease.
- **Onset/velocity monitor losses** should remain roughly stable because those layers are frozen; they are logged only as a regression monitor and are not part of the optimized loss.
- If note evaluation F1 drops after this workflow, suspect checkpoint selection/evaluation thresholding rather than pedal training updates, because note parameters are frozen.

If `note_monitor_losses.ons` is extremely high at the start, first check that a
real note checkpoint was loaded. Freezing a random onset backbone will produce
large monitor losses and poor note F1. The default baseline path is:

```text
out/model_snapshots/OnsetsAndVelocities_2023_03_04_09_53_53.289step=43500_f1=0.9675__0.9480.torch
```

---

## 📈 **Expected Results**

### After Pedal-Only Fine-Tuning:
- **Onset F1 / Onset+Velocity F1:** should remain close to the loaded note checkpoint after threshold search.
- **Sustain-pedal F1:** should improve compared with the untrained/random pedal head.
- **Predictions per file:** should remain reasonable for notes because note logits are preserved.

### Compared to Original Paper (15 epochs, no pedal):
- **Onset F1:** ~0.967 (reference)
- **Onset+Velocity F1:** ~0.945 (reference)

Because the note backbone is frozen, the goal is not to relearn note detection;
it is to preserve the checkpoint’s note performance while adding usable pedal
predictions.

---

## 🎯 **Why These Settings Work**

### 1. **4-Second Chunks (was 0.05s)**
**Old problem:** 50ms chunks = model learns from tiny snippets, can't understand musical context

**Solution:** 4 seconds = enough for:
- Multiple note onsets
- Velocity patterns
- Pedal state transitions
- Temporal context

### 2. **8 Epochs (was 2)**
**Old problem:** 2 epochs = model barely sees the data

**Solution:** 8 epochs = good balance:
- Enough to learn patterns
- Not so many that it overfits
- Finishes in ~10-12 hours

### 3. **Frozen Note Backbone + Small Pedal LR**
**Old problem:** pedal loss was introduced into a workflow that could update or resume from degraded note checkpoints, so onset/velocity F-score suffered.

**Solution:**
- `PEDAL_ONLY_FINETUNE=true` freezes note layers.
- `DETACH_PEDAL_FEATURES=true` prevents pedal gradients from touching shared features.
- `PEDAL_LR_MAX=0.0003` avoids unstable updates in the new pedal head.
- `MAX_GRAD_NORM=1.0` clips occasional transition-heavy gradient spikes.

---

## 💾 **Model Checkpoints**

The training script will save models periodically (every LR cycle):
```
out/model_snapshots/
├── OnsetsAndVelocities_2026_02_01_00_15_32.torch
├── OnsetsAndVelocities_2026_02_01_01_23_45.torch
└── ...
```

**Use the latest checkpoint only if it came from the safe pedal-only workflow.**
If you have mixed older experimental checkpoints in `out/model_snapshots`, pass the
exact checkpoint path to evaluation instead of relying on “latest”.

Checkpoint artifact policy:
- `.torch` checkpoints are generated binary artifacts and are ignored by Git.
- Do **not** commit new checkpoints directly and do **not** add them to Git LFS for normal development.
- To share a selected trained model, upload it as a versioned GitHub/GitLab release asset or another documented download artifact, then document the download URL, expected filename/path, and any relevant metric/checksum metadata.
- Keep local training outputs under `out/model_snapshots/`; copy or download a shared checkpoint there when running evaluation locally.

---

## 🔍 **Monitoring Training**

### Check Training Logs:
```bash
# View latest log
tail -f out/txt_logs/[02_train_pedal_model.py]*.log
```

### Check Loss Trends:
```bash
python scripts/05_analyze_training_logs.py LOG_PATH="out/txt_logs/YOUR_LOG.json"
```

---

## ⚠️ **If Training Stops/Crashes**

The training script only auto-resumes from the latest generated checkpoint when
you explicitly set `RESUME_FROM_LATEST=true`:

```bash
# Continue a known-good pedal-only run
python scripts/02_train_pedal_model.py RESUME_FROM_LATEST=true
```

**Resume state saved every 500 steps in:**
`out/model_snapshots/.resume_state.json`

---

## 🎯 **After Training Completes**

### Step 1: Evaluate on Test Set
```bash
python scripts/03_evaluate_pedal_model.py
```

You should now see:
- ✅ **Reasonable prediction counts** (2k-5k per file, not 500k!)
- ✅ **Good F1 scores** (~0.85-0.90)
- ✅ **Most files processed successfully**

### Step 2: Generate Plots
```bash
python scripts/06_visualize_pedal_predictions.py
```

### Step 3: Analyze Logs
```bash
python scripts/05_analyze_training_logs.py
```

---

## 📊 **Memory Usage**

With these settings:
- **VRAM:** ~6-7 GB (fits in 8GB)
- **RAM:** ~10-12 GB (fits in 16GB)
- **Disk:** ~25 GB (model checkpoints)

---

## 🔧 **If You Need Faster Training**

### Reduce Epochs (Less Quality)
```bash
python scripts/02_train_pedal_model.py NUM_EPOCHS=5
```
**Time:** ~6-8 hours (1 day max)

### Increase Chunk Size (More Memory, Faster)
```bash
python scripts/02_train_pedal_model.py TRAIN_BATCH_SECS=5.0
```
**Time:** ~8-10 hours (slightly faster)

---

## 🎉 **Summary**

**Old Config:**
- 50ms chunks
- 2 epochs
- Result: 500k predictions per file (broken)
- Time: ~2 hours (but useless model)

**New Config:**
- 4-second chunks
- 8 epochs of pedal-head-only fine-tuning
- Result: Preserved onset/velocity predictions plus trained sustain-pedal output
- Time: **10-12 hours (< 2 days)**

**Your pedal head will be specialized while the high-F1 note model is protected.** 🎹🎶

---

## 🚀 **Start Training Now**

```bash
conda activate onsvel
python scripts/02_train_pedal_model.py RESUME_FROM_LATEST=false PEDAL_ONLY_FINETUNE=true
```

**The training will complete in 1-2 days, well within your 3-5 day target!**
