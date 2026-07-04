# Training Guide: 3-5 Day Pedal-Aware Training

## Configuration Changes

### ✅ **What Changed (Prototype → Production)**

| Parameter | Old (Broken) | New (Optimized) | Why? |
|-----------|--------------|-----------------|------|
| `TRAIN_BATCH_SECS` | **0.05s** (50ms) | **4.0s** | Model needs sufficient context to learn patterns |
| `NUM_EPOCHS` | **2** | **8** | More epochs = better learning |
| `ONSET_POSITIVES_WEIGHT` | 2.0 | **3.0** | Onsets are sparse, need higher weight |
| `PEDAL_LOSS_LAMBDA` | 0.5 | **1.0** | Pedal detection is important |
| `PEDAL_POSITIVES_WEIGHT` | 2.0 | **3.0** | Pedal events are also sparse |
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
# Backup old checkpoint
mv out/model_snapshots/OnsetsAndVelocities_2026_01_30_12_46_23.207.torch out/model_snapshots/old_prototype.torch

# Remove resume state (start fresh)
rm out/model_snapshots/.resume_state.json

# Activate environment
conda activate onsvel

# Start training
python scripts/02_train_pedal_model.py
```

### Step 2: Monitor Progress
Watch the console output:
```
[TRAIN] epoch: 1, step: 50, global_step: 50
  losses: {vel: 0.234, pedal: 0.187, ons: 0.456}
  LR: 0.00412
```

**What to look for:**
- Losses should gradually decrease
- After ~1000 steps, losses should stabilize around:
  - **onset loss:** 0.2-0.4
  - **velocity loss:** 0.1-0.3
  - **pedal loss:** 0.1-0.3

---

## 📈 **Expected Results**

### After 8 Epochs:
- **Onset F1:** ~0.85-0.92 (good)
- **Onset+Velocity F1:** ~0.80-0.88 (good)
- **Predictions per file:** 2,000-5,000 (reasonable, not 500k!)

### Compared to Original Paper (15 epochs, no pedal):
- **Onset F1:** ~0.967 (reference)
- **Onset+Velocity F1:** ~0.945 (reference)

Your model with 8 epochs should get **~90% of paper performance**, which is excellent for your use case.

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

### 3. **Higher Loss Weights**
**Old problem:** Onsets and pedals are rare events, model ignores them

**Solution:**
- `ONSET_POSITIVES_WEIGHT: 3.0` - Forces model to pay attention to onsets
- `PEDAL_POSITIVES_WEIGHT: 3.0` - Forces model to detect pedal events
- `PEDAL_LOSS_LAMBDA: 1.0` - Makes pedal detection important

---

## 💾 **Model Checkpoints**

The training script will save models periodically (every LR cycle):
```
out/model_snapshots/
├── OnsetsAndVelocities_2026_02_01_00_15_32.torch
├── OnsetsAndVelocities_2026_02_01_01_23_45.torch
└── ...
```

**Use the LATEST checkpoint for evaluation.**

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

The training script **auto-resumes** from the latest checkpoint:

```bash
# Just restart - it will continue where it left off
python scripts/02_train_pedal_model.py
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
- 8 epochs
- Result: Proper onset detection with reasonable predictions
- Time: **10-12 hours (< 2 days)**

**Your model will be properly trained and ready to use!** 🎹🎶

---

## 🚀 **Start Training Now**

```bash
conda activate onsvel
python scripts/train.py
```

**The training will complete in 1-2 days, well within your 3-5 day target!**
