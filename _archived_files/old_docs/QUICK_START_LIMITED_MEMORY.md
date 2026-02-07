# Windows Memory Setup - For Limited Systems

Your system: **17.1 GB RAM + 32 GB paging file = ~49 GB total virtual memory**

## Quick Setup

### 1. Set Paging File (One-time setup)

1. Right-click **This PC** → **Properties**
2. Click **Advanced system settings**
3. Go to **Advanced** tab → **Performance** → **Settings**
4. Click **Advanced** tab → **Virtual Memory** → **Change**
5. Set values:
   - **Initial size:** 2000 MB (let Windows grow it)
   - **Maximum size:** 32768 MB (32 GB)
6. Click **Set** → **OK** → **Restart computer**

### 2. Before Training

Close these applications:
- Web browsers (Chrome uses 2-4 GB per window)
- Discord, Slack, Teams
- IDEs except VS Code (if running training script)
- Anything in system tray

### 3. Start Training

**In onsvel environment:**
```bash
conda activate onsvel
cd E:\FYP\iamusica_training
python 1_train_onsets_velocities.py
```

## What Changed (Memory-optimized settings)

| Setting | Old | New | Why |
|---------|-----|-----|-----|
| Batch seconds | 0.5 | 0.125 | Smaller chunks = less memory |
| Gradient accumulation | 4 | 8 | Compensates for smaller batches |
| Validation frequency | 1000 steps | 5000 steps | Less frequent = less memory spikes |
| Thresholds | 5 values | 1 value | Faster validation = less memory |

## What to Monitor

While training runs, watch for:

✓ **Good signs:**
- Memory usage stable (doesn't keep growing)
- GPU utilization 80-90%
- Training logs appear every 10 steps

✗ **Bad signs:**
- Memory growing continuously (memory leak)
- GPU errors every few minutes
- "Paging file too small" errors

## If You Still Get Memory Errors

Try these in order:

**Option 1** - Skip validation during training:
```bash
python 1_train_onsets_velocities.py XV_EVERY=999999
```
(Validation will only happen at end)

**Option 2** - Even smaller batches:
```bash
python 1_train_onsets_velocities.py TRAIN_BATCH_SECS=0.06
```

**Option 3** - Use CPU for inference only:
```bash
python 1_train_onsets_velocities.py DEVICE=cuda XV_DEVICE=cpu
```

**Option 4** - Reduce precision (uses less memory):
Add to training script before training loop:
```python
model.half()  # Use float16 instead of float32
```

## Paging File Details

- **Initial size:** Start small (2 GB) - Windows expands as needed
- **Maximum size:** 32 GB (your limit)
- **Location:** Should be on different drive if possible (you only have C:, so that's fine)
- **Best practice:** Max should be 1-2x your physical RAM (you have 17 GB RAM, so 32 GB max is good)

---

**Your Configuration Summary:**
- Physical RAM: 17.1 GB
- VRAM (GPU): 8.6 GB
- Paging File: 2-32 GB (growing as needed)
- **Total available:** ~57 GB (should be enough!)

This should work now. The key is: **paging file will grow as needed**, and **more aggressive memory cleanup** during training prevents spikes.
