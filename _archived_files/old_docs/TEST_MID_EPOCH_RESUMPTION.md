# Mid-Epoch Resumption - Testing Guide

## Quick Test (5 minutes)

### Step 1: Start Training
```bash
python train_with_optimizations.py
```

Wait for at least **100 steps** to accumulate so there's state to resume from. You'll see logs like:
```
["2026-01-28 22:XX:XX.XXX", ["TRAIN", {..., "global_step": 500, ...}]]
["2026-01-28 22:XX:XX.XXX", ["RESUME_STATE", "No previous state found, starting fresh"]]
```

### Step 2: Stop Training (Manually)
Press `Ctrl+C` to stop training mid-epoch. Should see output like:
```
KeyboardInterrupt
```

### Step 3: Check Resume State File
```bash
# List the resume state file
dir out\model_snapshots\.resume_state.json
```

Should see a file. Examine it:
```bash
type out\model_snapshots\.resume_state.json
```

Output should look like:
```json
{"epoch": 1, "batch_idx": 425, "global_step": 40500}
```

### Step 4: Restart Training
```bash
python train_with_optimizations.py
```

**Key observations**:
- ✅ Script finds latest checkpoint (auto-resume)
- ✅ Loads resume state
- ✅ Logs show: `"RESUME_STATE": {"epoch": 1, "batch_idx": 425, "global_step": 40500}`
- ✅ Logs show: `"RESUMING": {"epoch": 1, "batch_idx": 425, "message": "Resumed from saved state..."}`
- ✅ `global_step` starts from 40500 (continues correctly)
- ✅ **Most importantly**: Loss values should be similar (same data being trained)

---

## Advanced Test (Verifying Reproducibility)

### Test Shuffles are Reproducible

**Setup**: 
1. Train for 1000 steps
2. Note batch values in loss logs
3. Stop training
4. Delete all checkpoints EXCEPT the most recent
5. Restart from same point
6. Compare loss values

**Expected**:
- Same batch sequence = same loss values (within small numerical precision)
- Different epoch number = different batch sequence (verify shuffles differ per epoch)

### Test Data Coverage

**Setup**:
1. Run for 5000 steps
2. Track unique batch IDs (if using logging)
3. Stop and resume
4. Continue for another 5000 steps
5. Verify all 10000 steps are unique (no repeats)

---

## Edge Cases to Verify

### Case 1: Resume at Epoch Boundary
**Setup**:
1. Train until end of epoch 1 (batch_idx = 298667)
2. Stop training
3. Restart

**Expected**:
- Resume state shows: `{"epoch": 1, "batch_idx": 298667, ...}`
- Skips 298667 batches of epoch 1
- Moves to epoch 2 naturally
- No data repeated

### Case 2: Corrupt Resume State
**Setup**:
1. Train for 1000 steps
2. Delete `.resume_state.json` file
3. Restart training

**Expected**:
- Logs show: `"RESUME_STATE": "No previous state found, starting fresh"`
- Trains from epoch 1, batch 0 (fresh start)
- No errors

### Case 3: Resume After Long Gap
**Setup**:
1. Train for 50,000 steps
2. Stop and restart
3. Verify training continues correctly after long gap

**Expected**:
- Resume state is persistent across sessions
- Works even if trained days later
- Learning rate picks up from correct cycle position

---

## What NOT to Do

❌ **Don't change `CONF.RANDOM_SEED`** between resume and restart
- Will break shuffle reproducibility
- Training will see different batches than saved

❌ **Don't delete `.resume_state.json` manually** unless you want to retrain
- File is safely hidden (dot prefix) so won't interfere
- Leaving it enables resumption

❌ **Don't move or rename model checkpoints** after saving
- Auto-resume looks for latest checkpoint by modification time
- Moving breaks the reference

✅ **DO** interrupt training with Ctrl+C (safe)
✅ **DO** let the script auto-detect and resume (automatic)
✅ **DO** use same configuration/code (no changes between runs)

---

## Logging to Watch

### On First Run:
```
["RESUME_STATE", "No previous state found, starting fresh"]
```

### On Resumed Run:
```
["RESUME_STATE", {"epoch": 2, "batch_idx": 150, "global_step": 40150}]
["RESUMING", {"epoch": 2, "batch_idx": 150, "message": "Resumed from saved state, continuing from batch 150"}]
```

### Every 500 steps:
```
["TRAIN", {...}]  # Normal training log, then resume state updates silently
```

### On Checkpoint Save (XV or LR cycle):
```
["SAVED_MODEL", "Saved model to out\\model_snapshots\\OnsetsAndVelocities_2026_01_28_...torch"]
# Resume state also gets updated at checkpoint time
```

---

## Performance Indicators

**Good Signs**:
- ✅ Loss values stay in expected range (0.05-0.3)
- ✅ No sudden spikes in loss when resuming
- ✅ `global_step` increments continuously
- ✅ Resume state file (~100 bytes) updates every 500 steps
- ✅ Training speed unchanged (no overhead)

**Bad Signs**:
- ❌ Loss jumps significantly on resume (data mismatch)
- ❌ Resume state file not updating (check permissions)
- ❌ Training starts from epoch 1 on resume (resume state not found/loaded)
- ❌ Repeated loss values (batch not advancing properly)

---

## Cleanup

To start completely fresh (wipe all training state):
```bash
# Option 1: Delete all checkpoints and resume state
rm -r out/model_snapshots/*

# Option 2: Just delete resume state (keeps checkpoints)
rm out/model_snapshots/.resume_state.json

# Then restart training
python train_with_optimizations.py
```

---

## Troubleshooting

### Issue: "Resume state shows different epoch but training continues from epoch 1"

**Cause**: Resume state loaded but not acting on it
**Fix**: Check that training loop uses `resume_epoch` as range start
```python
# Should be:
for epoch in range(resume_epoch, CONF.NUM_EPOCHS + 1):  # ✅ Correct

# NOT:
for epoch in range(1, CONF.NUM_EPOCHS + 1):  # ❌ Wrong - ignores resume_epoch
```

### Issue: "Training seems slow after resume"

**Cause**: Might be DataLoader recreation overhead
**Fix**: Normal - DataLoader recreation happens once per epoch, negligible impact

### Issue: "Loss values don't match after resume"

**Cause**: DataLoader shuffle is different (broken reproducibility)
**Fix**: 
1. Check `CONF.RANDOM_SEED` is same between runs
2. Verify epoch-seeded shuffle works: `epoch_seed = CONF.RANDOM_SEED + epoch_num`

---

## Success Criteria

✅ Mid-epoch resumption is working correctly when:
1. Resume state file is created and persists
2. Training resumes at correct epoch and batch
3. Global step counter continues from saved value
4. Loss values stay consistent (no sudden jumps)
5. Training completes without repeating batches
6. All 298,668 batches trained per epoch (no gaps)

