# Windows Memory Optimization Guide

## Problem
You were experiencing:
- `ImportError: DLL load failed while importing parsing`
- `MemoryError`
- Memory issues despite 46GB free space and increased paging file

## Root Causes
1. **Memory fragmentation**: h5py DLL loading fails when physical memory is fragmented
2. **Insufficient paging file configuration**: Windows paging needs to be much larger
3. **No aggressive garbage collection**: Python doesn't free memory fast enough
4. **Large batch processing**: Loading full validation files without chunking
5. **h5py file handles not closing**: Files stay open across iterations

## Solutions Implemented

### 1. Training Script Changes (`1_train_onsets_velocities.py`)
- Reduced `TRAIN_BATCH_SECS` from 0.5 to 0.25 seconds (smaller chunks)
- Increased `GRADIENT_ACCUMULATION_STEPS` from 2 to 4 (compensates for smaller batches)
- Added `gc.collect()` calls for aggressive garbage collection
- Added `cleanup_memory()` function that:
  - Calls Python's garbage collector
  - Empties CUDA cache
  - Prints memory statistics
- Added periodic memory cleanup every 50 training steps
- Added aggressive cleanup before/after cross-validation
- Added explicit `del` statements to free variables immediately

### 2. New Utility Script (`memory_optimizations.py`)
Provides Windows-specific memory management:
```python
optimize_windows_memory()  # Configure PyTorch for Windows
check_memory_requirements()  # Check system memory status
```

Features:
- Sets `PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"` (better memory allocation)
- Sets thread limits to reduce memory overhead
- Enables CUDA memory pooling
- Limits VRAM usage to 80% to avoid allocation errors
- Checks system memory and paging file size
- Warns if paging file is too small

### 3. Launcher Scripts
- `train.bat`: Windows batch script (easy double-click launching)
- `train_with_optimizations.py`: Python launcher that applies optimizations

## How to Use

### Option A: Using batch file (Easiest)
```bash
train.bat
```

### Option B: Using Python launcher
```bash
python train_with_optimizations.py
```

### Option C: Direct with memory optimization
```bash
python memory_optimizations.py
python 1_train_onsets_velocities.py
```

## Windows Paging File Configuration (Important!)

Even though you've increased it, ensure it's set correctly:

1. Right-click **This PC** → **Properties**
2. Click **Advanced system settings**
3. Go to **Advanced** tab → **Performance** → **Settings**
4. Click **Advanced** tab → **Virtual Memory** → **Change**
5. Set paging file size:
   - **Initial size**: 100,000 MB (100 GB)
   - **Maximum size**: 200,000 MB (200 GB)
6. Click **Set** → **OK** → Restart

## Configuration Tuning

If you still encounter memory errors, try:

### Option 1: Reduce batch size further
In `1_train_onsets_velocities.py`, change:
```python
TRAIN_BATCH_SECS: float = 0.125  # Reduce to 0.125 instead of 0.25
```

### Option 2: Skip validation more frequently
```python
XV_EVERY: int = 2000  # Validate every 2000 steps instead of 1000
```

### Option 3: Use CPU for validation
Modify the code to move model to CPU during validation to free VRAM.

### Option 4: Reduce validation set size (already done)
The code already reduces validation set by 5x (every 5th file).

## Monitoring Memory

The scripts now print memory statistics:
```
RSS Memory: 8234.5 MB, VMS Memory: 12450.3 MB
```

- **RSS**: Physical RAM being used
- **VMS**: Total virtual memory (RAM + Paging file)

Watch for RSS growing unbounded - that indicates a memory leak.

## If Problems Persist

1. **Check Windows Event Viewer** for system errors:
   - Press `Win+R`, type `eventvwr`, search for errors around crash time

2. **Monitor resource usage during training**:
   ```bash
   tasklist /v  # Check memory usage
   wmic OS get TotalVisibleMemorySize  # Total physical RAM
   ```

3. **Try disabling GPU temporarily**:
   ```bash
   python 1_train_onsets_velocities.py DEVICE=cpu
   ```
   (This will be much slower but can help isolate GPU memory issues)

4. **Check h5py installation**:
   ```bash
   pip install --upgrade h5py
   ```

5. **Consider using system memory less aggressively**:
   In `memory_optimizations.py`:
   ```python
   torch.cuda.set_per_process_memory_fraction(0.6)  # Use 60% instead of 80%
   ```

## Technical Details

### Why the DLL Error Happens
The h5py library uses a C-level DLL that requires contiguous memory allocation. When memory is fragmented, Windows cannot allocate a large enough contiguous block, causing the DLL load to fail.

### Why Larger Paging File Helps
A larger paging file on a different disk allows Windows to:
- Swap infrequently-used memory to disk
- Reduce fragmentation in physical RAM
- Provide more flexibility for large allocations

### Why Smaller Batches Help
- Less data loaded into memory at once
- Reduced peak memory usage
- More frequent checkpoints for garbage collection
- PyTorch can deallocate intermediate results more often

## Expected Performance After Changes

- **Training speed**: Slightly slower (smaller batches, more frequent cleanup)
- **Stability**: Much more stable (should no longer crash)
- **Memory usage**: More stable, peaks lower
- **Validation**: Takes slightly longer (but happens less frequently)

---

**Note**: These optimizations are specifically designed for Windows systems with the h5py DLL loading issue. Performance on Linux/Mac systems should not be significantly affected, though they benefit from the garbage collection improvements.
