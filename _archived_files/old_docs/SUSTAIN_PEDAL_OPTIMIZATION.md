# Sustain Pedal Training - Optimization Summary

## Hardware Configuration
- **GPU**: NVIDIA GeForce RTX 2070 SUPER (8GB VRAM)
- **CPU**: AMD Ryzen 5 3600 6-Core Processor
- **RAM**: 17.1 GB

## Optimizations Applied

### 1. Memory Optimizations
- ✓ **Batch Size**: Increased from 2 to 4 (RTX 2070 SUPER can handle this)
- ✓ **Gradient Accumulation**: Reduced from 8 to 4 (sufficient with BS=4)
- ✓ **PyTorch CUDA Config**: 
  - Enabled cuDNN benchmarking for auto-tuned convolutions
  - Enabled TensorFloat32 for faster matrix operations
  - Configured expandable memory segments to reduce fragmentation
- ✓ **Data Loading**: Set `DATALOADER_WORKERS=4` (optimal for Ryzen 5 3600)

### 2. Learning Rate Adjustments
- **LR_MAX**: 0.008 → 0.006 (reduced for stability)
- **LR_PERIOD**: 1000 → 2000 steps (longer, more stable cycles)
- **LR_DECAY**: 0.975 → 0.98 (more conservative decay)

### 3. Model Regularization
- **Dropout**: 0.15 → 0.2 (slightly increased)
- **Batch Norm Momentum**: 0.95 (unchanged - good for small-medium batches)

### 4. Training Duration
- **NUM_EPOCHS**: 10 → 15 (more training for better convergence)
- **XV_EVERY**: 999999 → 5000 (validate every 5000 steps for monitoring)
- **XV_THRESHOLDS**: [0.75] → [0.5, 0.75] (test multiple thresholds)

### 5. Sustain Pedal Loss
- **PEDAL_LOSS_LAMBDA**: 1.0 → 0.5 (reduced weight - sustain pedal shouldn't dominate)
- **Implementation**: Binary cross-entropy loss between model predictions and MIDI ground truth

## Sustain Pedal Implementation Details

### Data Flow
1. **HDF5 Structure**: `[onsets (88); frames (88); sustain_pedal (88); soft (88); tenuto (88)]`
2. **Extraction**: Extract sustain_pedal rows, average across 88 keys to get single signal
3. **Normalization**: Divide by 127.0 to normalize to [0, 1] range
4. **Model Output**: Single sustain pedal prediction per time step

### Model Architecture
- **Pedal Stage**: Uses same `get_cam_stage` architecture as velocity stage
- **Output Shape**: `(batch, 1, time)` - single pedal prediction
- **Per-Key Predictions**: Model outputs predictions for each piano key, then averaged to single pedal signal

### Loss Computation
```python
pedal_loss = BCE(model_pedals.reshape(-1), target_sustain.reshape(-1))
total_loss = vel_loss + 0.5 * pedal_loss
```

## Performance Expectations

### Memory Usage (RTX 2070 SUPER, 8GB)
- **Batch Size 4**: ~6.5-7 GB during training
- **Safety Margin**: ~0.5-1.5 GB remaining for spikes
- **Data Loading**: Minimal (~100-200 MB) due to 4 workers

### Compute Performance (Ryzen 5 3600)
- **6 Cores / 12 Threads**: 
  - 4 data loading workers utilizes cores efficiently
  - Leaves 2 cores for OS + PyTorch threading
- **Expected Training Time**: ~12-15 hours for 15 epochs on MAESTRO dataset

## Quality Assurance

### Validation Script
Run before training to verify everything is working:
```bash
python validate_sustain_pedal.py
```

Checks:
- ✓ HDF5 file structure and sustain pedal data presence
- ✓ Model output shapes for sustain pedal
- ✓ Loss computation correctness
- ✓ Data extraction logic

### Training Monitoring
- **Loss Logging**: Every 10 steps to console/logs
- **Validation**: Every 5000 steps (was disabled, now re-enabled)
- **F1 Metrics**: For both onsets and velocities
- **Sustain Pedal**: Evaluated with thresholds [0.5, 0.75]

## Production Checklist

Before running full training:

- [ ] Run `validate_sustain_pedal.py` and verify all checks pass
- [ ] Check GPU has 8GB+ VRAM available: `nvidia-smi`
- [ ] Check disk space for outputs: ~500MB per checkpoint
- [ ] Ensure no other GPU processes running
- [ ] Monitor first 50 steps to verify shapes and losses are correct
- [ ] Set appropriate `NUM_EPOCHS` based on time available

## Key Parameters Summary

```python
TRAIN_BS: 4                    # Batch size
TRAIN_BATCH_SECS: 0.06        # Seconds per batch chunk
DATALOADER_WORKERS: 4         # CPU workers
GRADIENT_ACCUMULATION_STEPS: 4 # Gradient accumulation
LR_MAX: 0.006                 # Max learning rate
LR_PERIOD: 2000               # LR cycle period
NUM_EPOCHS: 15                # Training epochs
PEDAL_LOSS_LAMBDA: 0.5        # Sustain pedal loss weight
XV_EVERY: 5000                # Validation frequency
```

## Important Notes

1. **Sustain Pedal is Critical**: The sustain pedal loss is integrated into the main training loop. Do not disable it.
2. **Data Validation**: The sustain pedal data comes directly from MIDI files via the HDF5 pipeline - it's verified data.
3. **Loss Weight Balance**: PEDAL_LOSS_LAMBDA=0.5 provides good balance. If sustain pedal performance is poor, increase to 0.75-1.0.
4. **Can't Retrain**: Given the memory and time constraints, this training run should work first time. Run validation before main training.

## Troubleshooting

If you encounter OOM errors:
- Reduce TRAIN_BS from 4 to 2
- Reduce TRAIN_BATCH_SECS from 0.06 to 0.03
- Increase GRADIENT_ACCUMULATION_STEPS to 8

If validation is slow:
- Increase XV_EVERY from 5000 to 10000
- Reduce XV_CHUNK_SIZE from 100.0 to 50.0
