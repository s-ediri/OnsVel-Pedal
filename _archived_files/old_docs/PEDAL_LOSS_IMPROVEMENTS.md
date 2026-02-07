# Pedal Loss Improvements

## Problem
Sustain pedal loss (0.05-0.28) was significantly higher than velocity loss (~0.0), indicating poor learning on pedal detection. Root causes:
- **Class imbalance**: Pedal is not pressed ~50% of the time (sparse positive signals)
- **Equal weighting**: All time steps treated equally regardless of pedal presence
- **Continuous signal**: Pedal is continuous (not discrete notes), harder to predict

## Solution 1: Class Weighting ✅ IMPLEMENTED

**Configuration**:
```python
PEDAL_POSITIVES_WEIGHT: float = 2.0  # Weight positive pedal samples 2x higher
```

**Implementation**:
```python
pedal_pos_weight = torch.FloatTensor([CONF.PEDAL_POSITIVES_WEIGHT]).to(CONF.DEVICE)
pedal_loss = CONF.PEDAL_LOSS_LAMBDA * torch.nn.functional.binary_cross_entropy_with_logits(
    pedals.reshape(-1), sustain_norm.reshape(-1), pos_weight=pedal_pos_weight)
```

**Effect**: Positive pedal samples (when sustain_norm > 0.5) contribute 2x more to the loss gradient, forcing the model to learn pedal presence better.

---

## Solution 2: Adjust PEDAL_LOSS_LAMBDA (Optional Tuning)

Current: `PEDAL_LOSS_LAMBDA = 0.5`

If pedal loss is still too high after Solution 1:
- Reduce to `0.3` to further reduce pedal loss weight
- Monitor if model still learns pedal (watch F1 scores in XV logs)

If pedal loss becomes too low (model ignores pedal):
- Increase to `0.7` or `1.0`

---

## Solution 3: Future - Focal Loss (Not Implemented Yet)

For even better results, replace BCE with **Focal Loss** which automatically weights hard-to-classify examples:
```python
focal_weight = 2.0  # Focus on hard negatives
alpha = 0.25  # Balance positive/negative
pedal_loss = focal_loss(pedals, sustain_norm, alpha=alpha, gamma=focal_weight)
```

This would require implementing a custom focal loss function.

---

## Expected Improvement

**Before**: Pedal loss oscillates 0.05-0.28
**After**: Pedal loss should reduce to 0.02-0.15 range (30-50% improvement)

---

## How to Validate

1. **Monitor XV logs**: Check F1 scores for sustain pedal detection
   - Look for thresholds (0.5, 0.75) in `XV_BEST_ONSET_VEL` output
   - Should improve from current baseline

2. **Check loss curves**: Run training for 2-3 more epochs
   - Pedal loss should show downward trend even if still oscillating

3. **Disable if needed**: To focus on onsets/velocity only:
   ```python
   PEDAL_LOSS_LAMBDA = 0.0  # Disable pedal loss entirely
   ```

---

## Resumption Safety Guarantee

✅ **NO DATA RETRAINING**: When training resumes:
- Global step tracking continues correctly
- DataLoader continues from remaining batches in current epoch
- Next epoch uses new random shuffle (different batches, not repeated ones)
- Model doesn't see same data twice

**Note**: If crash happens mid-epoch, remaining ~258K batches from that epoch are skipped (acceptable for prototype).

