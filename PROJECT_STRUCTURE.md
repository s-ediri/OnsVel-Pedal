# Project Structure - Piano Transcription with Sustain Pedal

## Clean Directory Layout

```
iamusica_training/
├── 📄 Core Scripts (6 files)
│   ├── 0a_maestro_to_hdf5mel.py      # Preprocess MAESTRO dataset to HDF5
│   ├── 0b_maps_to_hdf5mel.py         # Preprocess MAPS dataset to HDF5
│   ├── 1_train_onsets_velocities.py  # Training script (with pedal support)
│   ├── 2_eval_onsets_velocities.py   # Evaluation script (FIXED)
│   ├── 3_analyze_logs.py             # Training log analysis
│   └── 4_qualitative_plots.py        # Generate visualization plots
│
├── 📁 ov_piano/ - Core Module
│   ├── __init__.py
│   ├── logging.py                    # Logging utilities
│   ├── utils.py                      # Model loading, training utils
│   ├── optimizers.py                 # AdamWR optimizer
│   ├── inference.py                  # Strided inference, decoders (FIXED)
│   ├── eval.py                       # Evaluation metrics, GT loaders
│   ├── data/
│   │   ├── __init__.py
│   │   ├── maestro.py                # MAESTRO dataset loaders
│   │   ├── maps.py                   # MAPS dataset loaders
│   │   ├── midi.py                   # MIDI parsing
│   │   └── key_model.py              # Keyboard state machine
│   └── models/
│       ├── __init__.py
│       ├── ov.py                     # OnsetsAndVelocities model (FIXED)
│       └── building_blocks.py        # Neural network components
│
├── 📚 Documentation (3 files)
│   ├── README.md                     # Original project documentation
│   ├── EVALUATION_FIXES_SUMMARY.md   # Comprehensive fixes & architecture
│   └── QUICK_START_EVALUATION.md     # Quick start evaluation guide
│
├── 🔧 Utilities
│   └── breakpoint.json               # Training debugging control
│
├── 📦 Data (not tracked)
│   ├── datasets/
│   │   ├── maestro/
│   │   │   └── maestro-v3.0.0/
│   │   ├── MAESTROv3_logmel_*.h5
│   │   └── MAESTROv3_roll_*.h5
│
├── 💾 Output (not tracked)
│   └── out/
│       ├── model_snapshots/          # Trained model checkpoints
│       └── txt_logs/                 # Training & evaluation logs
│
└── 🗄️ _archived_files/ (35+ files moved here)
    ├── old_docs/                     # 11 superseded documentation files
    ├── debug_scripts/                # 13 temporary test/debug scripts
    └── scripts/                      # Old analysis scripts
```

---

## Quick Reference

### Training
```bash
conda activate onsvel
python 1_train_onsets_velocities.py
```

### Evaluation
```bash
python 2_eval_onsets_velocities.py
```

### Log Analysis
```bash
python 3_analyze_logs.py LOG_PATH="out/txt_logs/YOUR_LOG.json"
```

### Visualization
```bash
python 4_qualitative_plots.py SNAPSHOT_INPATH="out/model_snapshots/YOUR_MODEL.torch"
```

---

## What Was Cleaned Up

### Removed from Root (moved to `_archived_files/`)

**Debug/Test Scripts (13 files):**
- analyze_dataset_splits.py
- check_pedal_structure.py
- debug_memory.py
- eval_with_pedals.py
- fixed_strided_inference.py
- memory_optimizations.py
- pedal_inference_demo.py
- quick_verify_sustain_pedal.py
- simple_eval.py
- test_pedal_evaluation.py
- train_with_optimizations.py
- validate_sustain_pedal.py
- verify_checkpoint.py

**Old Documentation (11 files):**
- FINAL_VERIFICATION.md
- IMPLEMENTATION_COMPLETE.md
- MEMORY_OPTIMIZATION_GUIDE.md
- MID_EPOCH_RESUMPTION.md
- PEDAL_IMPLEMENTATION_GUIDE.md
- PEDAL_LOSS_IMPROVEMENTS.md
- QUICK_REFERENCE_RESUMPTION.md
- QUICK_START_LIMITED_MEMORY.md
- READY_TO_TRAIN.md
- SUSTAIN_PEDAL_OPTIMIZATION.md
- TEST_MID_EPOCH_RESUMPTION.md

**Miscellaneous:**
- cleaned_attendance.csv (unrelated)
- train.bat (redundant)
- scripts/ directory

---

## Why These Files Were Removed

1. **Debug scripts:** Temporary testing code that's no longer needed
2. **Old docs:** Fragmented information now consolidated into:
   - `EVALUATION_FIXES_SUMMARY.md` (comprehensive technical guide)
   - `QUICK_START_EVALUATION.md` (quick start guide)
3. **Duplicate functionality:** Features now integrated into main scripts

---

## Recovery

All archived files are preserved in `_archived_files/` if you need to recover anything:
- `_archived_files/old_docs/` - Old documentation
- `_archived_files/debug_scripts/` - Test/debug scripts
- `_archived_files/scripts/` - Old analysis scripts

To restore a file:
```bash
cp _archived_files/debug_scripts/FILENAME.py .
```

---

## File Count Summary

**Before Cleanup:** 46+ files in root directory
**After Cleanup:** 11 essential files in root directory
**Reduction:** 76% fewer files for easier navigation

---

**Project is now clean, organized, and ready for evaluation!** ✨
