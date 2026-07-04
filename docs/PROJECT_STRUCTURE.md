# Project Structure - OnV+Pedal

## Clean Directory Layout

```
OnV+Pedal/
├── scripts/ - Runnable entry points
│   ├── 00_prepare_maestro_hdf5.py    # Preprocess MAESTRO dataset to HDF5
│   ├── 01_prepare_maps_hdf5.py       # Preprocess MAPS dataset to HDF5
│   ├── 02_train_pedal_model.py       # Training script (with pedal support)
│   ├── 03_evaluate_pedal_model.py    # Evaluation script
│   ├── 04_evaluate_test_split.py     # Test split evaluation
│   ├── 05_analyze_training_logs.py   # Training log analysis
│   └── 06_visualize_pedal_predictions.py # Generate visualization plots
│
├── 📁 ov_piano/ - Core Module
│   ├── __init__.py
│   ├── custom_logging.py             # Logging utilities
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
├── docs/ - Documentation
│   ├── README.md                     # Pedal-focused project documentation
│   ├── EVALUATION_FIXES_SUMMARY.md   # Comprehensive fixes & architecture
│   └── QUICK_START_EVALUATION.md     # Quick start evaluation guide
│
├── 🔧 Utilities
│   └── breakpoint.json               # Training debugging control
│
├── tests/                         # Pytest smoke/regression tests
│
├── web_app/                       # Flask transcription UI
│
├── assets/                        # Static project assets/checkpoints
│   └── OnsetsAndVelocities_*.torch # Reference pretrained checkpoint
│
├── Data (not tracked)
│   ├── datasets/
│   │   ├── maestro/
│   │   │   └── maestro-v3.0.0/
│   │   ├── MAESTROv3_logmel_*.h5
│   │   └── MAESTROv3_roll_*.h5
│
├── out/                           # Generated runs (not tracked)
│   ├── model_snapshots/            # Trained model checkpoints
│   └── txt_logs/                   # Training & evaluation logs
│
└── uploads/                       # Uploaded model/audio artifacts (not tracked)
```

---

## Quick Reference

### Training
```bash
conda activate onsvel
python scripts/02_train_pedal_model.py
```

### Evaluation
```bash
python scripts/03_evaluate_pedal_model.py
```

### Log Analysis
```bash
python scripts/05_analyze_training_logs.py LOG_PATH="out/txt_logs/YOUR_LOG.json"
```

### Visualization
```bash
python scripts/06_visualize_pedal_predictions.py SNAPSHOT_INPATH="out/model_snapshots/YOUR_MODEL.torch"
```

---

## Structure Decision

Git history shows three main layouts:

1. **Initial upstream layout (`c27f042`)**: root scripts plus `iamusica_ml/` package.
2. **Refactored package layout (`91be78d`)**: root scripts plus renamed `ov_piano/` package.
3. **Pedal-focused cleanup (`98a2b56` → `HEAD`)**: `ov_piano/` core, many root scripts/docs/debug artifacts, and archived temporary files.

The current foldered layout is the best direction for this repository: it keeps reusable package code in `ov_piano/`, command-line workflows in `scripts/`, documentation in `docs/`, regression tests in `tests/`, and the Flask UI in `web_app/`. This is easier to navigate than the historical root-script layouts while preserving the established `ov_piano` domain boundary.

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
- root-level script clutter (current entry points live in `scripts/`)

---

## Why These Files Were Removed

1. **Debug scripts:** Temporary testing code that's no longer needed
2. **Old docs:** Fragmented information now consolidated into:
   - `EVALUATION_FIXES_SUMMARY.md` (comprehensive technical guide)
   - `QUICK_START_EVALUATION.md` (quick start guide)
3. **Duplicate functionality:** Features now integrated into main scripts

---

## Recovery

Historical files can be recovered from Git history if needed:
- `_archived_files/old_docs/` - Old documentation
- `_archived_files/debug_scripts/` - Test/debug scripts
- `_archived_files/scripts/` - Old analysis scripts

To restore a file:
```bash
git checkout <commit> -- path/to/file
```

---

## File Count Summary

**Before Cleanup:** 46+ files in root directory
**After Cleanup:** 11 essential files in root directory
**Reduction:** 76% fewer files for easier navigation

---

**Project is now clean, organized, and ready for evaluation!** ✨
