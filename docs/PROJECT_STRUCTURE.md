# Project Structure - OnsVel-Pedal

This document describes the current repository layout after moving active workflows into
`scripts/`, reusable code into `ov_piano/`, regression coverage into `tests/`, and
one-off diagnostics into `_archived_files/`.

## Current Directory Layout

```text
OnsVel-Pedal/
├── .gitignore                         # Ignore generated data, caches, uploads, outputs
├── breakpoint.json                    # Optional training/debug control file
├── environment.yml                    # Reproducible Conda environment
├── LICENSE
├── pyproject.toml                     # Python project metadata and build config
├── requirements.txt                   # pip dependency list
│
├── assets/                            # Tracked static assets and small documentation media
│
├── data_processing/                   # Reserved for dataset processing utilities
│
├── docs/                              # Project documentation
│   ├── README.md                      # Main usage guide
│   ├── PROJECT_STRUCTURE.md           # This file
│   ├── TRAINING_GUIDE_3-5_DAYS.md     # Training workflow guide
│   ├── QUICK_START_EVALUATION.md      # Evaluation quick start
│   ├── EVALUATION_FIXES_SUMMARY.md    # Evaluation fixes and notes
│   └── MEMORY_ISSUE_SOLUTION.md       # Memory troubleshooting guidance
│
├── ov_piano/                          # Reusable package code
│   ├── __init__.py
│   ├── custom_logging.py              # Logging helpers
│   ├── eval.py                        # Evaluation metrics and GT loaders
│   ├── inference.py                   # Strided inference and decoders
│   ├── optimizers.py                  # Optimizer implementations
│   ├── transcription.py               # Audio-to-MIDI transcription pipeline
│   ├── utils.py                       # Model loading and training utilities
│   ├── data/
│   │   ├── __init__.py
│   │   ├── key_model.py               # Keyboard/pedal state helpers
│   │   ├── maestro.py                 # MAESTRO dataset loaders
│   │   ├── maps.py                    # MAPS dataset loaders
│   │   └── midi.py                    # MIDI parsing utilities
│   └── models/
│       ├── __init__.py
│       ├── building_blocks.py         # Neural network building blocks
│       └── ov.py                      # Onsets-and-velocities model
│
├── scripts/                           # Maintained command-line entry points
│   ├── 00_prepare_maestro_hdf5.py     # Preprocess MAESTRO to HDF5
│   ├── 01_prepare_maps_hdf5.py        # Preprocess MAPS to HDF5
│   ├── 02_train_pedal_model.py        # Train pedal-aware model
│   ├── 03_evaluate_pedal_model.py     # Evaluate a model/checkpoint
│   ├── 04_evaluate_test_split.py      # Test-split evaluation workflow
│   ├── 05_analyze_training_logs.py    # Training log analysis
│   ├── 06_visualize_pedal_predictions.py # Prediction visualization
│   └── transcribe.py                  # CLI transcription entry point
│
├── tests/                             # Pytest smoke/regression tests
│   ├── test_midi_parser_fixtures.py
│   ├── test_pedal_decoder.py
│   ├── test_smoke_inference.py
│   ├── test_transcribe_cli.py
│   ├── test_transcription.py
│   └── test_web_app_api.py
│
├── web_app/                           # Flask transcription UI/API
│   ├── app.py
│   ├── static/
│   │   ├── css/
│   │   └── js/
│   └── templates/
│       └── index.html
│
├── _archived_files/                   # Archived non-entry-point diagnostics
│   └── debug_scripts/
│       ├── debug_model_probe.py
│       ├── debug_model_probe2.py
│       ├── debug_pedal_evaluation.py
│       └── debug_pedal_gt.py
│
└── uploads/                           # Runtime upload/output area; contents ignored
```

## Generated or Local-Only Paths

The following paths are expected during local development or training but should not be
treated as maintained source files:

- `datasets/` - downloaded/raw datasets and generated HDF5 files.
- `out/` - training runs, model snapshots, logs, plots, and evaluation outputs.
- `uploads/` - files uploaded through the Flask app or generated for download.
- `__pycache__/`, `.pytest_cache/`, and other Python/tool caches.

## Model Checkpoint Artifact Policy

`.torch` model checkpoints are binary training artifacts and should not be added to Git
history or Git LFS for normal development. Training can create many checkpoints under
`out/model_snapshots/`, and those files are intentionally ignored by `.gitignore`.

When a checkpoint must be shared, publish it as a versioned release asset or another
documented download artifact, then reference its URL, filename, expected path, and any
checksum/metric metadata in the relevant usage guide. The repository may still contain
legacy tracked checkpoints from earlier history, but do not add new `.torch` files unless
the project explicitly revisits this policy.

## Debug and Temporary Artifact Policy

`scripts/` is reserved for maintained, documented entry points. Tracked ad-hoc debug and
temporary files were reviewed and handled as follows:

- Archived useful diagnostics in `_archived_files/debug_scripts/`:
  - `debug_model_probe.py`
  - `debug_model_probe2.py`
  - `debug_pedal_evaluation.py`
  - `debug_pedal_gt.py`
- Removed obsolete artifacts from `scripts/`:
  - `debug_eval_output.txt` - large captured evaluation log output.
  - `temp_smoke_test.py` - one-off smoke check superseded by pytest coverage.

If a diagnostic becomes part of a supported workflow, promote it back into `scripts/`
with clear CLI arguments and documentation. Otherwise keep it archived or recover older
versions from Git history when needed.

## Quick Reference

### Environment setup and tests

```bash
conda env create -f environment.yml
conda activate onsvel
python -m pytest tests -q
```

### Training

```bash
python scripts/02_train_pedal_model.py
```

### Evaluation

```bash
python scripts/03_evaluate_pedal_model.py
python scripts/04_evaluate_test_split.py
```

### Log analysis and visualization

```bash
python scripts/05_analyze_training_logs.py LOG_PATH="out/txt_logs/YOUR_LOG.json"
python scripts/06_visualize_pedal_predictions.py SNAPSHOT_INPATH="out/model_snapshots/YOUR_MODEL.torch"
```

### Transcription

```bash
python scripts/transcribe.py --help
```

### Web app

```bash
python web_app/app.py
```