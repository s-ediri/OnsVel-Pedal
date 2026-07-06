# OnV+Pedal: Pedal-Aware Piano Onset, Velocity, and Sustain Pedal Prediction

<p align="center">
<img src="assets/qualitative_plot_bone_small.png" alt="Pedal-aware piano transcription input/output example" width="60.0%"/>
</p>

This repository contains a pedal-focused adaptation of a piano transcription pipeline for predicting onsets, velocities, and sustain pedal events from audio. It builds on the original onset/velocity training framework from the iamusica_training project, but it has been substantially reworked and extended to make sustain-pedal prediction a first-class objective.

The project is designed to:
* install the required software dependencies
* download and preprocess the required datasets
* train and evaluate models for onset, velocity, and pedal prediction
* analyze checkpoints and generate qualitative visualizations

This repository is intentionally differentiated by its focus on sustain-pedal detection, which is treated as a core prediction target rather than an auxiliary add-on.

## What makes this project unique

- It is a pedal-aware extension of piano transcription, with sustain-pedal events modeled alongside onsets and velocities.
- The workflow is oriented toward evaluating and improving pedal prediction quality, not only note transcription.
- The repository structure and documentation are tailored for a pedal-focused training and evaluation pipeline.

Credit and attribution:
* The original onset/velocity model architecture and training workflow were inspired by the iamusica_training project by Andrés Fernández Rodríguez and collaborators.
* The pedal-prediction extensions, repository structure, and current evaluation workflow in this project are original to this adaptation.

This is [Free/Libre and Open Source Software](https://www.gnu.org/philosophy/floss-and-foss.en.html), see the [LICENSE](LICENSE) for more details. If you use this adapted work, please also credit the original paper: [Onsets and Velocities: Affordable Real-Time Piano Transcription Using Convolutional Neural Networks](https://arxiv.org/abs/2303.04485)

```
@inproceedings{onsvel,
      title={{Onsets and Velocities}: Affordable Real-Time Piano Transcription Using Convolutional Neural Networks},
      author={Andres Fernandez},
      year={2023},
      booktitle={{EUSIPCO} Proceedings},
}
```





---

# Software dependencies

This project uses `PyTorch` and provides a reproducible Windows Conda environment at the repository root: [`environment.yml`](../environment.yml). The environment is pinned to Python 3.9 with a CPU-only PyTorch 1.11 / torchaudio 0.11 stack so that setup and smoke tests do not depend on a local CUDA installation.

From the repository root (`OnsVel-Pedal`), create and validate the environment with:

```bash
conda env create -f environment.yml
conda activate onsvel
python -m pytest tests -q
```

If the environment already exists and `environment.yml` changes, update it with:

```bash
conda env update -f environment.yml --prune
conda activate onsvel
python -m pytest tests -q
```

For the most reproducible reset, especially after experimenting with `pip install` inside `onsvel`, remove and recreate the environment instead of updating it in place:

```bash
conda env remove -n onsvel
conda env create -f environment.yml
conda activate onsvel
python -m pytest tests -q
```

The fallback [`requirements.txt`](../requirements.txt) remains available for pip-only workflows, but the Conda file is the recommended setup path on Windows because it manages PyTorch and scientific binary dependencies consistently.

Audio upload decoding uses `pydub` plus `ffmpeg` for MP3 and other non-WAV formats. The Conda environment installs `ffmpeg` for you. If you run a pip-only setup on Python 3.13 or newer and see `No module named 'pyaudioop'`, install the Python 3.13 `audioop` compatibility package with `python -m pip install audioop-lts`, or switch back to the supported `onsvel` Conda environment.

`environment.yml` installs this project in editable mode (`-e .`) after the pinned Conda packages are present. That keeps direct commands such as `python scripts/03_evaluate_pedal_model.py` working from the repository root and lets local source edits take effect immediately. If you later need to refresh only the editable project install, activate `onsvel` and use:

```bash
python -m pip install --no-deps --no-build-isolation -e .
```

Avoid ad-hoc `pip install --upgrade ...` commands inside `onsvel`; they can leave pip-installed packages that `conda env update --prune` may not downgrade. Use the clean reset commands above whenever you need to return to the pinned environment.

> **GPU training note:** `environment.yml` intentionally uses `cpuonly` for reproducible setup and testing. For CUDA training, replace `cpuonly` with the PyTorch CUDA package that matches your Windows GPU driver, then run `conda env update -f environment.yml --prune`.






---

# Data downloading

For this project, training and evaluation is done using the [MAESTRO](https://magenta.tensorflow.org/datasets/maestro) dataset. Specifically, we focus on the latest version, `MAESTROv3`. The full dataset can be readily downloaded at the provided link, and the file structure is expected to end up looking like this:

```
MAESTROv3 ROOT PATH
├── LICENSE
├── maestro-v3.0.0.csv
├── maestro-v3.0.0.json
├── README
├── 2004
├── 2006
├── 2008
├── 2009
├── 2011
├── 2013
├── 2014
├── 2015
├── 2017
└── 2018
```

Where each of the `20xx` directories contains `wav` files with their corresponding `midi` annotations, making a total of 2552 files.

### Downloading other supported datasets:

To ensure compatibility with prior literature, this repository also provides functionality for `MAESTROv1` and `MAESTROv2` (the procedure for those is analogous to v3).

Furthermore, it also provides all functionality needed to use the [MAPS](https://hal.inria.fr/inria-00544155/document) dataset. To download it,

1. Request user and password here: https://adasp.telecom-paris.fr/resources/2010-07-08-maps-database/
2. Download e.g. via: `wget -r --ask-password --user="<YOUR EMAIL>" ftp://ftps.tsi.telecom-paristech.fr/share/maps/`
3. Merge partial zips into folders containing wavs, midis and txt files

For MAPS, the result should end up looking like this (9 folders with 11445 files each):

```
MAPS ROOT PATH
├── license.txt
├── MAPS_doc.pdf
├── MD5SUM
├── readme.txt
├── AkPnBcht
|   ├── ISOL
|   ├── MUS
│   ├── RAND
│   └── UCHO
├── AkPnBsdf
│   ├── ISOL ...
│   ├── MUS  ...
│   ├── RAND ...
│   └── UCHO ...
...
```



---

# Data preprocessing

To train the model, we represent the audio as log-mel spectrograms and the annotations as piano rolls (see [paper](https://arxiv.org/abs/2303.04485) for details). To speed up training and avoid redundant computations, we preprocess the full datasets ahead of time into [HDF5](https://www.h5py.org/) files.

Assuming `MAESTROv3` is in `datasets/maestro/maestro-v3.0.0`, preprocessing with the default parameters can be done by simply calling the following script:

```
python scripts/00_prepare_maestro_hdf5.py
```

Which will generate the `logmels` and `roll` inside the provided `OUTPUT_DIR` (default: `datasets`). Processing MAESTRO with our default parameters takes about 30min on a mid-end 16-core CPU; the piano roll HDF5 file takes about 0.5GB of space, and the log-mel file about 22.5GB.

> :warning: **onset/offset collision**:
> Note that creating piano rolls from MIDI requires to time-quantize the events. If the time resolution is too low, it could happen that two events for the same note end up in the same "bin", and therefore ignored. Another possible explanation is that the MIDI file includes redundant/inconsistent messages, which are also ignored.
> During the preprocessing of MAESTRO/MAPS we can expect quite a few of those to happen, most likely due to the latter reason. We can ignore them, since we don't use piano rolls for evaluation.



### Preprocessing other supported datasets:

The script also allows to precompute former maestro versions:

```
python scripts/00_prepare_maestro_hdf5.py MAESTRO_VERSION=1 MAESTRO_INPATH=datasets/maestro/maestro-v1.0.0
python scripts/00_prepare_maestro_hdf5.py MAESTRO_VERSION=2 MAESTRO_INPATH=datasets/maestro/maestro-v2.0.0
```

To precompute MAPS with default parameters (assuming it is inside `datasets/MAPS`):

```
python scripts/01_prepare_maps_hdf5.py
```

Processing `MAPS` with the default settings takes about 20min on a 16-core CPU. The piano roll HDF5 file takes about 100MB of space, and the log-mel file about 4GB.








---

# Running and evaluating the pedal-aware model

This repository supports evaluating trained `.torch` model checkpoints. Checkpoints are binary artifacts and should be treated as release/download artifacts, not regular source files: new `.torch` files are ignored by Git, should not be committed directly, and should not be moved to Git LFS for normal development. To share a selected pretrained model, publish it as a versioned release asset or another documented download, then record the URL, expected local path, and checksum/metric metadata.

Place a downloaded or locally trained checkpoint under `out/model_snapshots/` and evaluate it with:



```
python scripts/03_evaluate_pedal_model.py SNAPSHOT_INPATH=out/model_snapshots/YOUR_MODEL.torch
```

Yielding the following results after a few minutes:


```
                           PRECISION   RECALL    F1
ONSETS (t=0.74, s=-0.01)   0.985842    0.950764  0.967756
ONS+VEL (t=0.74, s=-0.01)  0.962538    0.928580  0.945033
```



---

# Training the model

For adequate training, a GPU with at least 8GB of memory is sufficient. The following command trains a model from scratch on `MAESTROv3`:

```
python scripts/02_train_pedal_model.py
```

The following is an excerpt from the default configuration that led to the results reported in our paper:

```
"OUTPUT_DIR": "out",
"MAESTRO_PATH": "datasets/maestro/maestro-v3.0.0",
"MAESTRO_VERSION": 3,
"HDF5_MEL_PATH": "datasets/MAESTROv3_logmel_sr=16000_stft=2048w384h_mel=229(50-8000).h5",
"HDF5_ROLL_PATH": "datasets/MAESTROv3_roll_quant=0.024_midivals=128_extendsus=True.h5",
"TRAIN_BS": 40,
"TRAIN_BATCH_SECS": 5.0,
"DATALOADER_WORKERS": 8,
"CONV1X1": [200, 200],
"LR_MAX": 0.008,
"LR_WARMUP": 0.5,
"LR_PERIOD": 1000,
"LR_DECAY": 0.975,
"LR_SLOWDOWN": 1.0,
"MOMENTUM": 0.95,
"WEIGHT_DECAY": 0.0003,
"BATCH_NORM": 0.95,
"DROPOUT": 0.15,
"LEAKY_RELU_SLOPE": 0.1,
"ONSET_POSITIVES_WEIGHT": 8.0,
"VEL_LOSS_LAMBDA": 10.0,
"XV_THRESHOLDS": [0.7, 0.725, 0.75, 0.775, 0.8],
"XV_TOLERANCE_SECS": 0.05,
"XV_TOLERANCE_VEL": 0.1
```

The model is periodically cross-validated and saved under `OUTPUT_DIR`, for further usage and analysis. The script also produces a log in the form or one JSON object per line (see below for an automated way to inspect the log).


### Log inspection

Since the log is a collection of JSON objects, its processing can be easily streamlined. The following script is an example, plotting the cross-validation metrics and fetching the maximum (requires `matplotlib`):

```
python scripts/05_analyze_training_logs.py PLOT_RANGE="[0.90, 0.97]" LOG_PATH=<...>
```


### Debugging/inspection during training

This repo also provides the possibility to pause the training script at arbitrary points, articulated through the [breakpoint.json](breakpoint.json) file, expected to be in the following JSON format:

```
{"inconditional": false,
 "step_gt": null,
 "step_every": null}
```

At every training step, after the loss is computed and before the backward pass and optimization step, the training script checks the contents of the JSON file:

* If `inconditional` is set to `true`, a `breakpoint()` will be called (otherwise ignore)
* If `step_gt` is an integer, `breakpoint()` if the current step is greater than the given integer (otherwise ignore).
* If the contents can't be understood, the file is ignored and training progresses

Note that the default is simply to ignore this file, and to stop the training, the user can e.g. open the file, set `inconditional` to `true`, and save. Then, the training script pauses and the state can be inspected. To resume training, set the value to `false`, save, and press `c` to continue with the process, as explained [here](https://docs.python.org/3/library/pdb.html).




---

# Plot examples

The qualitative plot used in the [paper](https://arxiv.org/abs/2303.04485) can be reproduced with the following command:


```
python scripts/06_visualize_pedal_predictions.py SNAPSHOT_INPATH=out/model_snapshots/YOUR_MODEL.torch OUTPUT_DIR=out
```
