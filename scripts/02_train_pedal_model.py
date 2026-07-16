#!/usr/bin/env python
# -*- coding:utf-8 -*-


"""
This module instantiates, trains and cross-validates a pedal-aware deep learning
model for piano onset, velocity, and sustain-pedal prediction on the MAESTRO
dataset.

It is structured in 3 parts:
1. Fetching and preparing global parameters
2. Instantiating required parts (dataloader, model, decoder, optimizer...)
3. Training loop, featuring an inner cross-validation loop for note and pedal metrics
"""

import gc
import os
import random
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# For omegaconf
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd
import psutil
import torch

#
from omegaconf import OmegaConf

# Optimization for RTX 2070 SUPER (8GB VRAM)
torch.backends.cudnn.benchmark = True  # Enable cuDNN auto-tuner for faster training
torch.backends.cuda.matmul.allow_tf32 = (
    True  # Allow TensorFloat32 for faster matrix ops
)
# Reduce memory fragmentation
if torch.cuda.is_available():
    torch.cuda.empty_cache()
from ov_piano import PIANO_MIDI_RANGE, HDF5PathManager
from ov_piano.custom_logging import JsonColorLogger
from ov_piano.data.maestro import (
    MelMaestro,
    MelMaestroChunks,
    MetaMAESTROv1,
    MetaMAESTROv2,
    MetaMAESTROv3,
)
from ov_piano.data.midi import MidiToPianoRoll
from ov_piano.data.pedal import sustain_pedal_targets_from_values
from ov_piano.eval import GtLoaderMaestro, eval_note_events, pedal_grid_search
from ov_piano.inference import (
    OnsetVelocityNmsDecoder,
    model_outputs_to_probabilities,
    strided_inference,
)
from ov_piano.models.ov import OnsetsAndVelocities
from ov_piano.optimizers import AdamWR
from ov_piano.utils import (
    MaskedBCEWithLogitsLoss,
    ModelSaver,
    breakpoint_json,
    format_load_model_warnings,
    load_model,
    load_resume_state,
    save_resume_state,
    set_seed,
)

# import matplotlib.pyplot as plt


# ##############################################################################
# # MEMORY UTILITIES
# ##############################################################################
def cleanup_memory(verbose=False):
    """
    Aggressive memory cleanup for Windows systems
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    if verbose:
        try:
            process = psutil.Process()
            mem_info = process.memory_info()
            print(
                f"RSS Memory: {mem_info.rss / 1024 / 1024:.1f} MB, "
                f"VMS Memory: {mem_info.vms / 1024 / 1024:.1f} MB"
            )
        except:
            pass


# ##############################################################################
# # GLOBALS
# ##############################################################################
@dataclass
class ConfDef:
    """
    :cvar str DEVICE: For the PyTorch operations. Can be ``cpu`` or ``cuda``
      if a GPU is present. GPU is highly recommended.
    :cvar MAESTRO_PATH: Path to the root directory of the MAESTRO version
    :cvar int MAESTRO_VERSION: Currently 1, 2, 3 supported. 3 recommended.
    :cvar str OUTPUT_DIR: Where to store model snapshots and text logs.
      Created if non-existing.

    :cvar HDF5_MEL_PATH: Path to the HDF5 mel file previously generated.
    :cvar HDF5_ROLL_PATH: Path to the HDF5 piano roll file previously
      generated, must be compatible with the corresponding mel file.
    :cvar SNAPSHOT_INPATH: Optional input path to a pre-trained model, used
      to intialize and resume training from.

    :cvar TRAIN_BS: Train batch size. Reduce if insufficient memory.
    :cvar TRAIN_BATCH_SECS: Time duration of the chunks used for training,
      reduce if insufficient memory.

    :cvar OPTIMIZER: Supported are SGDR and AdamWR (default)
    :cvar LR_MAX: Initial learning rate for the optimizer
    :cvar LR_PERIOD: Number of steps per LR cycle for the optimizer
    :cvar LR_DECAY: Each LR cycle, the max and min LR are multiplied by this
    :cvar LR_SLOWDOWN: Each LR cycle, the duration is multiplied by this
    :cvar MOMENTUM: Gradient momentum for the optimizer
    :cvar WEIGHT_DECAY: L2 regularization factor for the optimizer

    :cvar BATCH_NORM: Momentum for the (batch, spectral) normalization layers
    :cvar DROPOUT: Probability of dropping a weight
    :cvar LEAKY_RELU_SLOPE: Slope for the negative part of leaky ReLU

    :cvar ONSET_POSITIVES_WEIGHT: The loss function for the piano rolls will
      multiply the positive examples by this constant (used to compensate the
      fact that onsets are less than 50% of frames)
    :cvar VEL_LOSS_LAMBDA: total loss is
      ``onset_loss + LAMBDA * velocity loss`` for this lambda.
    :cvar TRAINABLE_ONSETS: If false, only the velocity-specific parameters are
      being trained. Useful e.g. for fine-tuning a model that already performs
      good onset detection.

    :cvar DECODER_GAUSS_STD: The decoder on top of the DNN predictions performs
      a Gaussian time-convolution to smoothen detections. This is the standard
      deviation, in time-frames.
    :cvar DECODER_GAUSS_KSIZE: The window size, in time-frames, for the
      smoothening Gaussian time-convolution.

    :cvar XV_TOLERANCE_SECS: The maximum absolute error between onset pred
      and ground truth, in seconds, to consider the prediction correct. Used
      during cross-vlaidation
    :cvar XV_TOLERANCE_VEL: The maximum absolute error between velocity pred
      and ground truth, in ratio between 0 and 1, to consider the prediction
      correct. To better understand this ratio, see the official documentation
      for ``mir_eval.transcription_velocity``. Used during cross-validation.

    :cvar XV_CHUNK_SIZE: For cross-validation, full files are processed, which
      may be too large for memory and have to be processed in strided chunks.
      This is the chunk size in seconds, it doesn't affect performance as long
      as it is large enough.
    :cvar XV_CHUNK_OVERLAP: See ``XV_CHUNK_SIZE``. This is the overlap among
      consecutive chunks. It doesn't affect performance as long as it is large
      enough to avoid boundary artifacts.
    :cvar XV_THRESHOLDS: List of thresholds to perform cross-validation on.
      Note that XV will be performed once per threshold, so the more, the
      slower training, but also better chances of assessing performance right.
    """

    # general
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    RANDOM_SEED: Optional[int] = None
    # I/O
    OUTPUT_DIR: str = "out"
    MAESTRO_PATH: str = os.path.join("datasets", "maestro", "maestro-v3.0.0")
    MAESTRO_VERSION: int = 3
    HDF5_MEL_PATH: str = os.path.join(
        "datasets", "MAESTROv3_logmel_sr=16000_stft=2048w384h_mel=229(50-8000).h5"
    )
    HDF5_ROLL_PATH: str = os.path.join(
        "datasets", "MAESTROv3_roll_quant=0.024_midivals=128_extendsus=True.h5"
    )
    SNAPSHOT_INPATH: Optional[str] = None  # Auto-detected below if None
    BASELINE_SNAPSHOT_INPATH: Optional[str] = os.path.join(
        PROJECT_ROOT,
        "out",
        "model_snapshots",
        "OnsetsAndVelocities_2023_03_04_09_53_53.289step=43500_f1=0.9675__0.9480.torch",
    )
    RESUME_FROM_LATEST: bool = False
    # data loader settings tested on the local 16GB RAM / RTX 2070 SUPER setup
    # Use longer chunks so the pedal head sees enough musical context.
    TRAIN_BS: int = 1  # Longer pedal context needs a smaller physical batch
    TRAIN_BATCH_SECS: float = (
        12.0  # Longer context helps phrase-level sustain-pedal decisions
    )
    DATALOADER_WORKERS: int = (
        0  # Windows + HDF5 doesn't support multiprocessing (h5py not pickleable)
    )
    GRADIENT_ACCUMULATION_STEPS: int = 16  # Effective batch size = 1 * 16 = 16
    # model/optimizer
    CONV1X1: List[int] = (200, 200)  # Matches the high-F1 bundled note checkpoint
    # optimizer
    # Keep the original note-training LR available for explicit full-model runs,
    # but use a much smaller default for pedal-head fine-tuning. The pedal head is
    # randomly initialized when starting from the bundled note checkpoint; using
    # the old full-model LR here makes optimization noisy and can encourage users
    # to resume from checkpoints whose note backbone was accidentally degraded by
    # earlier coupled training experiments.
    LR_MAX: float = 0.006
    PEDAL_LR_MAX: float = 0.0003
    LR_WARMUP: float = 0.5
    LR_PERIOD: int = 1500  # ~2-3 periods per epoch for good convergence
    LR_DECAY: float = 0.98  # Gentle decay across epochs
    LR_SLOWDOWN: float = 1.0
    MOMENTUM: float = 0.95
    WEIGHT_DECAY: float = 0.0003
    BATCH_NORM: float = 0.95  # Higher momentum helps with small batches
    DROPOUT: float = (
        0.15  # Moderate dropout for regularization without over-regularizing
    )
    LEAKY_RELU_SLOPE: Optional[float] = 0.1
    # loss
    ONSET_POSITIVES_WEIGHT: float = (
        3.0  # Increased - onsets are sparse, need higher weight
    )
    VEL_LOSS_LAMBDA: float = 10.0
    PEDAL_LOSS_LAMBDA: float = 1.0
    PEDAL_POSITIVES_WEIGHT: float = 2.0
    PEDAL_TRANSITION_WEIGHT: float = 6.0
    PEDAL_EVENT_LOSS_LAMBDA: float = 2.0
    PEDAL_EVENT_POSITIVES_WEIGHT: float = 12.0
    PEDAL_TRANSITION_TARGET_WIDTH: int = 1
    MAX_GRAD_NORM: float = 1.0
    DETACH_PEDAL_FEATURES: bool = True
    PEDAL_ONLY_FINETUNE: bool = True
    TRAINABLE_ONSETS: bool = False
    LOG_NOTE_MONITOR_LOSSES: bool = True
    # decoder
    DECODER_GAUSS_STD: float = 1
    DECODER_GAUSS_KSIZE: int = 11
    # training loop
    NUM_EPOCHS: int = (
        8  # 8 epochs balances quality and time (3-5 days on your hardware)
    )
    TRAIN_LOG_EVERY: int = 50  # Log less frequently (was 10)
    XV_EVERY: int = 999999999  # DISABLED: Use separate evaluation script after training
    XV_CHUNK_SIZE: float = 100
    XV_CHUNK_OVERLAP: float = 2.5
    XV_THRESHOLDS: List[float] = (
        0.5,
        0.75,
    )  # Test multiple thresholds for sustain pedal
    # pedal validation/search during training
    PEDAL_VALIDATION_EVERY: int = 1000
    PEDAL_VALIDATION_TAKE_ONE_EVERY: int = 20
    PEDAL_SEARCH_THRESHOLDS: List[float] = (
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
    )
    PEDAL_SEARCH_HYSTERESIS: List[float] = (0.02, 0.05, 0.1, 0.15)
    PEDAL_SEARCH_SMOOTHING_WINDOWS: List[int] = (1, 3, 5, 7, 11)
    PEDAL_SEARCH_MIN_HOLD_STEPS: List[int] = (1, 2, 4, 8)
    PEDAL_SEARCH_SHIFTS: List[float] = (-0.05, -0.025, 0.0, 0.025, 0.05)
    # xv tolerances
    XV_TOLERANCE_SECS: float = 0.05
    XV_TOLERANCE_VEL: float = 0.1
    # xv shortening
    XV_SHORTEN: bool = False  # Whether to shorten the validation split for faster CV
    XV_SHORTEN_FACTOR: int = (
        5  # Stride factor for shortening (e.g., 5 means keep every 5th sample)
    )


# ##############################################################################
# # MAIN LOOP INITIALIZATION
# ##############################################################################
if __name__ == "__main__":
    CONF = OmegaConf.structured(ConfDef())
    cli_conf = OmegaConf.from_cli()
    CONF = OmegaConf.merge(CONF, cli_conf)

    # Prefer the bundled note model by default so pedal-only fine-tuning starts
    # from high onset/velocity F1. Set RESUME_FROM_LATEST=true or pass
    # SNAPSHOT_INPATH explicitly to continue a previous generated run.
    if CONF.SNAPSHOT_INPATH is None and CONF.RESUME_FROM_LATEST:
        checkpoint_dirs = [os.path.join(CONF.OUTPUT_DIR, "model_snapshots")]
        for checkpoint_dir in checkpoint_dirs:
            if not os.path.isdir(checkpoint_dir):
                continue
            checkpoints = sorted(
                [f for f in os.listdir(checkpoint_dir) if f.endswith(".torch")],
                key=lambda x: os.path.getmtime(os.path.join(checkpoint_dir, x)),
                reverse=True,
            )
            for checkpoint in checkpoints:
                candidate_path = os.path.join(checkpoint_dir, checkpoint)
                if os.path.getsize(candidate_path) < 1024:
                    print(
                        f"[AUTO-RESUME] Skipping tiny or incomplete checkpoint: {candidate_path}"
                    )
                    continue
                try:
                    torch.load(candidate_path, map_location="cpu")
                    CONF.SNAPSHOT_INPATH = candidate_path
                    print(
                        f"[AUTO-RESUME] Found valid checkpoint: {CONF.SNAPSHOT_INPATH}"
                    )
                    break
                except Exception as exc:
                    print(
                        f"[AUTO-RESUME] Skipping invalid checkpoint {candidate_path}: {exc}"
                    )
            if CONF.SNAPSHOT_INPATH is not None:
                break
        if CONF.SNAPSHOT_INPATH is None:
            baseline_path = CONF.BASELINE_SNAPSHOT_INPATH
            if baseline_path and os.path.isfile(baseline_path):
                CONF.SNAPSHOT_INPATH = baseline_path
                print(f"[AUTO-RESUME] Using bundled note baseline: {baseline_path}")
            else:
                print("[AUTO-RESUME] No valid checkpoint found.")
    elif CONF.SNAPSHOT_INPATH is None:
        baseline_path = CONF.BASELINE_SNAPSHOT_INPATH
        if baseline_path and os.path.isfile(baseline_path):
            CONF.SNAPSHOT_INPATH = baseline_path
            print(f"[TRAINING] Using bundled note baseline: {baseline_path}")

    # if no seed is given, take a random one
    if CONF.RANDOM_SEED is None:
        CONF.RANDOM_SEED = random.randint(0, 10_000_000)
    set_seed(CONF.RANDOM_SEED)

    # derivative globals + parse HDF5 filenames and ensure they are consistent
    (DATASET_NAME, SAMPLERATE, WINSIZE, HOPSIZE, MELBINS, FMIN, FMAX) = (
        HDF5PathManager.parse_mel_hdf5_basename(os.path.basename(CONF.HDF5_MEL_PATH))
    )
    roll_params = HDF5PathManager.parse_roll_hdf5_basename(
        os.path.basename(CONF.HDF5_ROLL_PATH)
    )
    SECS_PER_FRAME = HOPSIZE / SAMPLERATE
    CHUNK_LENGTH = round(CONF.TRAIN_BATCH_SECS / SECS_PER_FRAME)
    CHUNK_STRIDE = round(CHUNK_LENGTH / CONF.TRAIN_BATCH_SECS)
    #
    assert DATASET_NAME == roll_params[0], "Inconsistent HDF5 datasets?"
    assert SECS_PER_FRAME == roll_params[1], "Inconsistent roll quantization?"
    #
    XV_CHUNK_SIZE = round(CONF.XV_CHUNK_SIZE / SECS_PER_FRAME)
    XV_CHUNK_OVERLAP = round(CONF.XV_CHUNK_OVERLAP / SECS_PER_FRAME)
    #
    METAMAESTRO_CLASS = {1: MetaMAESTROv1, 2: MetaMAESTROv2, 3: MetaMAESTROv3}[
        CONF.MAESTRO_VERSION
    ]
    # output dirs
    MODEL_SNAPSHOT_OUTDIR = os.path.join(CONF.OUTPUT_DIR, "model_snapshots")
    TXT_LOG_OUTDIR = os.path.join(CONF.OUTPUT_DIR, "txt_logs")
    os.makedirs(MODEL_SNAPSHOT_OUTDIR, exist_ok=True)
    os.makedirs(TXT_LOG_OUTDIR, exist_ok=True)

    txt_logger = JsonColorLogger(f"[{os.path.basename(__file__)}]", TXT_LOG_OUTDIR)
    txt_logger.loj("PARAMETERS", OmegaConf.to_container(CONF))

    # Load resume state only when resuming from a generated training checkpoint.
    snapshot_abs = os.path.abspath(CONF.SNAPSHOT_INPATH) if CONF.SNAPSHOT_INPATH else ""
    snapshot_dir_abs = os.path.abspath(MODEL_SNAPSHOT_OUTDIR)
    should_load_resume_state = snapshot_abs.startswith(snapshot_dir_abs + os.sep)
    resume_state = load_resume_state(MODEL_SNAPSHOT_OUTDIR) if should_load_resume_state else None
    resume_epoch = 1
    resume_batch_idx = 0
    resume_global_step = 1
    if resume_state is not None:
        resume_epoch = resume_state["epoch"]
        resume_batch_idx = resume_state["batch_idx"]
        resume_global_step = resume_state["global_step"]
        txt_logger.loj(
            "RESUME_STATE",
            {
                "epoch": resume_epoch,
                "batch_idx": resume_batch_idx,
                "global_step": resume_global_step,
            },
        )
    else:
        txt_logger.loj("RESUME_STATE", "No previous state found, starting fresh")

    # datasets and dataloaders
    metamaestro_train = METAMAESTRO_CLASS(
        CONF.MAESTRO_PATH, splits=["train"], years=METAMAESTRO_CLASS.ALL_YEARS
    )
    maestro_train = MelMaestroChunks(
        CONF.HDF5_MEL_PATH,
        CONF.HDF5_ROLL_PATH,
        CHUNK_LENGTH,
        CHUNK_STRIDE,
        *(x[0] for x in metamaestro_train.data),
        with_oob=True,
        logmel_oob_pad_val="min",
        as_torch_tensors=False,
    )
    train_dl = torch.utils.data.DataLoader(
        maestro_train,
        batch_size=CONF.TRAIN_BS,
        shuffle=True,
        num_workers=CONF.DATALOADER_WORKERS,
        pin_memory=False,
        persistent_workers=False,
    )
    #
    metamaestro_xv = METAMAESTRO_CLASS(
        CONF.MAESTRO_PATH, splits=["validation"], years=METAMAESTRO_CLASS.ALL_YEARS
    )
    # Conditionally shorten xv set to speed up cross validation times
    if CONF.XV_SHORTEN:
        txt_logger.loj(
            "WARNING",
            f"shortening xv split (factor={CONF.XV_SHORTEN_FACTOR}) for faster crossvalidation!",
        )
        metamaestro_xv.data = metamaestro_xv.data[:: CONF.XV_SHORTEN_FACTOR]
    #
    maestro_xv = MelMaestro(
        CONF.HDF5_MEL_PATH,
        CONF.HDF5_ROLL_PATH,
        *(x[0] for x in metamaestro_xv.data),
        as_torch_tensors=False,
    )
    try:
        xv_gt_loader = GtLoaderMaestro(maestro_xv, metamaestro_xv)
    except Exception as e:
        txt_logger.loj(
            "WARNING",
            f"Could not initialize validation loader: {e}. Skipping validation.",
        )
        xv_gt_loader = None

    # data-specific constants
    batches_per_epoch = len(train_dl)
    num_mels = maestro_train[0][0].shape[0]
    key_beg, key_end = PIANO_MIDI_RANGE
    num_piano_keys = key_end - key_beg

    # DNN (instantiation+serialization)
    model = OnsetsAndVelocities(
        in_chans=2,  # X and time_derivative(X)
        in_height=num_mels,
        out_height=num_piano_keys,
        conv1x1head=CONF.CONV1X1,
        bn_momentum=CONF.BATCH_NORM,
        leaky_relu_slope=CONF.LEAKY_RELU_SLOPE,
        dropout_drop_p=CONF.DROPOUT,
    ).to(CONF.DEVICE)
    if CONF.SNAPSHOT_INPATH is not None:
        load_report = load_model(model, CONF.SNAPSHOT_INPATH, eval_phase=False, strict=False)
        pedal_key_prefixes = (
            "pedal_stage.",
            "pedal_state_head.",
            "pedal_onset_head.",
            "pedal_offset_head.",
        )
        missing_pedal_keys = [
            key for key in load_report["missing_keys"]
            if key.startswith(pedal_key_prefixes)
        ]
        ignored_missing_prefixes = pedal_key_prefixes if CONF.PEDAL_ONLY_FINETUNE else ()
        if missing_pedal_keys and CONF.PEDAL_ONLY_FINETUNE:
            txt_logger.loj(
                "PEDAL_STAGE_INIT",
                {
                    "checkpoint": CONF.SNAPSHOT_INPATH,
                    "missing_pedal_keys": len(missing_pedal_keys),
                    "action": (
                        "Initialized pedal_stage with fresh random weights. "
                        "This is expected when fine-tuning pedals from the "
                        "bundled note-only baseline checkpoint."
                    ),
                },
            )
        for warning in format_load_model_warnings(
            load_report,
            ignored_missing_key_prefixes=ignored_missing_prefixes,
        ):
            txt_logger.loj("CHECKPOINT_LOAD_WARNING", warning)
    elif CONF.PEDAL_ONLY_FINETUNE:
        raise RuntimeError(
            "PEDAL_ONLY_FINETUNE=true requires a valid note checkpoint. "
            "No SNAPSHOT_INPATH/BASELINE_SNAPSHOT_INPATH was found, so refusing "
            "to freeze a randomly initialized onset/velocity backbone. Either "
            "set SNAPSHOT_INPATH to a high-F1 note checkpoint, place the bundled "
            "checkpoint under out/model_snapshots/, or run explicit full training "
            "with PEDAL_ONLY_FINETUNE=false."
        )

    frozen_note_modules = (
        model.specnorm,
        model.stem,
        model.onset_stages,
        model.velocity_stage,
    )
    pedal_modules = model.pedal_modules()
    if CONF.PEDAL_ONLY_FINETUNE:
        for module in frozen_note_modules:
            module.eval()
            for param in module.parameters():
                param.requires_grad = False
        for param in model.pedal_parameters():
            param.requires_grad = True
        txt_logger.loj(
            "TRAINING_MODE",
            {
                "mode": "pedal_only_finetune",
                "note_backbone": "frozen_eval_mode",
                "pedal_features": "detached",
                "effective_lr_max": CONF.PEDAL_LR_MAX,
                "reason": (
                    "Preserve onset/velocity F-score while specializing only "
                    "the sustain-pedal head."
                ),
            },
        )
    elif CONF.DETACH_PEDAL_FEATURES:
        txt_logger.loj(
            "TRAINING_MODE",
            {
                "mode": "joint_training_detached_pedal",
                "note_backbone": "trainable" if CONF.TRAINABLE_ONSETS else "onsets_frozen_velocity_trainable",
                "pedal_features": "detached",
                "warning": (
                    "Joint training can change note metrics. Use only after "
                    "pedal-only fine-tuning has been evaluated."
                ),
            },
        )
    model_saver = ModelSaver(
        model,
        MODEL_SNAPSHOT_OUTDIR,
        log_fn=lambda msg: txt_logger.loj("SAVED_MODEL", msg),
    )

    # Wrapper to save resume state whenever model is saved (LR cycle hook)
    def cycle_end_with_resume_state(suffix=None):
        """Save model checkpoint and update resume state."""
        model_saver(suffix)
        # Note: resume_state will be updated in training loop, this just acts as hook

    # decoder
    decoder = OnsetVelocityNmsDecoder(
        num_piano_keys,
        nms_pool_ksize=3,
        gauss_conv_stddev=CONF.DECODER_GAUSS_STD,
        gauss_conv_ksize=CONF.DECODER_GAUSS_KSIZE,
        vel_pad_left=1,
        vel_pad_right=1,
    )  # this module stays on cpu

    # loss
    ons_pos_weights = torch.FloatTensor([CONF.ONSET_POSITIVES_WEIGHT]).to(CONF.DEVICE)
    ons_loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=ons_pos_weights)
    vel_loss_fn = MaskedBCEWithLogitsLoss()
    # Pedal loss with class weighting for sparse pedal presence
    pedal_pos_weight = torch.FloatTensor([CONF.PEDAL_POSITIVES_WEIGHT]).to(CONF.DEVICE)

    # optimizer
    if CONF.PEDAL_ONLY_FINETUNE:
        trainable_params = list(model.pedal_parameters())
    elif CONF.TRAINABLE_ONSETS:
        trainable_params = list(model.parameters())
    else:
        trainable_params = (
            list(model.velocity_stage.parameters())
            + list(model.pedal_parameters())
        )
    trainable_param_count = sum(p.numel() for p in trainable_params if p.requires_grad)
    frozen_param_count = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    txt_logger.loj(
        "PARAMETER_COUNTS",
        {"trainable": trainable_param_count, "frozen": frozen_param_count},
    )
    if CONF.PEDAL_ONLY_FINETUNE:
        frozen_note_param_count = sum(
            p.numel() for module in frozen_note_modules for p in module.parameters()
            if not p.requires_grad
        )
        assert frozen_note_param_count > 0, "Expected note backbone parameters to be frozen"
        assert all(
            not p.requires_grad for module in frozen_note_modules for p in module.parameters()
        ), "PEDAL_ONLY_FINETUNE must not update onset/velocity parameters"

    effective_lr_max = CONF.PEDAL_LR_MAX if CONF.PEDAL_ONLY_FINETUNE else CONF.LR_MAX
    opt_hpars = {
        "lr_max": effective_lr_max,
        "lr": effective_lr_max,
        "lr_period": CONF.LR_PERIOD,
        "lr_decay": CONF.LR_DECAY,
        "lr_slowdown": CONF.LR_SLOWDOWN,
        "cycle_end_hook_fn": cycle_end_with_resume_state,
        "cycle_warmup": CONF.LR_WARMUP,
        "weight_decay": CONF.WEIGHT_DECAY,
        "betas": (0.9, 0.999),
        "eps": 1e-8,
        "amsgrad": False,
    }
    opt = AdamWR(trainable_params, **opt_hpars)

    def keep_frozen_note_modules_eval():
        if CONF.PEDAL_ONLY_FINETUNE:
            for module in frozen_note_modules:
                module.eval()

    def training_forward(x):
        if CONF.PEDAL_ONLY_FINETUNE:
            with torch.no_grad():
                onset_stages, stem_out = model.forward_onsets(x)
                features = torch.cat([stem_out, onset_stages[-1].unsqueeze(1)], dim=1)
                velocities = model.velocity_stage(features).squeeze(1)
            pedals = model.forward_pedals(features.detach())
            return onset_stages, velocities, pedals
        if CONF.TRAINABLE_ONSETS:
            onset_stages, stem_out = model.forward_onsets(x)
            features = torch.cat([stem_out, onset_stages[-1].unsqueeze(1)], dim=1)
        else:
            with torch.no_grad():
                onset_stages, stem_out = model.forward_onsets(x)
                features = torch.cat([stem_out, onset_stages[-1].unsqueeze(1)], dim=1)
        velocities = model.velocity_stage(features).squeeze(1)
        pedal_features = features.detach() if CONF.DETACH_PEDAL_FEATURES else features
        pedals = model.forward_pedals(pedal_features)
        return onset_stages, velocities, pedals

    # ##########################################################################
    # # XV HELPERS
    # ##########################################################################
    def model_inference(x):
        """
        Convenience wrapper around the DNN to ensure output and input sequences
        have same length. Model now returns (onsets, velocities, pedals).
        """
        return model_outputs_to_probabilities(model(x), include_pedals=False)

    def pedal_model_inference(x):
        """Model inference wrapper that includes sustain-pedal probabilities."""
        return model_outputs_to_probabilities(model(x), include_pedals=True)

    def xv_file(mel, md, thresholds=[0.5], verbose=False):
        """
        Convenience function to perform cross-validation on a single file:
        1. Loads ground-truth event sequence from given MIDI
        2. Performs strided inference on given mel, and extracts predicted
          event sequence
        3. Computes XV metrics for every given threshold, once for onsets only
          and once for onsets+velocities
        4. Returns ``(o_results, ov_results)`` as lists with one element per
          threshold
        """
        # gather ground truth
        df_gt = xv_gt_loader(md)[0]
        # gather onset predictions
        tmel = torch.from_numpy(mel).to(CONF.DEVICE).unsqueeze(0)
        onset_pred, vel_pred = strided_inference(
            model_inference, tmel, XV_CHUNK_SIZE, XV_CHUNK_OVERLAP
        )
        del tmel
        df_pred = decoder(onset_pred, vel_pred, pthresh=min(thresholds))
        # evaluate for all thresholds, without taking velocity into account
        results = []
        for t in thresholds:
            # prob must be above threshold, unless velocity score high enough
            df_pred_t = df_pred[df_pred["prob"] >= t]
            # evaluate
            prec, rec, f1 = eval_note_events(
                df_gt["onset"].to_numpy(),
                df_gt["key"].to_numpy(),
                df_pred_t["t_idx"].to_numpy(),
                df_pred_t["key"].to_numpy(),
                #
                tol_secs=CONF.XV_TOLERANCE_SECS,
                pitch_tolerance=0.1,
                pred_key_shift=key_beg,
                pred_onset_mul=SECS_PER_FRAME,
                pred_shift=0,
            )
            results.append((md[0], prec, rec, f1))
            if verbose:
                txt_logger.loj(
                    "XV_ONSET", {"threshold": t, "P": prec, "R": rec, "F1": f1}
                )
        # evaluate for all thresholds, taking velocity into account
        results_vel = []
        for t in thresholds:
            # threshold predictions
            df_pred_t = df_pred[df_pred["prob"] >= t]
            # evaluate
            prec, rec, f1 = eval_note_events(
                df_gt["onset"].to_numpy(),
                df_gt["key"].to_numpy(),
                df_pred_t["t_idx"].to_numpy(),
                df_pred_t["key"].to_numpy(),
                #
                gt_vels=df_gt["vel"].to_numpy(),
                pred_vels=df_pred_t["vel"].to_numpy(),
                #
                tol_secs=CONF.XV_TOLERANCE_SECS,
                pitch_tolerance=0.1,
                velocity_tolerance=CONF.XV_TOLERANCE_VEL,
                pred_key_shift=key_beg,
                pred_onset_mul=SECS_PER_FRAME,
                pred_shift=0,
            )
            results_vel.append((md[0], prec, rec, f1))
            if verbose:
                txt_logger.loj(
                    "XV_ONSET_VEL", {"threshold": t, "P": prec, "R": rec, "F1": f1}
                )
        #
        return results, results_vel

    def run_pedal_validation(epoch, batch_idx, global_step):
        """Evaluate sustain-pedal F1 on the validation split and tune decoder knobs."""
        if xv_gt_loader is None:
            txt_logger.loj(
                "PEDAL_VALIDATION_SKIP",
                {"reason": "validation ground-truth loader is unavailable"},
            )
            return None

        model.eval()
        cleanup_memory(verbose=False)
        pedal_eval_items = []
        len_xv = len(maestro_xv)
        take_every = max(1, int(CONF.PEDAL_VALIDATION_TAKE_ONE_EVERY))

        with torch.no_grad():
            for ii, (mel, roll, md) in enumerate(maestro_xv, 1):
                if take_every > 1 and ((ii - 1) % take_every) != 0:
                    continue
                txt_logger.loj(
                    "PEDAL_VALIDATION_PROCESSING",
                    {
                        "idx": ii,
                        "len_xv": len_xv,
                        "take_one_every": take_every,
                        "filename": md[0],
                    },
                )
                try:
                    tmel = torch.from_numpy(mel).to(CONF.DEVICE).unsqueeze(0)
                    if tmel.shape[-1] == 0:
                        txt_logger.loj(
                            "PEDAL_VALIDATION_SKIP_FILE",
                            {"filename": md[0], "reason": "empty mel input"},
                        )
                        continue

                    outputs = strided_inference(
                        pedal_model_inference, tmel, XV_CHUNK_SIZE, XV_CHUNK_OVERLAP
                    )
                    if not outputs or len(outputs) < 3 or outputs[2] is None:
                        txt_logger.loj(
                            "PEDAL_VALIDATION_SKIP_FILE",
                            {"filename": md[0], "reason": "missing pedal output"},
                        )
                        continue

                    pedal_pred = outputs[2]
                    if pedal_pred.dim() == 2:
                        pedal_pred = pedal_pred.unsqueeze(0)
                    gt_pedal_df = xv_gt_loader.get_sus_pedal_events(md, SECS_PER_FRAME)
                    pedal_eval_items.append((gt_pedal_df, pedal_pred))

                except Exception as exc:
                    txt_logger.loj(
                        "PEDAL_VALIDATION_FILE_ERROR",
                        {"filename": md[0], "error": str(exc)},
                    )
                finally:
                    for name in ("tmel", "outputs", "pedal_pred"):
                        if name in locals():
                            try:
                                del locals()[name]
                            except Exception:
                                pass
                    del mel, roll, md
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

        if len(pedal_eval_items) == 0:
            txt_logger.loj(
                "PEDAL_VALIDATION_SKIP",
                {"reason": "no validation files produced pedal predictions"},
            )
            return None

        txt_logger.loj(
            "PEDAL_VALIDATION_SEARCH_START",
            {
                "epoch": epoch,
                "batch_idx": batch_idx,
                "global_step": global_step,
                "num_files": len(pedal_eval_items),
                "thresholds": list(CONF.PEDAL_SEARCH_THRESHOLDS),
                "hysteresis": list(CONF.PEDAL_SEARCH_HYSTERESIS),
                "smoothing_windows": list(CONF.PEDAL_SEARCH_SMOOTHING_WINDOWS),
                "min_hold_steps": list(CONF.PEDAL_SEARCH_MIN_HOLD_STEPS),
                "shifts": list(CONF.PEDAL_SEARCH_SHIFTS),
            },
        )
        summary, best_params, best_metrics = pedal_grid_search(
            pedal_eval_items,
            SECS_PER_FRAME,
            thresholds=CONF.PEDAL_SEARCH_THRESHOLDS,
            hysteresis_values=CONF.PEDAL_SEARCH_HYSTERESIS,
            smoothing_windows=CONF.PEDAL_SEARCH_SMOOTHING_WINDOWS,
            min_hold_steps_values=CONF.PEDAL_SEARCH_MIN_HOLD_STEPS,
            shifts=CONF.PEDAL_SEARCH_SHIFTS,
            tol_secs=CONF.XV_TOLERANCE_SECS,
        )
        if best_params is None:
            txt_logger.loj(
                "PEDAL_VALIDATION_SKIP",
                {"reason": "no pedal hyperparameter combinations were evaluated"},
            )
            return None

        top_results = sorted(
            summary.items(), key=lambda item: float(item[1][2]), reverse=True
        )[:10]
        top_results = [
            {
                "threshold": params[0],
                "hysteresis": params[1],
                "smoothing_window": params[2],
                "min_hold_steps": params[3],
                "shift": params[4],
                "precision": float(metrics[0]),
                "recall": float(metrics[1]),
                "f1": float(metrics[2]),
            }
            for params, metrics in top_results
        ]
        best = {
            "threshold": best_params[0],
            "hysteresis": best_params[1],
            "smoothing_window": best_params[2],
            "min_hold_steps": best_params[3],
            "shift": best_params[4],
            "precision": float(best_metrics[0]),
            "recall": float(best_metrics[1]),
            "f1": float(best_metrics[2]),
        }
        txt_logger.loj(
            "PEDAL_VALIDATION_SUMMARY",
            {
                "epoch": epoch,
                "batch_idx": batch_idx,
                "global_step": global_step,
                "num_files": len(pedal_eval_items),
                "best": best,
                "top_results": top_results,
            },
        )
        return best

    # ##########################################################################
    # # TRAINING LOOP
    # ##########################################################################
    txt_logger.loj("MODEL_INFO", {"class": model.__class__.__name__})
    global_step = resume_global_step
    best_pedal_f1 = -1.0
    onsets_beg, onsets_end = maestro_train.ONSETS_RANGE
    frames_beg, frames_end = maestro_train.FRAMES_RANGE

    # Use epoch-seeded DataLoader for reproducible shuffles per epoch
    def get_epoch_dataloader(epoch_num):
        """Create DataLoader with epoch-specific seed for reproducible shuffles."""
        # Set seed to (base_seed + epoch) so each epoch has different but reproducible shuffle
        epoch_seed = CONF.RANDOM_SEED + epoch_num
        set_seed(epoch_seed)
        return torch.utils.data.DataLoader(
            maestro_train,
            batch_size=CONF.TRAIN_BS,
            shuffle=True,
            num_workers=CONF.DATALOADER_WORKERS,
            pin_memory=False,
            persistent_workers=False,
        )

    for epoch in range(resume_epoch, CONF.NUM_EPOCHS + 1):
        # Create fresh DataLoader with epoch-specific seed for reproducibility
        train_dl = get_epoch_dataloader(epoch)

        for batch_idx, (logmels, rolls, metas) in enumerate(train_dl):
            # Skip batches if resuming mid-epoch
            if epoch == resume_epoch and batch_idx < resume_batch_idx:
                continue
            # Reset resume_batch_idx after first epoch (only needed on resume)
            if batch_idx == resume_batch_idx and epoch == resume_epoch:
                txt_logger.loj(
                    "RESUMING",
                    {
                        "epoch": epoch,
                        "batch_idx": batch_idx,
                        "message": f"Resumed from saved state, continuing from batch {batch_idx}",
                    },
                )
            # ##################################################################
            # # CROSS VALIDATION
            # ##################################################################
            if (global_step % CONF.XV_EVERY) == 0:
                model.eval()
                #
                cleanup_memory(verbose=True)
                torch.cuda.empty_cache()
                with torch.no_grad():
                    xv_results = []
                    xv_results_vel = []
                    len_xv = len(maestro_xv)
                    for ii, (mel, roll, md) in enumerate(maestro_xv, 1):
                        txt_logger.loj(
                            "XV_PROCESSING",
                            {"idx": ii, "len_xv": len_xv, "filename": md[0]},
                        )
                        xv_result, xv_result_vel = xv_file(mel, md, CONF.XV_THRESHOLDS)
                        xv_results.append(xv_result)
                        xv_results_vel.append(xv_result_vel)
                        # Aggressive memory cleanup after each file
                        del mel, roll, md
                        gc.collect()
                        torch.cuda.empty_cache()
                # compare non-vel results and report best
                xv_dfs = [
                    (t, pd.DataFrame(x, columns=["filename", "P", "R", "F1"]))
                    for t, x in zip(CONF.XV_THRESHOLDS, zip(*xv_results))
                ]
                f1_avgs = []
                for t, df in xv_dfs:
                    averages = [f"AVERAGES (t={t})", *df.iloc[:, 1:].mean().tolist()]
                    df.loc[len(df)] = averages
                    f1_avgs.append(averages[-1])
                best_f1_idx = np.argmax(f1_avgs)
                best_f1 = f1_avgs[best_f1_idx]
                # compare vel results and report best
                xv_dfs_vel = [
                    (t, pd.DataFrame(x, columns=["filename", "P", "R", "F1"]))
                    for t, x in zip(CONF.XV_THRESHOLDS, zip(*xv_results_vel))
                ]
                f1_avgs_vel = []
                for t, df in xv_dfs_vel:
                    averages = [f"AVERAGES (t={t})", *df.iloc[:, 1:].mean().tolist()]
                    df.loc[len(df)] = averages
                    f1_avgs_vel.append(averages[-1])
                best_f1_idx_vel = np.argmax(f1_avgs_vel)
                best_f1_vel = f1_avgs_vel[best_f1_idx_vel]
                # report results, save model, resume training
                txt_logger.loj("XV_BEST_ONSET", str(xv_dfs[best_f1_idx][1]))
                txt_logger.loj("XV_BEST_ONSET_VEL", str(xv_dfs_vel[best_f1_idx_vel][1]))
                txt_logger.loj(
                    "XV_SUMMARY",
                    {
                        "epoch": epoch,
                        "global_step": global_step,
                        "best_f1_o_thresh": CONF.XV_THRESHOLDS[int(best_f1_idx)],
                        "best_f1_o": best_f1,
                        "best_f1_v_thresh": CONF.XV_THRESHOLDS[int(best_f1_idx_vel)],
                        "best_f1_v": best_f1_vel,
                    },
                )
                model_saver(f"step={global_step}_f1={best_f1:.4f}__{best_f1_vel:.4f}")
                # Save resume state for mid-epoch resumption
                save_resume_state(
                    epoch, batch_idx + 1, global_step, MODEL_SNAPSHOT_OUTDIR
                )
                #
                # Clear XV data and cleanup before resuming training
                del xv_results, xv_results_vel, xv_dfs, xv_dfs_vel
                cleanup_memory(verbose=True)
                torch.cuda.empty_cache()
                model.train()
                keep_frozen_note_modules_eval()

            # ##################################################################
            # # PEDAL VALIDATION + BEST CHECKPOINT
            # ##################################################################
            if (
                CONF.PEDAL_VALIDATION_EVERY > 0
                and (global_step % CONF.PEDAL_VALIDATION_EVERY) == 0
            ):
                try:
                    pedal_best = run_pedal_validation(epoch, batch_idx, global_step)
                    if pedal_best is not None:
                        pedal_f1 = float(pedal_best["f1"])
                        if pedal_f1 > best_pedal_f1:
                            best_pedal_f1 = pedal_f1
                            checkpoint_path = model_saver(
                                f"step={global_step}_pedal_f1={pedal_f1:.4f}_pedal_best"
                            )
                            save_resume_state(
                                epoch, batch_idx + 1, global_step, MODEL_SNAPSHOT_OUTDIR
                            )
                            txt_logger.loj(
                                "PEDAL_VALIDATION_BEST_CHECKPOINT",
                                {
                                    "epoch": epoch,
                                    "batch_idx": batch_idx,
                                    "global_step": global_step,
                                    "checkpoint": checkpoint_path,
                                    "best_pedal_f1": best_pedal_f1,
                                    "decoder_hyperparameters": {
                                        "threshold": pedal_best["threshold"],
                                        "hysteresis": pedal_best["hysteresis"],
                                        "smoothing_window": pedal_best[
                                            "smoothing_window"
                                        ],
                                        "min_hold_steps": pedal_best["min_hold_steps"],
                                        "shift": pedal_best["shift"],
                                    },
                                },
                            )
                except Exception as exc:
                    txt_logger.loj(
                        "PEDAL_VALIDATION_ERROR",
                        {
                            "epoch": epoch,
                            "batch_idx": batch_idx,
                            "global_step": global_step,
                            "error": str(exc),
                        },
                    )
                finally:
                    cleanup_memory(verbose=False)
                    model.train()
                    keep_frozen_note_modules_eval()

            # ##################################################################
            # # TRAINING
            # ##################################################################
            with torch.no_grad():
                logmels = logmels.to(CONF.DEVICE)
                rolls = rolls.to(CONF.DEVICE)
                model_aligned_rolls = rolls[:, :, 1:]
                onsets = model_aligned_rolls[:, onsets_beg:onsets_end][:, key_beg:key_end]
                # frames = rolls[:, frames_beg:frames_end][:, key_beg:key_end]

                sustain_pedal = rolls[
                    :, maestro_train.SUS_IDX : maestro_train.SUS_IDX + 1
                ]
                pedal_target_bundle = sustain_pedal_targets_from_values(
                    sustain_pedal,
                    threshold=MidiToPianoRoll.SUS_PEDAL_THRESH,
                    transition_width=CONF.PEDAL_TRANSITION_TARGET_WIDTH,
                    align_to_model_diff=True,
                )
                sustain_active = pedal_target_bundle.state

                # ##############################################################
                double_onsets = onsets.clone()
                torch.maximum(
                    onsets[..., :-1], onsets[..., 1:], out=double_onsets[..., 1:]
                )
                triple_onsets = double_onsets.clone()
                torch.maximum(
                    double_onsets[..., :-1],
                    double_onsets[..., 1:],
                    out=triple_onsets[..., 1:],
                )
                #
                onsets_clip = triple_onsets.clip(0, 1)
                onsets_norm = triple_onsets / 127.0
                del onsets
                del double_onsets
                del triple_onsets
                # idx = 0; plt.clf(); plt.imshow(logmels[idx].cpu().numpy()[::-1]); plt.show()
                # idx = 0; plt.clf(); plt.imshow(onsets[idx].cpu().numpy()[::-1]); plt.show()
                # idx = 0; plt.clf(); plt.imshow(double_onsets[idx].cpu().numpy()[::-1]); plt.show()

                # ##############################################################

            # zero the parameter gradients
            if (global_step - 1) % CONF.GRADIENT_ACCUMULATION_STEPS == 0:
                opt.zero_grad()

            # Batch norm requires batch_size > 1 during training
            # With batch size 1, we need to use eval mode for batch norm layers
            if CONF.TRAIN_BS == 1:
                for module in model.modules():
                    if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
                        module.eval()

            # Forward pass: returns onset_stages, velocities, and sustain pedal predictions
            keep_frozen_note_modules_eval()
            onset_stages, velocities, pedals = training_forward(logmels)
            # Shapes: onset_stages: list of (b, 88, t), velocities: (b, 88, t),
            # pedals: (b, 3, t) with sustain [state, onset/down, offset/up]

            if CONF.PEDAL_ONLY_FINETUNE:
                with torch.no_grad():
                    if CONF.LOG_NOTE_MONITOR_LOSSES:
                        vel_loss = CONF.VEL_LOSS_LAMBDA * vel_loss_fn(
                            velocities, onsets_norm, mask=onsets_clip
                        )
                        ons_loss = sum(
                            ons_loss_fn(ons, onsets_clip) for ons in onset_stages
                        ) / len(onset_stages)
                    else:
                        vel_loss = torch.zeros((), device=CONF.DEVICE)
                        ons_loss = torch.zeros((), device=CONF.DEVICE)
            else:
                vel_loss = CONF.VEL_LOSS_LAMBDA * vel_loss_fn(
                    velocities, onsets_norm, mask=onsets_clip
                )

            pedal_frame_targets = sustain_active.squeeze(1)
            pedal_transition_mask = torch.maximum(
                pedal_target_bundle.onset, pedal_target_bundle.offset
            ).squeeze(1)
            pedal_state_logits = pedals[:, model.PEDAL_STATE_IDX:model.PEDAL_STATE_IDX + 1]
            pedal_targets = sustain_active
            pedal_weights = torch.ones_like(pedal_targets)
            if pedal_transition_mask.numel() > 0:
                pedal_transition_weights = torch.ones_like(pedal_frame_targets)
                pedal_transition_weights += (
                    CONF.PEDAL_TRANSITION_WEIGHT - 1.0
                ) * pedal_transition_mask
                pedal_weights = pedal_transition_weights.unsqueeze(1)
            pedal_state_loss = (
                CONF.PEDAL_LOSS_LAMBDA
                * torch.nn.functional.binary_cross_entropy_with_logits(
                    pedal_state_logits,
                    pedal_targets,
                    pos_weight=pedal_pos_weight,
                    reduction="none",
                )
            )
            pedal_state_loss = (pedal_state_loss * pedal_weights).mean()

            pedal_onset_loss = torch.zeros((), device=CONF.DEVICE)
            pedal_offset_loss = torch.zeros((), device=CONF.DEVICE)
            if pedals.shape[1] >= model.PEDAL_NUM_OUTPUTS:
                event_pos_weight = torch.FloatTensor(
                    [CONF.PEDAL_EVENT_POSITIVES_WEIGHT]
                ).to(CONF.DEVICE)
                pedal_onset_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    pedals[:, model.PEDAL_ONSET_IDX:model.PEDAL_ONSET_IDX + 1],
                    pedal_target_bundle.onset,
                    pos_weight=event_pos_weight,
                )
                pedal_offset_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    pedals[:, model.PEDAL_OFFSET_IDX:model.PEDAL_OFFSET_IDX + 1],
                    pedal_target_bundle.offset,
                    pos_weight=event_pos_weight,
                )

            pedal_event_loss = CONF.PEDAL_EVENT_LOSS_LAMBDA * (
                pedal_onset_loss + pedal_offset_loss
            )
            pedal_loss = pedal_state_loss + pedal_event_loss

            loss = pedal_loss if CONF.PEDAL_ONLY_FINETUNE else vel_loss + pedal_loss
            if CONF.TRAINABLE_ONSETS and not CONF.PEDAL_ONLY_FINETUNE:
                ons_loss = sum(
                    ons_loss_fn(ons, onsets_clip) for ons in onset_stages
                ) / len(onset_stages)
                loss += ons_loss

            # Scale loss by accumulation steps
            loss = loss / CONF.GRADIENT_ACCUMULATION_STEPS

            if breakpoint_json("breakpoint.json", global_step):
                onsets = rolls[:, onsets_beg:onsets_end][:, key_beg:key_end]
                breakpoint()
                # idx=0; vel_t=0.1; ons=torch.sigmoid(onset_stages[-1][idx]); plt.clf(); plt.imshow(torch.cat([onsets_clip[idx], onsets_norm[idx], ons, torch.sigmoid(velocities[idx]) * (ons > vel_t)], dim=0).detach().cpu().numpy()[::-1, :1000]); plt.show()
                # idx=0; vel_t=0.1; ons=torch.sigmoid(onset_stages[-1][idx]); plt.clf(); plt.imshow(torch.cat([onsets_norm[idx], torch.sigmoid(velocities[idx]) * (ons > vel_t)], dim=0).detach().cpu().exp().numpy()[::-1, :1000]); plt.show()
                # idx=0; plt.clf(); plt.imshow(torch.cat([onsets_norm[idx], torch.sigmoid(velocities[idx])], dim=0).detach().cpu().exp().numpy()[::-1, :1000]); plt.show()
            #
            loss.backward()

            # Update weights after accumulation steps
            if (global_step % CONF.GRADIENT_ACCUMULATION_STEPS) == 0:
                if CONF.MAX_GRAD_NORM and CONF.MAX_GRAD_NORM > 0:
                    torch.nn.utils.clip_grad_norm_(trainable_params, CONF.MAX_GRAD_NORM)
                opt.step()
                # Resume training mode for batch norm (was in eval for forward pass)
                for module in model.modules():
                    if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
                        module.train()

            # Periodic memory cleanup during training
            if (global_step % 50) == 0:
                cleanup_memory(verbose=False)
            #
            if (global_step % CONF.TRAIN_LOG_EVERY) == 0:
                note_monitor_losses = {
                    "vel": vel_loss.item(),
                    "ons": ons_loss.item() if "ons_loss" in locals() else None,
                }
                logged_losses = {
                    "pedal": pedal_loss.item(),
                    "pedal_state": pedal_state_loss.item(),
                    "pedal_onset": pedal_onset_loss.item(),
                    "pedal_offset": pedal_offset_loss.item(),
                }
                if not CONF.PEDAL_ONLY_FINETUNE:
                    logged_losses["vel"] = vel_loss.item()
                    logged_losses["ons"] = note_monitor_losses["ons"]
                txt_logger.loj(
                    "TRAIN",
                    {
                        "epoch": epoch,
                        "step": batch_idx,
                        "global_step": global_step,
                        "batches_per_epoch": batches_per_epoch,
                        "losses": logged_losses,
                        "note_monitor_losses": (
                            note_monitor_losses if CONF.PEDAL_ONLY_FINETUNE else None
                        ),
                        "loss_mode": (
                            "pedal_only_note_losses_are_monitoring_only"
                            if CONF.PEDAL_ONLY_FINETUNE
                            else "joint_loss"
                        ),
                        "LR": opt.get_lr(),
                    },
                )
                #
            global_step += 1

            # Save resume state periodically (every 500 steps) for efficient I/O
            if (global_step % 500) == 0:
                save_resume_state(
                    epoch, batch_idx + 1, global_step, MODEL_SNAPSHOT_OUTDIR
                )
