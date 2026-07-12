#!/usr/bin/env python
# -*- coding:utf-8 -*-


"""
Assuming a pretrained model for pedal-aware piano transcription, this script uses
the MAESTRO validation split to find its optimal detection thresholds and delay
via a grid search, and then the MAESTRO test split to compute evaluation results
for onsets, velocities, and sustain-pedal events. Specifically:

1. loads cross-validation and test datasets
2. loads ground truth annotations and converts them into event format
3. instantiates the model and decoder to predict log-mel features into musical events
4. performs model inference on the full XV dataset
5. performs grid search XV eval to find the best thresholds and delay hyperparameters
6. performs full test evaluation with XV-optimal thresholds and delay

The workflow is designed to evaluate the project’s unique sustain-pedal prediction
capabilities alongside standard onset and velocity metrics.
"""

import gc
import json
import os

# For omegaconf
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

from ov_piano import PIANO_MIDI_RANGE, HDF5PathManager
from ov_piano.custom_logging import ColorLogger
from ov_piano.data.maestro import (
    MelMaestro,
    MetaMAESTROv1,
    MetaMAESTROv2,
    MetaMAESTROv3,
)
from ov_piano.eval import (
    EvaluationCheckpointStore,
    GtLoaderMaestro,
    evaluation_checkpoint_path,
    evaluation_fingerprint,
    metadata_to_file_id,
    pedal_grid_search,
    threshold_eval_pedals,
    threshold_eval_single_file,
)
from ov_piano.inference import (
    OnsetVelocityNmsDecoder,
    model_outputs_to_probabilities,
    strided_inference,
)
from ov_piano.models.ov import OnsetsAndVelocities
from ov_piano.utils import format_load_model_warnings, load_model

# import matplotlib.pyplot as plt


EVALUATION_PRESETS: Dict[str, Dict[str, object]] = {
    "quick": {
        "XV_TAKE_ONE_EVERY": 50,
        "SEARCH_THRESHOLDS": (0.85,),
        "SEARCH_SHIFTS": (-0.01,),
        "PEDAL_SEARCH_THRESHOLDS": (0.1, 0.3, 0.5, 0.7, 0.9),
        "PEDAL_SEARCH_HYSTERESIS": (0.02, 0.05, 0.1, 0.15),
        "PEDAL_SEARCH_MIN_HOLD_STEPS": (1, 2, 4, 8),
        "PEDAL_SEARCH_SMOOTHING_WINDOWS": (1, 3, 5, 7, 11),
        "PEDAL_SEARCH_SHIFTS": (-0.05, 0.0, 0.05),
        "MAX_PREDICTIONS_PER_FILE": 20000,
        "INFERENCE_CHUNK_SIZE": 30,
    },
    "low_memory": {
        "XV_TAKE_ONE_EVERY": 20,
        "SEARCH_THRESHOLDS": (0.85,),
        "SEARCH_SHIFTS": (-0.01,),
        "PEDAL_SEARCH_THRESHOLDS": (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
        "PEDAL_SEARCH_HYSTERESIS": (0.02, 0.05, 0.1, 0.15),
        "PEDAL_SEARCH_MIN_HOLD_STEPS": (1, 2, 4, 8),
        "PEDAL_SEARCH_SMOOTHING_WINDOWS": (1, 3, 5, 7, 11),
        "PEDAL_SEARCH_SHIFTS": (-0.05, -0.025, 0.0, 0.025, 0.05),
        "MAX_PREDICTIONS_PER_FILE": 20000,
        "INFERENCE_CHUNK_SIZE": 60,
    },
    "full": {
        "XV_TAKE_ONE_EVERY": 1,
        "SEARCH_THRESHOLDS": (0.70, 0.75, 0.80, 0.85),
        "SEARCH_SHIFTS": (-0.03, -0.02, -0.01, 0.0, 0.01),
        "PEDAL_SEARCH_THRESHOLDS": (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
        "PEDAL_SEARCH_HYSTERESIS": (0.02, 0.05, 0.1, 0.15),
        "PEDAL_SEARCH_MIN_HOLD_STEPS": (1, 2, 4, 8),
        "PEDAL_SEARCH_SMOOTHING_WINDOWS": (1, 3, 5, 7, 11),
        "PEDAL_SEARCH_SHIFTS": (-0.05, -0.025, 0.0, 0.025, 0.05),
        "MAX_PREDICTIONS_PER_FILE": 50000,
        "INFERENCE_CHUNK_SIZE": 300,
    },
}


def _preset_help() -> str:
    return ", ".join(sorted(EVALUATION_PRESETS))


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

    :cvar XV_TAKE_ONE_EVERY: Since we are doing a (likely inefficient)
      grid search on the cross-validation set, and the size is considerable,
      we can use this parameter to take only one file from every N in the set.
      Use the ``full`` preset for final/reportable metrics because shortened
      validation searches are intended for smoke tests or memory-constrained
      debugging only.
      Experiments show that taking 1 of 5 doesn't alter results significantly.
    :cvar SEARCH_THRESHOLDS: Before running the test, several thresholds are
      being searched via grid search on the cross-validation split. This list
      determines said thresholds.
    :cvar SEARCH_SHIFTS: Analogous to the thresholds, but determines what delay
      offset, in seconds, is applied to the predictions.

    :cvar DECODER_GAUSS_STD: The decoder on top of the DNN predictions performs
      a Gaussian time-convolution to smoothen detections. This is the standard
      deviation, in time-frames.
    :cvar DECODER_GAUSS_KSIZE: The window size, in time-frames, for the
      smoothening Gaussian time-convolution.

    :cvar TOLERANCE_SECS: The maximum absolute error between onset prediction
      and ground truth, in seconds, to consider the prediction correct.
    :cvar TOLERANCE_VEL: The maximum absolute error between velocity prediction
      and ground truth, in ratio between 0 and 1, to consider the prediction
      correct. To better understand this ratio, see the official documentation
      for ``mir_eval.transcription_velocity``.

    :cvar INFERENCE_CHUNK_SIZE: In this module, full files are processed, which
      may be too large for memory and have to be processed in strided chunks.
      This is the chunk size in seconds, it doesn't affect performance as long
      as it is large enough.
    :cvar INFERENCE_CHUNK_OVERLAP: See ``INFERENCE_CHUNK_SIZE``. This is the
      overlap among consecutive chunks. It doesn't affect performance as long
      as it is large enough to avoid boundary artifacts.
    """

    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    EVALUATION_PRESET: str = "low_memory"
    MAESTRO_PATH: str = os.path.join("datasets", "maestro", "maestro-v3.0.0")
    MAESTRO_VERSION: int = 3
    OUTPUT_DIR: str = "out"
    EVALUATION_CHECKPOINTS_ENABLED: bool = True
    EVALUATION_CHECKPOINT_DIR: Optional[str] = None
    RESET_EVALUATION_CHECKPOINTS: bool = False
    #
    HDF5_MEL_PATH: str = os.path.join(
        "datasets", "MAESTROv3_logmel_sr=16000_stft=2048w384h_mel=229(50-8000).h5"
    )
    HDF5_ROLL_PATH: str = os.path.join(
        "datasets", "MAESTROv3_roll_quant=0.024_midivals=128_extendsus=True.h5"
    )
    SNAPSHOT_INPATH: str = os.path.join(
        "assets",
        "OnsetsAndVelocities_2026_07_12_08_12_56.769.torch",
    )
    #
    CONV1X1: List[int] = (200, 200)  # MUST match checkpoint architecture!
    LEAKY_RELU_SLOPE: Optional[float] = 0.1
    #
    XV_TAKE_ONE_EVERY: int = 20  # Increased from 5 to reduce memory usage
    SEARCH_THRESHOLDS: List[float] = (
        0.85,
    )  # High threshold to prevent too many predictions
    SEARCH_SHIFTS: List[float] = (-0.01,)
    PEDAL_SEARCH_THRESHOLDS: List[float] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
    PEDAL_SEARCH_HYSTERESIS: List[float] = (0.02, 0.05, 0.1, 0.15)
    PEDAL_SEARCH_MIN_HOLD_STEPS: List[int] = (1, 2, 4, 8)
    PEDAL_SEARCH_SMOOTHING_WINDOWS: List[int] = (1, 3, 5, 7, 11)
    PEDAL_SEARCH_SHIFTS: List[float] = (-0.05, -0.025, 0.0, 0.025, 0.05)
    MAX_PREDICTIONS_PER_FILE: int = (
        20000  # Safety limit (normal files have 1k-5k notes)
    )
    #
    DECODER_GAUSS_STD: float = 1
    DECODER_GAUSS_KSIZE: int = 11
    #
    TOLERANCE_SECS: float = 0.05
    TOLERANCE_VEL: float = 0.1
    #
    INFERENCE_CHUNK_SIZE: float = 60  # Reduced from 300 to 60 for lower memory usage
    INFERENCE_CHUNK_OVERLAP: float = 11


# ##############################################################################
# # MAIN LOOP INITIALIZATION
# ##############################################################################
if __name__ == "__main__":
    CONF = OmegaConf.structured(ConfDef())
    cli_conf = OmegaConf.from_cli()
    preset_name = str(cli_conf.get("EVALUATION_PRESET", CONF.EVALUATION_PRESET))
    if preset_name not in EVALUATION_PRESETS:
        raise ValueError(
            f"Unknown EVALUATION_PRESET={preset_name!r}. "
            f"Choose one of: {_preset_help()}"
        )
    preset_conf = OmegaConf.create(EVALUATION_PRESETS[preset_name])
    # Merge order: defaults < named preset < explicit CLI overrides.
    # This lets users start from a preset while still overriding individual knobs.
    CONF = OmegaConf.merge(CONF, preset_conf, cli_conf)

    # derivative globals + parse HDF5 filenames and ensure they are consistent
    (DATASET_NAME, SAMPLERATE, WINSIZE, HOPSIZE, MELBINS, FMIN, FMAX) = (
        HDF5PathManager.parse_mel_hdf5_basename(os.path.basename(CONF.HDF5_MEL_PATH))
    )
    roll_params = HDF5PathManager.parse_roll_hdf5_basename(
        os.path.basename(CONF.HDF5_ROLL_PATH)
    )
    SECS_PER_FRAME = HOPSIZE / SAMPLERATE
    #
    CHUNK_SIZE = round(CONF.INFERENCE_CHUNK_SIZE / SECS_PER_FRAME)
    CHUNK_OVERLAP = round(CONF.INFERENCE_CHUNK_OVERLAP / SECS_PER_FRAME)
    #
    assert DATASET_NAME == roll_params[0], "Inconsistent HDF5 datasets?"
    assert SECS_PER_FRAME == roll_params[1], "Inconsistent roll quantization?"
    assert (CHUNK_OVERLAP % 2) == 0, f"Only even overlap allowed! {CHUNK_OVERLAP}"
    #
    METAMAESTRO_CLASS = {1: MetaMAESTROv1, 2: MetaMAESTROv2, 3: MetaMAESTROv3}[
        CONF.MAESTRO_VERSION
    ]
    TXT_LOG_OUTDIR = os.path.join(CONF.OUTPUT_DIR, "txt_logs")
    os.makedirs(TXT_LOG_OUTDIR, exist_ok=True)

    txt_logger = ColorLogger(os.path.basename(__file__), TXT_LOG_OUTDIR)
    txt_logger.info("\n\nCONFIGURATION:\n" + OmegaConf.to_yaml(CONF) + "\n\n")
    txt_logger.warning(
        f"EVALUATION_PRESET={CONF.EVALUATION_PRESET!r}. Available presets: {_preset_help()}."
    )
    if CONF.XV_TAKE_ONE_EVERY != 1:
        txt_logger.warning(
            "NON-FINAL VALIDATION SEARCH: this run uses only every "
            f"{CONF.XV_TAKE_ONE_EVERY} validation file(s). Use "
            "EVALUATION_PRESET=full (XV_TAKE_ONE_EVERY=1) before reporting "
            "validation-selected thresholds or final benchmark metrics."
        )
    else:
        txt_logger.warning(
            "FULL VALIDATION SEARCH: all validation files are used. This is the "
            "recommended mode for reportable/final metrics."
        )

    EVAL_CHECKPOINT_DIR = CONF.EVALUATION_CHECKPOINT_DIR
    if EVAL_CHECKPOINT_DIR is None:
        EVAL_CHECKPOINT_DIR = os.path.join(CONF.OUTPUT_DIR, "eval_checkpoints")
    if CONF.EVALUATION_CHECKPOINTS_ENABLED:
        os.makedirs(EVAL_CHECKPOINT_DIR, exist_ok=True)
        txt_logger.info(f"Evaluation checkpoints enabled: {EVAL_CHECKPOINT_DIR}")
    else:
        txt_logger.warning("Evaluation checkpoints disabled")

    def _snapshot_signature(path):
        abspath = os.path.abspath(os.fspath(path))
        signature = {"path": abspath}
        try:
            stat = os.stat(abspath)
            signature.update({"size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)})
        except OSError:
            signature["missing"] = True
        return signature

    SNAPSHOT_SIGNATURE = _snapshot_signature(CONF.SNAPSHOT_INPATH)

    def _make_checkpoint_store(stage, config):
        fingerprint = evaluation_fingerprint(config)
        path = evaluation_checkpoint_path(
            EVAL_CHECKPOINT_DIR,
            os.path.basename(__file__),
            stage,
            fingerprint,
        )
        store = EvaluationCheckpointStore(
            path,
            fingerprint,
            stage,
            enabled=CONF.EVALUATION_CHECKPOINTS_ENABLED,
            reset=CONF.RESET_EVALUATION_CHECKPOINTS,
            logger=txt_logger.info,
        )
        if CONF.EVALUATION_CHECKPOINTS_ENABLED:
            txt_logger.info(f"Evaluation checkpoint [{stage}]: {path}")
        return store

    txt_logger.info("Loading datasets")
    metamaestro_xv = METAMAESTRO_CLASS(
        CONF.MAESTRO_PATH, splits=["validation"], years=METAMAESTRO_CLASS.ALL_YEARS
    )
    maestro_xv = MelMaestro(
        CONF.HDF5_MEL_PATH,
        CONF.HDF5_ROLL_PATH,
        *(x[0] for x in metamaestro_xv.data),
        as_torch_tensors=False,
    )
    metamaestro_test = METAMAESTRO_CLASS(
        CONF.MAESTRO_PATH, splits=["test"], years=METAMAESTRO_CLASS.ALL_YEARS
    )
    maestro_test = MelMaestro(
        CONF.HDF5_MEL_PATH,
        CONF.HDF5_ROLL_PATH,
        *(x[0] for x in metamaestro_test.data),
        as_torch_tensors=False,
    )

    # shorten xv set to speed up cross validation times
    if CONF.XV_TAKE_ONE_EVERY != 1:
        txt_logger.critical(
            "SHORTENING XV SPLIT FOR FASTER CROSSVALIDATION! "
            "Do not report validation-search metrics from this preset as final."
        )
        maestro_xv.data = maestro_xv.data[:: CONF.XV_TAKE_ONE_EVERY]
        metamaestro_xv.data = metamaestro_xv.data[:: CONF.XV_TAKE_ONE_EVERY]
    #
    txt_logger.info("Loading XV ground truths")
    xv_gts = GtLoaderMaestro(maestro_xv, metamaestro_xv)

    txt_logger.info("Loading test ground truths")
    test_gts = GtLoaderMaestro(maestro_test, metamaestro_test)

    # instantiate and load trained NN model
    txt_logger.info("Loading NN")
    num_mels = maestro_xv[0][0].shape[0]
    key_beg, key_end = PIANO_MIDI_RANGE
    num_piano_keys = key_end - key_beg
    #
    model = OnsetsAndVelocities(
        in_chans=2,  # X and time_derivative(X)
        in_height=num_mels,
        out_height=num_piano_keys,
        conv1x1head=CONF.CONV1X1,
        bn_momentum=0,
        leaky_relu_slope=CONF.LEAKY_RELU_SLOPE,
        dropout_drop_p=0,
    ).to(CONF.DEVICE)
    load_report = load_model(
        model,
        CONF.SNAPSHOT_INPATH,
        eval_phase=True,
        to_cpu=(CONF.DEVICE == "cpu"),
        strict=False,
    )
    for warning in format_load_model_warnings(load_report):
        txt_logger.warning(f"CHECKPOINT LOAD WARNING: {warning}")
    # instantiate decoder
    decoder = OnsetVelocityNmsDecoder(
        num_piano_keys,
        nms_pool_ksize=3,
        gauss_conv_stddev=CONF.DECODER_GAUSS_STD,
        gauss_conv_ksize=CONF.DECODER_GAUSS_KSIZE,
        vel_pad_left=1,
        vel_pad_right=1,
    )
    ##############
    # XV INFERENCE
    ##############
    def model_inference(x):
        """
        Convenience wrapper around the DNN to ensure output and input sequences
        have same length.
        """
        return model_outputs_to_probabilities(model(x), include_pedals=True)

    xv_file_ids = [metadata_to_file_id(md) for _, _, md in maestro_xv.data]
    xv_inference_store = _make_checkpoint_store(
        "xv_inference",
        {
            "snapshot": SNAPSHOT_SIGNATURE,
            "maestro_version": int(CONF.MAESTRO_VERSION),
            "maestro_path": os.path.abspath(CONF.MAESTRO_PATH),
            "hdf5_mel_path": os.path.abspath(CONF.HDF5_MEL_PATH),
            "hdf5_roll_path": os.path.abspath(CONF.HDF5_ROLL_PATH),
            "split": "validation",
            "file_ids": xv_file_ids,
            "evaluation_preset": str(CONF.EVALUATION_PRESET),
            "xv_take_one_every": int(CONF.XV_TAKE_ONE_EVERY),
            "secs_per_frame": float(SECS_PER_FRAME),
            "chunk_size": int(CHUNK_SIZE),
            "chunk_overlap": int(CHUNK_OVERLAP),
            "conv1x1": list(CONF.CONV1X1),
            "leaky_relu_slope": CONF.LEAKY_RELU_SLOPE,
            "decoder_gauss_std": float(CONF.DECODER_GAUSS_STD),
            "decoder_gauss_ksize": int(CONF.DECODER_GAUSS_KSIZE),
            "decoder_pthresh": float(min(CONF.SEARCH_THRESHOLDS)),
            "max_predictions_per_file": int(CONF.MAX_PREDICTIONS_PER_FILE),
        },
    )

    len_xv = len(maestro_xv)
    xv_dataframes = []
    for i, (mel, roll, md) in enumerate(maestro_xv, 1):
        file_id = metadata_to_file_id(md)
        txt_logger.info(f"[{i}/{len_xv}] XV inference: {md}")
        try:
            cached_entry = xv_inference_store.get(file_id)
            if isinstance(cached_entry, dict):
                if cached_entry.get("status") == "ok":
                    txt_logger.info(f"  XV checkpoint hit: {file_id}")
                    xv_dataframes.append(
                        (
                            file_id,
                            cached_entry["gt_df"],
                            cached_entry["pred_df"],
                            cached_entry["gt_pedal_df"],
                            cached_entry["pedal_pred"],
                        )
                    )
                    continue
                if cached_entry.get("status") == "skipped":
                    txt_logger.warning(
                        f"  XV checkpoint skip: {file_id} "
                        f"({cached_entry.get('reason', 'unknown')})"
                    )
                    continue

            with torch.no_grad():
                tmel = torch.from_numpy(mel).to(CONF.DEVICE).unsqueeze(0)
                # Skip empty inputs
                if tmel.shape[-1] == 0:
                    txt_logger.warning(f"SKIPPING {md[0]}: Empty mel input (0 frames)")
                    xv_inference_store.upsert(
                        file_id,
                        {"status": "skipped", "metadata": tuple(md), "reason": "empty_mel"},
                    )
                    del tmel
                    continue

                # Run strided inference and validate output
                _res = strided_inference(
                    model_inference, tmel, CHUNK_SIZE, CHUNK_OVERLAP
                )
                del tmel
                if not _res or len(_res) < 3:
                    msg = f"SKIPPING {md[0]}: strided_inference returned invalid result: {type(_res)} len={len(_res) if hasattr(_res, '__len__') else 'NA'}"
                    txt_logger.error(msg)
                    xv_inference_store.upsert(
                        file_id,
                        {
                            "status": "skipped",
                            "metadata": tuple(md),
                            "reason": "invalid_inference_result",
                        },
                    )
                    continue

                onset_pred, vel_pred, pedal_pred = _res
                pred_df = decoder(
                    onset_pred, vel_pred, pthresh=min(CONF.SEARCH_THRESHOLDS)
                )
                # Process pedal predictions
                if pedal_pred.dim() == 2:
                    pedal_pred = pedal_pred.unsqueeze(0)  # Add batch dimension
                pedal_pred = pedal_pred.cpu()

                # Safety check: skip files with excessive predictions
                num_preds = len(pred_df)
                gt_df = xv_gts(md)[0]
                gt_pedal_df = xv_gts.get_sus_pedal_events(md, SECS_PER_FRAME)
                num_gt = len(gt_df)

                if num_preds > CONF.MAX_PREDICTIONS_PER_FILE:
                    txt_logger.warning(
                        f"SKIPPING {md[0]}: Too many predictions ({num_preds:,}) "
                        f"vs {num_gt:,} ground truth. This would cause OOM."
                    )
                    xv_inference_store.upsert(
                        file_id,
                        {
                            "status": "skipped",
                            "metadata": tuple(md),
                            "reason": "too_many_predictions",
                            "num_predictions": int(num_preds),
                            "num_ground_truth": int(num_gt),
                        },
                    )
                    continue

                txt_logger.info(f"  GT: {num_gt:,} notes, Pred: {num_preds:,} notes")
                xv_inference_store.upsert(
                    file_id,
                    {
                        "status": "ok",
                        "metadata": tuple(md),
                        "gt_df": gt_df,
                        "pred_df": pred_df,
                        "gt_pedal_df": gt_pedal_df,
                        "pedal_pred": pedal_pred,
                    },
                )
                xv_dataframes.append((file_id, gt_df, pred_df, gt_pedal_df, pedal_pred))

        except Exception as e:
            txt_logger.error(f"ERROR processing {md[0]}: {e}")
            continue
        finally:
            # Aggressive memory cleanup (delete only defined names)
            for v in (
                "mel",
                "roll",
                "onset_pred",
                "vel_pred",
                "pedal_pred",
                "tmel",
                "_res",
                "pred_df",
                "gt_df",
                "gt_pedal_df",
                "cached_entry",
            ):
                if v in locals():
                    try:
                        del locals()[v]
                    except Exception:
                        pass
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    ###############
    # XV GRIDSEARCH
    ###############
    # Check if we have enough valid files to continue
    if len(xv_dataframes) == 0:
        txt_logger.error(
            "ERROR: No valid validation files processed! "
            "All files were skipped due to excessive predictions or errors."
        )
        txt_logger.error(
            "Try: 1) Lower MAX_PREDICTIONS_PER_FILE, or "
            "2) Increase SEARCH_THRESHOLDS to reduce predictions, or "
            "3) Check if model is trained properly."
        )
        raise RuntimeError("No valid validation files for evaluation")

    txt_logger.info(
        f"Successfully processed {len(xv_dataframes)} / {len_xv} validation files"
    )

    xv_note_grid_store = _make_checkpoint_store(
        "xv_note_grid",
        {
            "source_stage": "xv_inference",
            "source_fingerprint": xv_inference_store.fingerprint,
            "file_ids": [item[0] for item in xv_dataframes],
            "thresholds": list(CONF.SEARCH_THRESHOLDS),
            "shifts": list(CONF.SEARCH_SHIFTS),
            "secs_per_frame": float(SECS_PER_FRAME),
            "key_beg": int(key_beg),
            "tolerance_secs": float(CONF.TOLERANCE_SECS),
            "tolerance_vel": float(CONF.TOLERANCE_VEL),
        },
    )
    xv_gridsearch = {}
    xv_gridsearch_vel = {}
    for thresh in CONF.SEARCH_THRESHOLDS:
        for shift in CONF.SEARCH_SHIFTS:
            checkpoint_key = json.dumps((float(thresh), float(shift)))
            cached_note_grid = xv_note_grid_store.get(checkpoint_key)
            if isinstance(cached_note_grid, dict) and cached_note_grid.get("status") == "ok":
                txt_logger.info(f"XV notes checkpoint hit: {(thresh, shift)}")
                xv_gridsearch[(thresh, shift)] = [tuple(x) for x in cached_note_grid["eval"]]
                xv_gridsearch_vel[(thresh, shift)] = [tuple(x) for x in cached_note_grid["eval_vel"]]
                continue

            this_eval = []
            this_eval_vel = []

            for i, (_, gtdf, preddf, _, _) in enumerate(xv_dataframes, 1):
                txt_logger.info(
                    f"[{i}/{len(xv_dataframes)} XV notes]: {(thresh, shift)}"
                )
                prf1, prf1_v = threshold_eval_single_file(
                    gtdf,
                    preddf,
                    SECS_PER_FRAME,
                    key_beg,
                    thresh=thresh,
                    shift_preds=shift,
                    tol_secs=CONF.TOLERANCE_SECS,
                    tol_vel=CONF.TOLERANCE_VEL,
                )
                this_eval.append(prf1)
                this_eval_vel.append(prf1_v)

            xv_gridsearch[(thresh, shift)] = this_eval
            xv_gridsearch_vel[(thresh, shift)] = this_eval_vel
            xv_note_grid_store.upsert(
                checkpoint_key,
                {
                    "status": "ok",
                    "threshold": float(thresh),
                    "shift": float(shift),
                    "eval": [tuple(map(float, x)) for x in this_eval],
                    "eval_vel": [tuple(map(float, x)) for x in this_eval_vel],
                },
            )

    xv_pedal_grid_store = _make_checkpoint_store(
        "xv_pedal_grid",
        {
            "source_stage": "xv_inference",
            "source_fingerprint": xv_inference_store.fingerprint,
            "file_ids": [item[0] for item in xv_dataframes],
            "secs_per_frame": float(SECS_PER_FRAME),
            "thresholds": list(CONF.PEDAL_SEARCH_THRESHOLDS),
            "hysteresis_values": list(CONF.PEDAL_SEARCH_HYSTERESIS),
            "smoothing_windows": list(CONF.PEDAL_SEARCH_SMOOTHING_WINDOWS),
            "min_hold_steps_values": list(CONF.PEDAL_SEARCH_MIN_HOLD_STEPS),
            "shifts": list(CONF.PEDAL_SEARCH_SHIFTS),
            "tolerance_secs": float(CONF.TOLERANCE_SECS),
        },
    )
    xv_summary_pedal, best_pedal_params, best_pedal_metrics = pedal_grid_search(
        ((gt_pedal_df, pedal_pred) for _, _, _, gt_pedal_df, pedal_pred in xv_dataframes),
        SECS_PER_FRAME,
        thresholds=CONF.PEDAL_SEARCH_THRESHOLDS,
        hysteresis_values=CONF.PEDAL_SEARCH_HYSTERESIS,
        smoothing_windows=CONF.PEDAL_SEARCH_SMOOTHING_WINDOWS,
        min_hold_steps_values=CONF.PEDAL_SEARCH_MIN_HOLD_STEPS,
        shifts=CONF.PEDAL_SEARCH_SHIFTS,
        tol_secs=CONF.TOLERANCE_SECS,
        logger=txt_logger.info,
        log_prefix="XV pedal",
        max_logged_items=1,
        checkpoint_store=xv_pedal_grid_store,
    )

    # Compute mean metrics ensuring proper array shape
    xv_summary = {}
    for k, v in xv_gridsearch.items():
        mean_val = np.mean(v, axis=0)
        # Ensure it's always a 1D array of length 3
        if mean_val.ndim == 0:
            mean_val = np.array([mean_val, mean_val, mean_val])
        xv_summary[k] = mean_val

    xv_summary_vel = {}
    for k, v in xv_gridsearch_vel.items():
        mean_val = np.mean(v, axis=0)
        if mean_val.ndim == 0:
            mean_val = np.array([mean_val, mean_val, mean_val])
        xv_summary_vel[k] = mean_val

    # Find best threshold/shift based on F1 score
    try:
        ((best_t, best_s), (best_p, best_r, best_f1)) = max(
            xv_summary.items(), key=lambda elt: elt[1][2]
        )
    except (IndexError, ValueError) as e:
        txt_logger.error(f"Error finding best hyperparameters: {e}")
        txt_logger.error(f"xv_summary structure: {xv_summary}")
        # Use first threshold/shift as fallback
        (best_t, best_s) = list(xv_summary.keys())[0]
        (best_p, best_r, best_f1) = xv_summary[(best_t, best_s)]
    #
    xv_summary_df = pd.DataFrame(
        ((t, s, p, r, f1) for ((t, s), (p, r, f1)) in xv_summary.items()),
        columns=["threshold", "shift", "P", "R", "F1"],
    )

    xv_summary_df_vel = pd.DataFrame(
        ((t, s, p, r, f1) for ((t, s), (p, r, f1)) in xv_summary_vel.items()),
        columns=["threshold", "shift", "P", "R", "F1"],
    )

    if best_pedal_params is None:
        raise RuntimeError("No pedal hyperparameter combinations were evaluated")
    (
        best_pedal_t,
        best_pedal_h,
        best_pedal_smoothing,
        best_pedal_min_hold,
        best_pedal_shift,
    ) = best_pedal_params
    best_pedal_p, best_pedal_r, best_pedal_f1 = best_pedal_metrics

    xv_summary_df_pedal = pd.DataFrame(
        (
            (t, h, sw, mh, s, p, r, f1)
            for ((t, h, sw, mh, s), (p, r, f1)) in xv_summary_pedal.items()
        ),
        columns=[
            "threshold",
            "hysteresis",
            "smoothing_window",
            "min_hold_steps",
            "shift",
            "P",
            "R",
            "F1",
        ],
    )

    txt_logger.warning("XV HYPERPARAMETER SEARCH:")
    txt_logger.warning("Summary (without velocity):\n" + str(xv_summary_df))
    txt_logger.warning("Summary (with velocity):\n" + str(xv_summary_df_vel))
    txt_logger.warning("Summary (sustain pedal):\n" + str(xv_summary_df_pedal))

    ###############
    # TEST
    ###############
    test_results = []
    test_results_vel = []
    test_results_pedal = []
    len_test = len(maestro_test)
    test_file_ids = [metadata_to_file_id(md) for _, _, md in maestro_test.data]
    test_metrics_store = _make_checkpoint_store(
        "test_metrics",
        {
            "snapshot": SNAPSHOT_SIGNATURE,
            "maestro_version": int(CONF.MAESTRO_VERSION),
            "maestro_path": os.path.abspath(CONF.MAESTRO_PATH),
            "hdf5_mel_path": os.path.abspath(CONF.HDF5_MEL_PATH),
            "hdf5_roll_path": os.path.abspath(CONF.HDF5_ROLL_PATH),
            "split": "test",
            "file_ids": test_file_ids,
            "secs_per_frame": float(SECS_PER_FRAME),
            "chunk_size": int(CHUNK_SIZE),
            "chunk_overlap": int(CHUNK_OVERLAP),
            "conv1x1": list(CONF.CONV1X1),
            "leaky_relu_slope": CONF.LEAKY_RELU_SLOPE,
            "decoder_gauss_std": float(CONF.DECODER_GAUSS_STD),
            "decoder_gauss_ksize": int(CONF.DECODER_GAUSS_KSIZE),
            "decoder_pthresh": float(min(CONF.SEARCH_THRESHOLDS)),
            "max_predictions_per_file": int(CONF.MAX_PREDICTIONS_PER_FILE),
            "best_note_threshold": float(best_t),
            "best_note_shift": float(best_s),
            "best_pedal_threshold": float(best_pedal_t),
            "best_pedal_hysteresis": float(best_pedal_h),
            "best_pedal_smoothing_window": int(best_pedal_smoothing),
            "best_pedal_min_hold_steps": int(best_pedal_min_hold),
            "best_pedal_shift": float(best_pedal_shift),
            "tolerance_secs": float(CONF.TOLERANCE_SECS),
            "tolerance_vel": float(CONF.TOLERANCE_VEL),
        },
    )
    for i, (mel, roll, md) in enumerate(maestro_test, 1):
        file_id = metadata_to_file_id(md)
        txt_logger.info(f"[{i}/{len_test} (test set)] {md}")
        try:
            cached_entry = test_metrics_store.get(file_id)
            if isinstance(cached_entry, dict):
                if cached_entry.get("status") == "ok":
                    txt_logger.info(f"  Test checkpoint hit: {file_id}")
                    filename = cached_entry.get("filename", file_id)
                    prf1 = tuple(cached_entry["prf1"])
                    prf1_v = tuple(cached_entry["prf1_v"])
                    pedal_prf1 = tuple(cached_entry["pedal_prf1"])
                    test_results.append((filename, *prf1))
                    test_results_vel.append((filename, *prf1_v))
                    test_results_pedal.append((filename, *pedal_prf1))
                    continue
                if cached_entry.get("status") == "skipped":
                    txt_logger.warning(
                        f"  Test checkpoint skip: {file_id} "
                        f"({cached_entry.get('reason', 'unknown')})"
                    )
                    continue

            with torch.no_grad():
                tmel = torch.from_numpy(mel).to(CONF.DEVICE).unsqueeze(0)
                if tmel.shape[-1] == 0:
                    txt_logger.warning(f"SKIPPING {md[0]}: Empty mel input (0 frames)")
                    test_metrics_store.upsert(
                        file_id,
                        {"status": "skipped", "filename": md[0], "metadata": tuple(md), "reason": "empty_mel"},
                    )
                    del tmel
                    continue
                onset_pred, vel_pred, pedal_pred = strided_inference(
                    model_inference, tmel, CHUNK_SIZE, CHUNK_OVERLAP
                )
                del tmel
                pred_df = decoder(
                    onset_pred, vel_pred, pthresh=min(CONF.SEARCH_THRESHOLDS)
                )
                gt_df = test_gts(md)[0]
                gt_pedal_df = test_gts.get_sus_pedal_events(md, SECS_PER_FRAME)
                # Process pedal predictions
                if pedal_pred.dim() == 2:
                    pedal_pred = pedal_pred.unsqueeze(0)  # Add batch dimension
                pedal_pred = pedal_pred.cpu()

            # Safety check: skip files with excessive predictions
            num_preds = len(pred_df)
            num_gt = len(gt_df)

            if num_preds > CONF.MAX_PREDICTIONS_PER_FILE:
                txt_logger.warning(
                    f"SKIPPING {md[0]}: Too many predictions ({num_preds:,}) "
                    f"vs {num_gt:,} ground truth. This would cause OOM."
                )
                test_metrics_store.upsert(
                    file_id,
                    {
                        "status": "skipped",
                        "filename": md[0],
                        "metadata": tuple(md),
                        "reason": "too_many_predictions",
                        "num_predictions": int(num_preds),
                        "num_ground_truth": int(num_gt),
                    },
                )
                continue

            txt_logger.info(f"  GT: {num_gt:,} notes, Pred: {num_preds:,} notes")

            prf1, prf1_v = threshold_eval_single_file(
                gt_df,
                pred_df,
                SECS_PER_FRAME,
                key_beg,
                thresh=best_t,
                shift_preds=best_s,
                tol_secs=CONF.TOLERANCE_SECS,
                tol_vel=CONF.TOLERANCE_VEL,
            )
            # Evaluate pedal predictions
            try:
                pedal_results = threshold_eval_pedals(
                    gt_pedal_df,
                    pedal_pred,
                    SECS_PER_FRAME,
                    thresh=best_pedal_t,
                    shift_preds=best_pedal_shift,
                    tol_secs=CONF.TOLERANCE_SECS,
                    hysteresis=best_pedal_h,
                    smoothing_window=best_pedal_smoothing,
                    min_hold_steps=best_pedal_min_hold,
                )
                # Extract sustain pedal metrics (index 0)
                if "sustain" in pedal_results:
                    pedal_prf1 = (
                        pedal_results["sustain"]["precision"],
                        pedal_results["sustain"]["recall"],
                        pedal_results["sustain"]["f1"],
                    )
                else:
                    # Fallback if no sustain pedal results
                    pedal_prf1 = (0.0, 0.0, 0.0)
            except Exception as e:
                txt_logger.warning(f"Pedal evaluation failed for {md[0]}: {e}")
                pedal_prf1 = (0.0, 0.0, 0.0)

            test_results.append((md[0], *prf1))
            test_results_vel.append((md[0], *prf1_v))
            test_results_pedal.append((md[0], *pedal_prf1))
            test_metrics_store.upsert(
                file_id,
                {
                    "status": "ok",
                    "filename": md[0],
                    "metadata": tuple(md),
                    "prf1": tuple(map(float, prf1)),
                    "prf1_v": tuple(map(float, prf1_v)),
                    "pedal_prf1": tuple(map(float, pedal_prf1)),
                },
            )

        except Exception as e:
            txt_logger.error(f"ERROR processing {md[0]}: {e}")
            continue
        finally:
            # Aggressive memory cleanup (delete only names that were assigned).
            # If inference fails before creating onset_pred/vel_pred/etc., a
            # plain ``del`` raises NameError and masks the real processing error.
            for v in (
                "mel",
                "roll",
                "tmel",
                "onset_pred",
                "vel_pred",
                "pedal_pred",
                "pred_df",
                "gt_df",
                "gt_pedal_df",
                "cached_entry",
            ):
                if v in locals():
                    try:
                        del locals()[v]
                    except Exception:
                        pass
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    #
    test_results_df = pd.DataFrame(test_results, columns=["Filename", "P", "R", "F1"])
    averages = [
        f"AVERAGES (t={best_t}, s={best_s})",
        *test_results_df.iloc[:, 1:].mean().tolist(),
    ]
    test_results_df.loc[len(test_results_df)] = averages
    #
    test_results_df_vel = pd.DataFrame(
        test_results_vel, columns=["Filename", "P", "R", "F1"]
    )
    averages_vel = [
        f"AVERAGES (t={best_t}, s={best_s})",
        *test_results_df_vel.iloc[:, 1:].mean().tolist(),
    ]
    test_results_df_vel.loc[len(test_results_df_vel)] = averages_vel
    #
    test_results_df_pedal = pd.DataFrame(
        test_results_pedal, columns=["Filename", "P", "R", "F1"]
    )
    averages_pedal = [
        (
            f"AVERAGES (pedal_t={best_pedal_t}, h={best_pedal_h}, "
            f"smooth={best_pedal_smoothing}, hold={best_pedal_min_hold}, "
            f"shift={best_pedal_shift})"
        ),
        *test_results_df_pedal.iloc[:, 1:].mean().tolist(),
    ]
    test_results_df_pedal.loc[len(test_results_df_pedal)] = averages_pedal
    #
    txt_logger.warning(
        "TEST RESULTS WITH BEST XV HYPERPARS "
        + f"(MAESTROv{CONF.MAESTRO_VERSION}, "
        + f"{CONF.SNAPSHOT_INPATH}, "
        + f"preset={CONF.EVALUATION_PRESET}, "
        + f"xv_take_one_every={CONF.XV_TAKE_ONE_EVERY})\n"
    )
    if CONF.XV_TAKE_ONE_EVERY != 1:
        txt_logger.warning(
            "IMPORTANT: these test results used hyperparameters selected from a "
            "shortened validation split. Treat them as quick/diagnostic numbers; "
            "rerun with EVALUATION_PRESET=full for final reporting."
        )
    txt_logger.warning("ONSETS:\n" + str(test_results_df))
    txt_logger.warning("ONSETS+VELOCITIES:\n" + str(test_results_df_vel))
    txt_logger.warning("SUSTAIN PEDAL:\n" + str(test_results_df_pedal))

    # Export full results to CSV
    csv_dir = "out_test/results_csv"
    os.makedirs(csv_dir, exist_ok=True)
    test_results_df.to_csv(f"{csv_dir}/test_onsets_results.csv", index=False)
    test_results_df_vel.to_csv(f"{csv_dir}/test_velocities_results.csv", index=False)
    test_results_df_pedal.to_csv(f"{csv_dir}/test_pedal_results.csv", index=False)
    txt_logger.warning(f"Results exported to {csv_dir}/")
