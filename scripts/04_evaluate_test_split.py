#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
Evaluate a trained pedal-aware model on the MAESTRO test split.

This script mirrors the workflow of 03_evaluate_pedal_model.py but focuses on
running a full test-split evaluation with the same decoder and pedal evaluation
logic. It is intentionally lightweight so it can be run after a checkpoint has
been trained or restored.
"""

import os
import gc
import json
from dataclasses import dataclass
from typing import Optional, List

import torch
import numpy as np
from omegaconf import OmegaConf

from ov_piano import PIANO_MIDI_RANGE, HDF5PathManager
from ov_piano.utils import format_load_model_warnings, load_model
from ov_piano.custom_logging import ColorLogger
from ov_piano.data.maestro import MetaMAESTROv1, MetaMAESTROv2, MetaMAESTROv3
from ov_piano.data.maestro import MelMaestro
from ov_piano.models.ov import OnsetsAndVelocities
from ov_piano.inference import (
    strided_inference,
    OnsetVelocityNmsDecoder,
    model_outputs_to_probabilities,
)
from ov_piano.eval import (
    EvaluationCheckpointStore,
    GtLoaderMaestro,
    evaluation_checkpoint_path,
    evaluation_fingerprint,
    metadata_to_file_id,
    threshold_eval_single_file,
    threshold_eval_pedals,
)


@dataclass
class ConfDef:
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    MAESTRO_PATH: str = os.path.join("datasets", "maestro", "maestro-v3.0.0")
    MAESTRO_VERSION: int = 3
    OUTPUT_DIR: str = "out"
    EVALUATION_CHECKPOINTS_ENABLED: bool = True
    EVALUATION_CHECKPOINT_DIR: Optional[str] = None
    RESET_EVALUATION_CHECKPOINTS: bool = False
    HDF5_MEL_PATH: str = os.path.join(
        "datasets",
        "MAESTROv3_logmel_sr=16000_stft=2048w384h_mel=229(50-8000).h5")
    HDF5_ROLL_PATH: str = os.path.join(
        "datasets",
        "MAESTROv3_roll_quant=0.024_midivals=128_extendsus=True.h5")
    SNAPSHOT_INPATH: str = os.path.join(
        "assets",
        "OnsetsAndVelocities_2026_07_12_08_12_56.769.torch")
    CONV1X1: List[int] = (200, 200)
    LEAKY_RELU_SLOPE: Optional[float] = 0.1
    SEARCH_THRESHOLDS: List[float] = (0.75,)
    SEARCH_SHIFTS: List[float] = (-0.01,)
    PEDAL_THRESHOLD: float = 0.5
    PEDAL_HYSTERESIS: float = 0.1
    PEDAL_SMOOTHING_WINDOW: int = 3
    PEDAL_MIN_HOLD_STEPS: int = 2
    PEDAL_SHIFT: float = 0.0
    MAX_PREDICTIONS_PER_FILE: int = 20000
    DECODER_GAUSS_STD: float = 1
    DECODER_GAUSS_KSIZE: int = 11
    TOLERANCE_SECS: float = 0.05
    TOLERANCE_VEL: float = 0.1
    INFERENCE_CHUNK_SIZE: float = 60
    INFERENCE_CHUNK_OVERLAP: float = 11


if __name__ == "__main__":
    CONF = OmegaConf.structured(ConfDef())
    cli_conf = OmegaConf.from_cli()
    CONF = OmegaConf.merge(CONF, cli_conf)

    (DATASET_NAME, SAMPLERATE, WINSIZE, HOPSIZE,
     MELBINS, FMIN, FMAX) = HDF5PathManager.parse_mel_hdf5_basename(
        os.path.basename(CONF.HDF5_MEL_PATH))
    roll_params = HDF5PathManager.parse_roll_hdf5_basename(
        os.path.basename(CONF.HDF5_ROLL_PATH))
    SECS_PER_FRAME = HOPSIZE / SAMPLERATE

    CHUNK_SIZE = round(CONF.INFERENCE_CHUNK_SIZE / SECS_PER_FRAME)
    CHUNK_OVERLAP = round(CONF.INFERENCE_CHUNK_OVERLAP / SECS_PER_FRAME)

    assert DATASET_NAME == roll_params[0], "Inconsistent HDF5 datasets?"
    assert SECS_PER_FRAME == roll_params[1], "Inconsistent roll quantization?"
    assert (CHUNK_OVERLAP % 2) == 0, f"Only even overlap allowed! {CHUNK_OVERLAP}"

    METAMAESTRO_CLASS = {1: MetaMAESTROv1, 2: MetaMAESTROv2, 3: MetaMAESTROv3}[CONF.MAESTRO_VERSION]
    TXT_LOG_OUTDIR = os.path.join(CONF.OUTPUT_DIR, "txt_logs")
    os.makedirs(TXT_LOG_OUTDIR, exist_ok=True)

    txt_logger = ColorLogger(os.path.basename(__file__), TXT_LOG_OUTDIR)
    txt_logger.info("\n\nCONFIGURATION:\n" + OmegaConf.to_yaml(CONF) + "\n\n")

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

    txt_logger.info("Loading test split datasets")
    metamaestro_test = METAMAESTRO_CLASS(CONF.MAESTRO_PATH, splits=["test"], years=METAMAESTRO_CLASS.ALL_YEARS)
    maestro_test = MelMaestro(
        CONF.HDF5_MEL_PATH,
        CONF.HDF5_ROLL_PATH,
        *(x[0] for x in metamaestro_test.data),
        as_torch_tensors=False)

    txt_logger.info("Loading test ground truths")
    test_gts = GtLoaderMaestro(maestro_test, metamaestro_test)

    txt_logger.info("Loading NN")
    num_mels = maestro_test[0][0].shape[0]
    key_beg, key_end = PIANO_MIDI_RANGE
    num_piano_keys = key_end - key_beg

    model = OnsetsAndVelocities(
        in_chans=2,
        in_height=num_mels,
        out_height=num_piano_keys,
        conv1x1head=CONF.CONV1X1,
        bn_momentum=0,
        leaky_relu_slope=CONF.LEAKY_RELU_SLOPE,
        dropout_drop_p=0).to(CONF.DEVICE)
    load_report = load_model(model, CONF.SNAPSHOT_INPATH, eval_phase=True, strict=False)
    for warning in format_load_model_warnings(load_report):
        txt_logger.warning(f"CHECKPOINT LOAD WARNING: {warning}")

    decoder = OnsetVelocityNmsDecoder(
        num_piano_keys,
        nms_pool_ksize=3,
        gauss_conv_stddev=CONF.DECODER_GAUSS_STD,
        gauss_conv_ksize=CONF.DECODER_GAUSS_KSIZE,
        vel_pad_left=1,
        vel_pad_right=1)
    def model_inference(x):
        try:
            return model_outputs_to_probabilities(model(x), include_pedals=True)
        except Exception as e:
            txt_logger.error(f"model_inference exception: {e}")
            return (), (), ()

    len_test = len(maestro_test)
    test_file_ids = [metadata_to_file_id(md) for _, _, md in maestro_test.data]
    test_inference_store = _make_checkpoint_store(
        "test_inference",
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
        },
    )
    test_dataframes = []
    for i, (mel, roll, md) in enumerate(maestro_test, 1):
        file_id = metadata_to_file_id(md)
        txt_logger.info(f"[{i}/{len_test}] Test inference: {md}")
        try:
            cached_entry = test_inference_store.get(file_id)
            if isinstance(cached_entry, dict):
                if cached_entry.get("status") == "ok":
                    txt_logger.info(f"  Test inference checkpoint hit: {file_id}")
                    test_dataframes.append(
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
                        f"  Test inference checkpoint skip: {file_id} "
                        f"({cached_entry.get('reason', 'unknown')})"
                    )
                    continue

            with torch.no_grad():
                tmel = torch.from_numpy(mel).to(CONF.DEVICE).unsqueeze(0)
                if tmel.shape[-1] == 0:
                    txt_logger.warning(f"SKIPPING {md[0]}: Empty mel input")
                    test_inference_store.upsert(
                        file_id,
                        {"status": "skipped", "metadata": tuple(md), "reason": "empty_mel"},
                    )
                    del tmel
                    continue
                _res = strided_inference(model_inference, tmel, CHUNK_SIZE, CHUNK_OVERLAP)
                del tmel
                if not _res or len(_res) < 3:
                    txt_logger.error(f"SKIPPING {md[0]}: invalid inference result")
                    test_inference_store.upsert(
                        file_id,
                        {
                            "status": "skipped",
                            "metadata": tuple(md),
                            "reason": "invalid_inference_result",
                        },
                    )
                    continue

                onset_pred, vel_pred, pedal_pred = _res
                pred_df = decoder(onset_pred, vel_pred, pthresh=min(CONF.SEARCH_THRESHOLDS))
                if pedal_pred.dim() == 2:
                    pedal_pred = pedal_pred.unsqueeze(0)
                pedal_pred = pedal_pred.cpu()

                gt_df = test_gts(md)[0]
                gt_pedal_df = test_gts.get_sus_pedal_events(md, SECS_PER_FRAME)

                num_preds = len(pred_df)
                num_gt = len(gt_df)
                if num_preds > CONF.MAX_PREDICTIONS_PER_FILE:
                    txt_logger.warning(
                        f"SKIPPING {md[0]}: Too many predictions ({num_preds:,}) vs {num_gt:,} ground truth")
                    test_inference_store.upsert(
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

                test_inference_store.upsert(
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
                test_dataframes.append((file_id, gt_df, pred_df, gt_pedal_df, pedal_pred))
        except Exception as e:
            txt_logger.error(f"ERROR processing {md[0]}: {e}")
            continue
        finally:
            for v in (
                "mel", "roll", "onset_pred", "vel_pred", "pedal_pred",
                "tmel", "_res", "pred_df", "gt_df", "gt_pedal_df",
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

    if len(test_dataframes) == 0:
        raise RuntimeError("No valid test files processed for evaluation")

    txt_logger.info(f"Successfully processed {len(test_dataframes)} / {len_test} test files")

    test_grid_store = _make_checkpoint_store(
        "test_grid",
        {
            "source_stage": "test_inference",
            "source_fingerprint": test_inference_store.fingerprint,
            "file_ids": [item[0] for item in test_dataframes],
            "thresholds": list(CONF.SEARCH_THRESHOLDS),
            "shifts": list(CONF.SEARCH_SHIFTS),
            "secs_per_frame": float(SECS_PER_FRAME),
            "key_beg": int(key_beg),
            "tolerance_secs": float(CONF.TOLERANCE_SECS),
            "tolerance_vel": float(CONF.TOLERANCE_VEL),
            "pedal_threshold": float(CONF.PEDAL_THRESHOLD),
            "pedal_shift": float(CONF.PEDAL_SHIFT),
            "pedal_hysteresis": float(CONF.PEDAL_HYSTERESIS),
            "pedal_smoothing_window": int(CONF.PEDAL_SMOOTHING_WINDOW),
            "pedal_min_hold_steps": int(CONF.PEDAL_MIN_HOLD_STEPS),
        },
    )
    summary = []
    for thresh in CONF.SEARCH_THRESHOLDS:
        for shift in CONF.SEARCH_SHIFTS:
            checkpoint_key = json.dumps((float(thresh), float(shift)))
            cached_grid = test_grid_store.get(checkpoint_key)
            if isinstance(cached_grid, dict) and cached_grid.get("status") == "ok":
                txt_logger.info(f"Test grid checkpoint hit: {(thresh, shift)}")
                summary.append(
                    (
                        thresh,
                        shift,
                        np.asarray(cached_grid["onset_metrics"], dtype=float),
                        np.asarray(cached_grid["velocity_metrics"], dtype=float),
                        np.asarray(cached_grid["pedal_metrics"], dtype=float),
                    )
                )
                continue

            evals = []
            evals_vel = []
            evals_pedal = []
            for _, gtdf, preddf, gt_pedal_df, pedal_pred in test_dataframes:
                prf1, prf1_v = threshold_eval_single_file(
                    gtdf,
                    preddf,
                    SECS_PER_FRAME,
                    key_beg,
                    thresh=thresh,
                    shift_preds=shift,
                    tol_secs=CONF.TOLERANCE_SECS,
                    tol_vel=CONF.TOLERANCE_VEL)
                try:
                    pedal_results = threshold_eval_pedals(
                        gt_pedal_df,
                        pedal_pred,
                        SECS_PER_FRAME,
                        thresh=CONF.PEDAL_THRESHOLD,
                        shift_preds=CONF.PEDAL_SHIFT,
                        tol_secs=CONF.TOLERANCE_SECS,
                        hysteresis=CONF.PEDAL_HYSTERESIS,
                        smoothing_window=CONF.PEDAL_SMOOTHING_WINDOW,
                        min_hold_steps=CONF.PEDAL_MIN_HOLD_STEPS)
                    pedal_prf1 = (pedal_results["sustain"]["precision"],
                                  pedal_results["sustain"]["recall"],
                                  pedal_results["sustain"]["f1"])
                except Exception as e:
                    txt_logger.warning(f"Pedal eval failed: {e}")
                    pedal_prf1 = (0.0, 0.0, 0.0)
                evals.append(prf1)
                evals_vel.append(prf1_v)
                evals_pedal.append(pedal_prf1)

            onset_metrics = np.mean(evals, axis=0)
            velocity_metrics = np.mean(evals_vel, axis=0)
            pedal_metrics = np.mean(evals_pedal, axis=0)
            test_grid_store.upsert(
                checkpoint_key,
                {
                    "status": "ok",
                    "threshold": float(thresh),
                    "shift": float(shift),
                    "onset_metrics": onset_metrics.tolist(),
                    "velocity_metrics": velocity_metrics.tolist(),
                    "pedal_metrics": pedal_metrics.tolist(),
                },
            )
            summary.append((thresh, shift, onset_metrics, velocity_metrics, pedal_metrics))

    for thresh, shift, onset_metrics, vel_metrics, pedal_metrics in summary:
        txt_logger.info(
            f"(t={thresh}, s={shift}) ONSETS {onset_metrics} VEL {vel_metrics} PEDAL {pedal_metrics}")
