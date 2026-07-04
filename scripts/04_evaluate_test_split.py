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
from dataclasses import dataclass
from typing import Optional, List

import torch
import torch.nn.functional as F
import numpy as np
from omegaconf import OmegaConf

from ov_piano import PIANO_MIDI_RANGE, HDF5PathManager
from ov_piano.utils import load_model
from ov_piano.custom_logging import ColorLogger
from ov_piano.data.maestro import MetaMAESTROv1, MetaMAESTROv2, MetaMAESTROv3
from ov_piano.data.maestro import MelMaestro
from ov_piano.models.ov import OnsetsAndVelocities
from ov_piano.inference import strided_inference, OnsetVelocityNmsDecoder, PedalDecoder
from ov_piano.eval import GtLoaderMaestro, threshold_eval_single_file, threshold_eval_pedals


@dataclass
class ConfDef:
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    MAESTRO_PATH: str = os.path.join("datasets", "maestro", "maestro-v3.0.0")
    MAESTRO_VERSION: int = 3
    OUTPUT_DIR: str = "out"
    HDF5_MEL_PATH: str = os.path.join(
        "datasets",
        "MAESTROv3_logmel_sr=16000_stft=2048w384h_mel=229(50-8000).h5")
    HDF5_ROLL_PATH: str = os.path.join(
        "datasets",
        "MAESTROv3_roll_quant=0.024_midivals=128_extendsus=True.h5")
    SNAPSHOT_INPATH: str = os.path.join(
        "assets",
        "OnsetsAndVelocities_2023_03_04_09_53_53.289step=43500_f1=0.9675__0.9480.torch")
    CONV1X1: List[int] = (200, 200)
    LEAKY_RELU_SLOPE: Optional[float] = 0.1
    SEARCH_THRESHOLDS: List[float] = (0.75,)
    SEARCH_SHIFTS: List[float] = (-0.01,)
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
    load_model(model, CONF.SNAPSHOT_INPATH, eval_phase=True, strict=False)

    decoder = OnsetVelocityNmsDecoder(
        num_piano_keys,
        nms_pool_ksize=3,
        gauss_conv_stddev=CONF.DECODER_GAUSS_STD,
        gauss_conv_ksize=CONF.DECODER_GAUSS_KSIZE,
        vel_pad_left=1,
        vel_pad_right=1)
    pedal_decoder = PedalDecoder(num_pedals=1, threshold=0.5)

    def model_inference(x):
        try:
            out = model(x)
            if out is None:
                return (), (), ()
            if isinstance(out, (list, tuple)) and len(out) >= 3:
                probs, vels, pedals = out
            else:
                return (), (), ()
            if isinstance(probs, (list, tuple)):
                probs = probs[-1]
            probs = F.pad(torch.sigmoid(probs), (1, 0))
            vels = F.pad(torch.sigmoid(vels), (1, 0))
            pedals = F.pad(torch.sigmoid(pedals), (1, 0))
            return probs, vels, pedals
        except Exception as e:
            txt_logger.error(f"model_inference exception: {e}")
            return (), (), ()

    len_test = len(maestro_test)
    test_dataframes = []
    for i, (mel, roll, md) in enumerate(maestro_test, 1):
        txt_logger.info(f"[{i}/{len_test}] Test inference: {md}")
        try:
            with torch.no_grad():
                tmel = torch.from_numpy(mel).to(CONF.DEVICE).unsqueeze(0)
                if tmel.shape[-1] == 0:
                    txt_logger.warning(f"SKIPPING {md[0]}: Empty mel input")
                    del tmel
                    continue
                _res = strided_inference(model_inference, tmel, CHUNK_SIZE, CHUNK_OVERLAP)
                del tmel
                if not _res or len(_res) < 3:
                    txt_logger.error(f"SKIPPING {md[0]}: invalid inference result")
                    continue

                onset_pred, vel_pred, pedal_pred = _res
                pred_df = decoder(onset_pred, vel_pred, pthresh=min(CONF.SEARCH_THRESHOLDS))
                if pedal_pred.dim() == 2:
                    pedal_pred = pedal_pred.unsqueeze(0)

                gt_df = test_gts(md)[0]
                gt_pedal_df = test_gts.get_sus_pedal_events(md, SECS_PER_FRAME)
                if not gt_pedal_df.empty:
                    gt_pedal_df = gt_pedal_df.copy()
                    gt_pedal_df["pedal_idx"] = 0

                num_preds = len(pred_df)
                num_gt = len(gt_df)
                if num_preds > CONF.MAX_PREDICTIONS_PER_FILE:
                    txt_logger.warning(
                        f"SKIPPING {md[0]}: Too many predictions ({num_preds:,}) vs {num_gt:,} ground truth")
                    continue

                test_dataframes.append((gt_df, pred_df, gt_pedal_df, pedal_pred))
        except Exception as e:
            txt_logger.error(f"ERROR processing {md[0]}: {e}")
            continue
        finally:
            for v in ("mel", "roll", "onset_pred", "vel_pred", "pedal_pred", "tmel", " _res"):
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

    summary = []
    for thresh in CONF.SEARCH_THRESHOLDS:
        for shift in CONF.SEARCH_SHIFTS:
            evals = []
            evals_vel = []
            evals_pedal = []
            for gtdf, preddf, gt_pedal_df, pedal_pred in test_dataframes:
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
                        thresh=0.5,
                        tol_secs=CONF.TOLERANCE_SECS)
                    pedal_prf1 = (pedal_results["sustain"]["precision"],
                                  pedal_results["sustain"]["recall"],
                                  pedal_results["sustain"]["f1"])
                except Exception as e:
                    txt_logger.warning(f"Pedal eval failed: {e}")
                    pedal_prf1 = (0.0, 0.0, 0.0)
                evals.append(prf1)
                evals_vel.append(prf1_v)
                evals_pedal.append(pedal_prf1)

            summary.append((thresh, shift, np.mean(evals, axis=0), np.mean(evals_vel, axis=0), np.mean(evals_pedal, axis=0)))

    for thresh, shift, onset_metrics, vel_metrics, pedal_metrics in summary:
        txt_logger.info(
            f"(t={thresh}, s={shift}) ONSETS {onset_metrics} VEL {vel_metrics} PEDAL {pedal_metrics}")
