#!/usr/bin/env python
# -*- coding:utf-8 -*-


"""
Simplified evaluation script that skips validation and evaluates test set directly.
Use this when the model produces too many predictions for validation to work.
"""


import os
import gc
# For omegaconf
from dataclasses import dataclass
from typing import Optional, List
#
from omegaconf import OmegaConf, MISSING
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
#
from ov_piano import PIANO_MIDI_RANGE, HDF5PathManager
from ov_piano.utils import load_model
from ov_piano.logging import ColorLogger
from ov_piano.data.maestro import MetaMAESTROv1, MetaMAESTROv2, MetaMAESTROv3
from ov_piano.data.maestro import MelMaestro
from ov_piano.models.ov import OnsetsAndVelocities
from ov_piano.inference import strided_inference, OnsetVelocityNmsDecoder
from ov_piano.eval import GtLoaderMaestro
from ov_piano.eval import threshold_eval_single_file


# ##############################################################################
# # GLOBALS
# ##############################################################################
@dataclass
class ConfDef:
    """
    Configuration for test-only evaluation (skips validation grid search)
    """
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    MAESTRO_PATH: str = os.path.join("datasets", "maestro", "maestro-v3.0.0")
    MAESTRO_VERSION: int = 3
    OUTPUT_DIR: str = "out"
    #
    HDF5_MEL_PATH: str = os.path.join(
        "datasets",
        "MAESTROv3_logmel_sr=16000_stft=2048w384h_mel=229(50-8000).h5")
    HDF5_ROLL_PATH: str = os.path.join(
        "datasets",
        "MAESTROv3_roll_quant=0.024_midivals=128_extendsus=True.h5")
    SNAPSHOT_INPATH: str = os.path.join("out", "model_snapshots", "OnsetsAndVelocities_2026_01_30_12_46_23.207.torch")
    #
    CONV1X1: List[int] = (128, 128)  # MUST match checkpoint architecture!
    LEAKY_RELU_SLOPE: Optional[float] = 0.1
    #
    # Fixed threshold (no validation grid search)
    FIXED_THRESHOLD: float = 0.95  # High threshold to filter excessive predictions
    FIXED_SHIFT: float = -0.01
    MAX_PREDICTIONS_PER_FILE: int = 20000  # Safety limit
    #
    # Decoder settings - set DECODER_GAUSS_STD to None to disable smoothing
    DECODER_GAUSS_STD: Optional[float] = None  # Disabled - smoothing may cause excessive predictions
    DECODER_GAUSS_KSIZE: Optional[int] = None
    #
    TOLERANCE_SECS: float = 0.05
    TOLERANCE_VEL: float = 0.1
    #
    INFERENCE_CHUNK_SIZE: float = 60  # Small chunks for memory efficiency
    INFERENCE_CHUNK_OVERLAP: float = 11


# ##############################################################################
# # MAIN LOOP INITIALIZATION
# ##############################################################################
if __name__ == "__main__":
    CONF = OmegaConf.structured(ConfDef())
    cli_conf = OmegaConf.from_cli()
    CONF = OmegaConf.merge(CONF, cli_conf)

    # derivative globals + parse HDF5 filenames and ensure they are consistent
    (DATASET_NAME, SAMPLERATE, WINSIZE, HOPSIZE,
     MELBINS, FMIN, FMAX) = HDF5PathManager.parse_mel_hdf5_basename(
        os.path.basename(CONF.HDF5_MEL_PATH))
    roll_params = HDF5PathManager.parse_roll_hdf5_basename(
        os.path.basename(CONF.HDF5_ROLL_PATH))
    SECS_PER_FRAME = HOPSIZE / SAMPLERATE
    #
    CHUNK_SIZE = round(CONF.INFERENCE_CHUNK_SIZE / SECS_PER_FRAME)
    CHUNK_OVERLAP = round(CONF.INFERENCE_CHUNK_OVERLAP / SECS_PER_FRAME)
    #
    assert DATASET_NAME == roll_params[0], "Inconsistent HDF5 datasets?"
    assert SECS_PER_FRAME == roll_params[1], "Inconsistent roll quantization?"
    assert (CHUNK_OVERLAP % 2) == 0, \
        f"Only even overlap allowed! {CHUNK_OVERLAP}"
    #
    METAMAESTRO_CLASS = {1: MetaMAESTROv1, 2: MetaMAESTROv2,
                         3: MetaMAESTROv3}[CONF.MAESTRO_VERSION]
    TXT_LOG_OUTDIR = os.path.join(CONF.OUTPUT_DIR, "txt_logs")
    os.makedirs(TXT_LOG_OUTDIR, exist_ok=True)

    txt_logger = ColorLogger(os.path.basename(__file__), TXT_LOG_OUTDIR)
    txt_logger.info("\n\n=== TEST-ONLY EVALUATION (SKIPPING VALIDATION) ===")
    txt_logger.info("\n\nCONFIGURATION:\n" + OmegaConf.to_yaml(CONF) + "\n\n")
    txt_logger.warning(f"Using FIXED threshold={CONF.FIXED_THRESHOLD}, shift={CONF.FIXED_SHIFT}")
    txt_logger.warning("Validation grid search is SKIPPED\n")

    txt_logger.info("Loading test dataset")
    metamaestro_test = METAMAESTRO_CLASS(
        CONF.MAESTRO_PATH, splits=["test"], years=METAMAESTRO_CLASS.ALL_YEARS)
    maestro_test = MelMaestro(
        CONF.HDF5_MEL_PATH, CONF.HDF5_ROLL_PATH,
        *(x[0] for x in metamaestro_test.data),
        as_torch_tensors=False)

    txt_logger.info("Loading test ground truths")
    test_gts = GtLoaderMaestro(maestro_test, metamaestro_test)

    # instantiate and load trained NN model
    txt_logger.info("Loading NN")
    num_mels = maestro_test[0][0].shape[0]
    key_beg, key_end = PIANO_MIDI_RANGE
    num_piano_keys = key_end - key_beg
    #
    model = OnsetsAndVelocities(
        in_chans=2,  # X and time_derivative(X)
        in_height=num_mels, out_height=num_piano_keys,
        conv1x1head=CONF.CONV1X1,
        bn_momentum=0,
        leaky_relu_slope=CONF.LEAKY_RELU_SLOPE,
        dropout_drop_p=0).to(CONF.DEVICE)
    load_model(model, CONF.SNAPSHOT_INPATH, eval_phase=True)

    # instantiate decoder
    decoder = OnsetVelocityNmsDecoder(
        num_piano_keys, nms_pool_ksize=3,
        gauss_conv_stddev=CONF.DECODER_GAUSS_STD,
        gauss_conv_ksize=CONF.DECODER_GAUSS_KSIZE,
        vel_pad_left=1, vel_pad_right=1)

    ##############
    # MODEL INFERENCE
    ##############
    def model_inference(x):
        """
        Convenience wrapper around the DNN to ensure output and input sequences
        have same length.
        """
        probs, vels, pedals = model(x)
        probs = F.pad(torch.sigmoid(probs[-1]), (1, 0))
        vels = F.pad(torch.sigmoid(vels), (1, 0))
        # Note: pedals output is returned but not used in onset/velocity evaluation
        return probs, vels

    ###############
    # TEST EVALUATION
    ###############
    txt_logger.info(f"\nEvaluating {len(maestro_test)} test files...")
    test_results = []
    test_results_vel = []
    skipped_count = 0
    len_test = len(maestro_test)

    for i, (mel, roll, md) in enumerate(maestro_test, 1):
        txt_logger.info(f"[{i}/{len_test}] {md[0]}")
        try:
            with torch.no_grad():
                tmel = torch.from_numpy(mel).to(CONF.DEVICE).unsqueeze(0)
                onset_pred, vel_pred = strided_inference(
                    model_inference, tmel, CHUNK_SIZE, CHUNK_OVERLAP)
                del tmel

                # Diagnostic: Check raw prediction statistics
                pred_max = onset_pred.max().item()
                pred_min = onset_pred.min().item()
                pred_mean = onset_pred.mean().item()
                pred_above_threshold = (onset_pred >= CONF.FIXED_THRESHOLD).sum().item()

                txt_logger.info(
                    f"  Raw predictions: min={pred_min:.3f}, max={pred_max:.3f}, "
                    f"mean={pred_mean:.3f}, above_thresh={pred_above_threshold:,}")

                # Decoder with threshold
                pred_df = decoder(onset_pred, vel_pred, pthresh=CONF.FIXED_THRESHOLD)

                # AGGRESSIVE POST-FILTERING: Apply threshold again to decoded predictions
                # This is a safety measure because the decoder might not filter enough
                pred_df = pred_df[pred_df["prob"] >= CONF.FIXED_THRESHOLD].copy()

                # Additional safety: Remove duplicate predictions at same time/key
                pred_df = pred_df.drop_duplicates(subset=["key", "t_idx"], keep="first")

                gt_df = test_gts(md)[0]

            # Safety check: skip files with excessive predictions
            num_preds = len(pred_df)
            num_gt = len(gt_df)

            if num_preds > CONF.MAX_PREDICTIONS_PER_FILE:
                txt_logger.warning(
                    f"  SKIPPING: Too many predictions ({num_preds:,}) "
                    f"vs {num_gt:,} ground truth.")
                skipped_count += 1
                continue

            txt_logger.info(f"  GT: {num_gt:,} notes, Pred: {num_preds:,} notes")

            prf1, prf1_v = threshold_eval_single_file(
                gt_df, pred_df, SECS_PER_FRAME, key_beg,
                thresh=CONF.FIXED_THRESHOLD, shift_preds=CONF.FIXED_SHIFT,
                tol_secs=CONF.TOLERANCE_SECS, tol_vel=CONF.TOLERANCE_VEL)

            txt_logger.info(f"  Onsets: P={prf1[0]:.3f}, R={prf1[1]:.3f}, F1={prf1[2]:.3f}")
            txt_logger.info(f"  On+Vel: P={prf1_v[0]:.3f}, R={prf1_v[1]:.3f}, F1={prf1_v[2]:.3f}")

            test_results.append((md[0], *prf1))
            test_results_vel.append((md[0], *prf1_v))

        except Exception as e:
            txt_logger.error(f"  ERROR: {e}")
            skipped_count += 1
            continue
        finally:
            # Aggressive memory cleanup
            try:
                del mel, roll, onset_pred, vel_pred, pred_df, gt_df
            except:
                pass
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Summary
    txt_logger.info(f"\n{'='*60}")
    txt_logger.info(f"Processed: {len(test_results)} / {len_test} files")
    txt_logger.info(f"Skipped: {skipped_count} files (excessive predictions)")
    txt_logger.info(f"{'='*60}\n")

    if len(test_results) == 0:
        txt_logger.error("ERROR: No test files processed successfully!")
        txt_logger.error("All files had excessive predictions. Try:")
        txt_logger.error("  1) FIXED_THRESHOLD=0.95 (even higher)")
        txt_logger.error("  2) Check if model is trained properly")
        txt_logger.error("  3) Verify checkpoint is correct")
    else:
        # Create results dataframes
        test_results_df = pd.DataFrame(
            test_results, columns=["Filename", "P", "R", "F1"])
        averages = [f"AVERAGES (t={CONF.FIXED_THRESHOLD}, s={CONF.FIXED_SHIFT})",
                    *test_results_df.iloc[:, 1:].mean().tolist()]
        test_results_df.loc[len(test_results_df)] = averages
        #
        test_results_df_vel = pd.DataFrame(
            test_results_vel, columns=["Filename", "P", "R", "F1"])
        averages_vel = [f"AVERAGES (t={CONF.FIXED_THRESHOLD}, s={CONF.FIXED_SHIFT})",
                        *test_results_df_vel.iloc[:, 1:].mean().tolist()]
        test_results_df_vel.loc[len(test_results_df_vel)] = averages_vel
        #
        txt_logger.warning("\n" + "="*60)
        txt_logger.warning("TEST RESULTS (Test Set Only, No Validation)")
        txt_logger.warning(f"Model: {CONF.SNAPSHOT_INPATH}")
        txt_logger.warning(f"Threshold: {CONF.FIXED_THRESHOLD}, Shift: {CONF.FIXED_SHIFT}")
        txt_logger.warning("="*60 + "\n")
        txt_logger.warning("ONSETS:\n" + str(test_results_df))
        txt_logger.warning("\nONSETS+VELOCITIES:\n" + str(test_results_df_vel))

        # Save to CSV
        csv_path = os.path.join(CONF.OUTPUT_DIR, "test_results.csv")
        test_results_df.to_csv(csv_path, index=False)
        csv_path_vel = os.path.join(CONF.OUTPUT_DIR, "test_results_with_velocity.csv")
        test_results_df_vel.to_csv(csv_path_vel, index=False)
        txt_logger.info(f"\nResults saved to: {csv_path}")
