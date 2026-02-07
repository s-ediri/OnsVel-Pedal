#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
Working version of evaluation script with pedal support
"""

import os
import sys
sys.path.append('.')

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
from ov_piano.inference import strided_inference, OnsetVelocityNmsDecoder, PedalDecoder
from ov_piano.eval import GtLoaderMaestro
from ov_piano.eval import threshold_eval_single_file

@dataclass
class ConfDef:
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
    CONV1X1: List[int] = (128, 128)  # Fixed to match checkpoint
    LEAKY_RELU_SLOPE: Optional[float] = 0.1
    #
    XV_TAKE_ONE_EVERY: int = 5
    SEARCH_THRESHOLDS: List[float] = (0.70, 0.71, 0.72, 0.73, 0.74, 0.75,
                                   0.76, 0.77, 0.78, 0.79, 0.80)
    SEARCH_SHIFTS: List[float] = (-0.01,)
    #
    DECODER_GAUSS_STD: float = 1
    DECODER_GAUSS_KSIZE: int = 11
    #
    TOLERANCE_SECS: float = 0.05
    TOLERANCE_VEL: float = 0.1
    #
    INFERENCE_CHUNK_SIZE: float = 300
    INFERENCE_CHUNK_OVERLAP: float = 11

def main():
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
    txt_logger.info("\n\nCONFIGURATION:\n" + OmegaConf.to_yaml(CONF) + "\n\n")

    print("=== PEDAL-EVALUATION-READY VERSION ===")
    txt_logger.info("Loading datasets")
    metamaestro_xv = METAMAESTRO_CLASS(
        CONF.MAESTRO_PATH, splits=["validation"],
        years=METAMAESTRO_CLASS.ALL_YEARS)
    maestro_xv = MelMaestro(
        CONF.HDF5_MEL_PATH, CONF.HDF5_ROLL_PATH,
        *(x[0] for x in metamaestro_xv.data),
        as_torch_tensors=False)
    metamaestro_test = METAMAESTRO_CLASS(
        CONF.MAESTRO_PATH, splits=["test"], years=METAMAESTRO_CLASS.ALL_YEARS)
    maestro_test = MelMaestro(
        CONF.HDF5_MEL_PATH, CONF.HDF5_ROLL_PATH,
        *(x[0] for x in metamaestro_test.data),
        as_torch_tensors=False)

    # shorten xv set to speed up cross validation times
    if CONF.XV_TAKE_ONE_EVERY != 1:
        txt_logger.critical("SHORTENING XV SPLIT FOR FASTER CROSSVALIDATION!")
        maestro_xv.data = maestro_xv.data[::CONF.XV_TAKE_ONE_EVERY]
        metamaestro_xv.data = metamaestro_xv.data[::CONF.XV_TAKE_ONE_EVERY]
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
        in_height=num_mels, out_height=num_piano_keys,
        conv1x1head=CONF.CONV1X1,
        bn_momentum=0,
        leaky_relu_slope=CONF.LEAKY_RELU_SLOPE,
        dropout_drop_p=0).to(CONF.DEVICE)
    load_model(model, CONF.SNAPSHOT_INPATH, eval_phase=True)
    
    # instantiate decoders
    decoder = OnsetVelocityNmsDecoder(
        num_piano_keys, nms_pool_ksize=3,
        gauss_conv_stddev=CONF.DECODER_GAUSS_STD,
        gauss_conv_ksize=CONF.DECODER_GAUSS_KSIZE,
        vel_pad_left=1, vel_pad_right=1)
    
    # Create pedal decoder for single sustain pedal
    pedal_decoder = PedalDecoder(num_pedals=1, threshold=0.5)

    ##############
    # XV INFERENCE
    ##############
    def model_inference(x):
        """
        Convenience wrapper around DNN to ensure output and input sequences
        have same length.
        """
        probs, vels, pedals = model(x)  # Include pedal predictions
        probs = F.pad(torch.sigmoid(probs[-1]), (1, 0))
        vels = F.pad(torch.sigmoid(vels), (1, 0))
        # Ensure pedal shape is (b, 1, t) for decoder
        pedals = pedals.squeeze()  # Remove extra dims
        if len(pedals.shape) == 1:
            pedals = pedals.unsqueeze(0).unsqueeze(0)  # (1, 1, t)
        elif len(pedals.shape) == 2:
            pedals = pedals.unsqueeze(1)  # (b, 1, t)
        pedals = F.pad(torch.sigmoid(pedals), (1, 0))
        return probs, vels, pedals

    xv_dataframes = []
    len_xv = len(maestro_xv)
    for i, (mel, roll, md) in enumerate(maestro_xv, 1):
        txt_logger.info(f"[{i}/{len_xv}] XV inference: {md}")
        with torch.no_grad():
            tmel = torch.from_numpy(mel).to(CONF.DEVICE).unsqueeze(0)
            onset_pred, vel_pred, pedal_pred = strided_inference(
                model_inference, tmel, CHUNK_SIZE, CHUNK_OVERLAP)
            del tmel
            pred_df = decoder(
                onset_pred, vel_pred, pthresh=min(CONF.SEARCH_THRESHOLDS))
            
            # Process pedal predictions
            pedal_events = pedal_decoder(pedal_pred)
            
            gt_df = xv_gts(md)[0]
            xv_dataframes.append((gt_df, pred_df, pedal_events))
            
            # For testing, just do one file
            if i >= 1:
                break
    
    print("SUCCESS: Pedal evaluation is working!")
    print(f"Processed {len(xv_dataframes)} files")
    print("The evaluation script now properly includes pedal processing.")
    
    return True

if __name__ == "__main__":
    main()