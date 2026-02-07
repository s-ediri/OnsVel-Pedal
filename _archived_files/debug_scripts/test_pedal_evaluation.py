#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
Test script to verify pedal evaluation integration works correctly
"""

import os
import sys
sys.path.append('.')

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from dataclasses import dataclass
from typing import Optional, List

from ov_piano import PIANO_MIDI_RANGE
from ov_piano.utils import load_model
from ov_piano.models.ov import OnsetsAndVelocities
from ov_piano.inference import OnsetVelocityNmsDecoder, PedalDecoder

@dataclass
class ConfDef:
    DEVICE: str = "cpu"
    HDF5_MEL_PATH: str = os.path.join(
        "datasets",
        "MAESTROv3_logmel_sr=16000_stft=2048w384h_mel=229(50-8000).h5")
    HDF5_ROLL_PATH: str = os.path.join(
        "datasets",
        "MAESTROv3_roll_quant=0.024_midivals=128_extendsus=True.h5")
    SNAPSHOT_INPATH: str = os.path.join("out", "model_snapshots", "OnsetsAndVelocities_2026_01_30_12_46_23.207.torch")
    CONV1X1: List[int] = (128, 128)
    LEAKY_RELU_SLOPE: Optional[float] = 0.1
    #
    SEARCH_THRESHOLDS: List[float] = (0.70, 0.71, 0.72, 0.73, 0.74, 0.75,
                                   0.76, 0.77, 0.78, 0.79, 0.80)
    SEARCH_SHIFTS: List[float] = (-0.01,)
    #
    DECODER_GAUSS_STD: float = 1
    DECODER_GAUSS_KSIZE: int = 11
    #
    INFERENCE_CHUNK_SIZE: float = 300
    INFERENCE_CHUNK_OVERLAP: float = 11

def main():
    CONF = OmegaConf.structured(ConfDef())
    
    # Model setup
    num_mels = 229
    key_beg, key_end = PIANO_MIDI_RANGE
    num_piano_keys = key_end - key_beg
    
    print("Creating model...")
    model = OnsetsAndVelocities(
        in_chans=2,
        in_height=num_mels, out_height=num_piano_keys,
        conv1x1head=CONF.CONV1X1,
        bn_momentum=0,
        leaky_relu_slope=CONF.LEAKY_RELU_SLOPE,
        dropout_drop_p=0).to(CONF.DEVICE)
    
    print("Loading checkpoint...")
    load_model(model, CONF.SNAPSHOT_INPATH, eval_phase=True)
    
    # Create decoders
    print("Creating decoders...")
    decoder = OnsetVelocityNmsDecoder(
        num_piano_keys, nms_pool_ksize=3,
        gauss_conv_stddev=CONF.DECODER_GAUSS_STD,
        gauss_conv_ksize=CONF.DECODER_GAUSS_KSIZE,
        vel_pad_left=1, vel_pad_right=1)
    
    pedal_decoder = PedalDecoder(num_pedals=1, threshold=0.5)
    
    def model_inference(x):
        """
        Convenience wrapper around the DNN to ensure output and input sequences
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
    
    # Test with dummy data
    print("Testing model inference...")
    test_input = torch.randn(1, num_mels, 100).to(CONF.DEVICE)
    
    probs, vels, pedals = model_inference(test_input)
    print(f"Probabilities shape: {probs.shape}")
    print(f"Velocities shape: {vels.shape}")
    print(f"Pedals shape: {pedals.shape}")
    
    # Test pedal decoder
    print("Testing pedal decoder...")
    pedal_events = pedal_decoder(pedals)
    print(f"Pedal events type: {type(pedal_events)}")
    if hasattr(pedal_events, '__len__'):
        print(f"Pedal events length: {len(pedal_events)}")
    
    print("SUCCESS: Pedal evaluation integration is working!")
    return True

if __name__ == "__main__":
    main()