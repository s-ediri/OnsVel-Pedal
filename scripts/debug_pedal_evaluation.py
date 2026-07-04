#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
Debug script to check pedal evaluation issues
"""

import sys
import os
sys.path.append('.')

import torch
import pandas as pd
import numpy as np
from ov_piano.data.maestro import Maestro
from ov_piano.eval import GtLoaderMaestro, sus_states_to_events

def debug_pedal_data():
    """Debug pedal data processing"""
    
    print("=== Debugging Pedal Data ===")
    
    # Load a small sample of data
    dataset = Maestro(
        rootpath="data/MAESTRO",
        split="test",
        secs_per_frame=0.016,
        hopsize=512,
        sample_rate=16000,
        seed=42,
        num_frames=2000,  # Short for debugging
        augment=False,
    )
    
    # Create ground truth loader
    meta_dataset = Maestro(
        rootpath="data/MAESTRO",
        split="test",
        secs_per_frame=0.016,
        hopsize=512,
        sample_rate=16000,
        seed=42,
        num_frames=-1,  # Full length for GT
        augment=False,
        is_meta=True,
    )
    
    gts = GtLoaderMaestro(dataset, meta_dataset)
    
    print(f"Dataset size: {len(dataset)}")
    
    # Check first few files
    for i in range(min(3, len(dataset))):
        md = dataset.metadata[i]
        print(f"\n--- File {i+1}: {md[0]} ---")
        
        try:
            # Get pedal events
            pedal_events = gts.get_sus_pedal_events(md, 0.016)
            print(f"Pedal events shape: {pedal_events.shape}")
            print(f"Pedal events columns: {pedal_events.columns.tolist()}")
            if not pedal_events.empty:
                print(f"Sample pedal events:\n{pedal_events.head(10)}")
                print(f"Onset events: {len(pedal_events[pedal_events['event_type'] == 'onset'])}")
                print(f"Offset events: {len(pedal_events[pedal_events['event_type'] == 'offset'])}")
            else:
                print("No pedal events found")
                
        except Exception as e:
            print(f"Error processing pedal events: {e}")

if __name__ == "__main__":
    debug_pedal_data()