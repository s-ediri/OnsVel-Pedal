#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
Quick script to inspect HDF5 pedal data structure
"""

import h5py
import os

HDF5_ROLL_PATH = "datasets/MAESTROv3_roll_quant=0.024_midivals=128_extendsus=True.h5"

if os.path.exists(HDF5_ROLL_PATH):
    with h5py.File(HDF5_ROLL_PATH, 'r') as f:
        print("=" * 80)
        print(f"HDF5 File: {HDF5_ROLL_PATH}")
        print("=" * 80)
        
        # List all keys in the file
        print("\nTop-level keys:")
        for key in f.keys():
            print(f"  - {key}")
        
        # Check first piece's structure
        if len(f.keys()) > 0:
            first_piece = list(f.keys())[0]
            print(f"\nFirst piece: {first_piece}")
            piece_data = f[first_piece]
            
            print(f"  Shape: {piece_data.shape}")
            print(f"  Dtype: {piece_data.dtype}")
            
            # Try to understand the structure
            total_rows = piece_data.shape[0]
            total_cols = piece_data.shape[1] if len(piece_data.shape) > 1 else 1
            
            print(f"\n  Total rows: {total_rows}")
            print(f"  Total cols: {total_cols}")
            
            # Print sample data
            print(f"\n  Sample (first 10 rows, first 10 cols):")
            if len(piece_data.shape) > 1:
                sample = piece_data[:10, :10]
            else:
                sample = piece_data[:10]
            print(sample)
            
            # Check if rows are structured as: onsets (88) + frames (88) + sustain (88) + soft (88) + tenuto (88)
            print("\n" + "=" * 80)
            print("Expected structure according to code:")
            print("  Rows 0-87:    Onsets")
            print("  Rows 88-175:  Frames")
            print("  Rows 176-263: Sustain pedal")
            print("  Rows 264-351: Soft pedal")
            print("  Rows 352-439: Tenuto pedal")
            print("=" * 80)
            
            if total_rows == 440:
                print("✓ Structure matches! (440 rows = 88*5 pedal types)")
            elif total_rows == 88:
                print("✗ Only 88 rows - missing pedal data!")
            else:
                print(f"? Unexpected: {total_rows} rows (expected 88 or 440)")
            
            # Check sustain pedal specifically
            if total_rows >= 264:
                sustain_beg = 176
                sustain_end = 264
                sustain_data = piece_data[sustain_beg:sustain_end, :10]
                print(f"\nSustain pedal (rows {sustain_beg}-{sustain_end-1}):")
                print(f"  Sample shape: {sustain_data.shape}")
                print(f"  Min: {sustain_data.min()}, Max: {sustain_data.max()}")
                print(f"  Mean per row (first 5 rows):")
                for i in range(min(5, sustain_data.shape[0])):
                    print(f"    Row {sustain_beg + i}: {sustain_data[i].mean():.4f}")
else:
    print(f"File not found: {HDF5_ROLL_PATH}")
