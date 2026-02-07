#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
Debug script to identify which part is causing memory issues
"""

import os
import gc
import psutil
import torch

def print_memory():
    """Print current memory usage"""
    process = psutil.Process()
    mem = process.memory_info()
    print(f"  RAM: {mem.rss / 1e9:.2f} GB | VRAM: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

print("=" * 60)
print("MEMORY DEBUG - Testing data loading")
print("=" * 60)

print("\n1. Initial state")
print_memory()

print("\n2. Loading PyTorch & CUDA...")
print_memory()

print("\n3. Importing ov_piano modules...")
from ov_piano import PIANO_MIDI_RANGE, HDF5PathManager
from ov_piano.data.maestro import MetaMAESTROv3, MelMaestroChunks
print_memory()

print("\n4. Loading MAESTRO metadata...")
maestro_path = os.path.join("datasets", "maestro", "maestro-v3.0.0")
try:
    metamaestro = MetaMAESTROv3(maestro_path, splits=["train"], years={2018})
    print(f"   Found {len(metamaestro.data)} files")
    print_memory()
except Exception as e:
    print(f"   ERROR: {e}")
    exit(1)

print("\n5. Opening HDF5 files...")
mel_path = os.path.join("datasets", "MAESTROv3_logmel_sr=16000_stft=2048w384h_mel=229(50-8000).h5")
roll_path = os.path.join("datasets", "MAESTROv3_roll_quant=0.024_midivals=128_extendsus=True.h5")

try:
    dataset = MelMaestroChunks(
        mel_path, roll_path,
        *(x[0] for x in metamaestro.data),
        with_oob=True, logmel_oob_pad_val="min",
        as_torch_tensors=False)
    print(f"   Created dataset with {len(dataset)} chunks")
    print_memory()
except Exception as e:
    print(f"   ERROR: {e}")
    exit(1)

print("\n6. Loading first batch...")
try:
    logmel, roll, meta = dataset[0]
    print(f"   Logmel shape: {logmel.shape}, Roll shape: {roll.shape}")
    print_memory()
    del logmel, roll, meta
    gc.collect()
    torch.cuda.empty_cache()
except Exception as e:
    print(f"   ERROR: {e}")
    exit(1)

print("\n7. Loading 10 consecutive batches...")
try:
    for i in range(10):
        logmel, roll, meta = dataset[i]
        del logmel, roll, meta
        if (i + 1) % 5 == 0:
            gc.collect()
            torch.cuda.empty_cache()
            print(f"   Batch {i + 1}/10: ", end="")
            print_memory()
except Exception as e:
    print(f"   ERROR at batch {i}: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n✓ All data loading tests passed!")
print("=" * 60)
