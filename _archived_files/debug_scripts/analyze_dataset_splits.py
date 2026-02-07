#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
Analyze and report train/validation/test splits for MAESTRO and MAPS datasets.
"""

import os
import pandas as pd
from pathlib import Path
from collections import defaultdict

# ==============================================================================
# MAESTRO ANALYSIS
# ==============================================================================
print("=" * 80)
print("MAESTRO v3.0.0 Dataset Split Analysis")
print("=" * 80)

maestro_csv = "datasets/maestro/maestro-v3.0.0/maestro-v3.0.0.csv"
df_maestro = pd.read_csv(maestro_csv)

# Analyze by split
splits_info = df_maestro.groupby('split').agg({
    'audio_filename': 'count',
    'duration': ['sum', 'mean', 'min', 'max']
}).round(2)

print("\nMAESTRO Split Summary:")
print("-" * 80)
for split in ['train', 'validation', 'test']:
    split_data = df_maestro[df_maestro['split'] == split]
    count = len(split_data)
    total_hours = split_data['duration'].sum() / 3600
    avg_hours = split_data['duration'].mean() / 3600
    min_seconds = split_data['duration'].min()
    max_seconds = split_data['duration'].max() / 60  # convert to minutes for readability
    
    print(f"\n{split.upper()}:")
    print(f"  Files:        {count}")
    print(f"  Total:        {total_hours:.1f} hours")
    print(f"  Avg/File:     {avg_hours:.2f} hours")
    print(f"  Min/File:     {min_seconds:.1f} seconds")
    print(f"  Max/File:     {max_seconds:.1f} minutes")

# ==============================================================================
# MAPS ANALYSIS
# ==============================================================================
print("\n" + "=" * 80)
print("MAPS Dataset Split Analysis")
print("=" * 80)

maps_root = "datasets/MAPS"
maps_pianos = sorted([d for d in os.listdir(maps_root) 
                      if os.path.isdir(os.path.join(maps_root, d))])

# MAPS has ISOL (isolated notes), MUS (music), RAND (random), UCHO (chords)
maps_categories = ["ISOL", "MUS", "RAND", "UCHO"]

maps_split_counts = defaultdict(lambda: defaultdict(int))
maps_split_files = defaultdict(list)

for piano in maps_pianos:
    piano_path = os.path.join(maps_root, piano)
    for category in maps_categories:
        category_path = os.path.join(piano_path, category)
        if os.path.isdir(category_path):
            # Count MIDI files
            midi_files = [f for f in os.listdir(category_path) if f.endswith('.mid')]
            maps_split_counts[category][piano] = len(midi_files)
            maps_split_files[category].extend(midi_files)

print("\nMAPS Split Summary by Category:")
print("-" * 80)
for category in maps_categories:
    total_files = sum(maps_split_counts[category].values())
    print(f"\n{category} (Notes/Chords/Music/Random):")
    print(f"  Total files across all pianos: {total_files}")
    for piano in maps_pianos:
        count = maps_split_counts[category].get(piano, 0)
        if count > 0:
            print(f"    {piano:12s}: {count:4d} files")

# ==============================================================================
# RECOMMENDATIONS
# ==============================================================================
print("\n" + "=" * 80)
print("DATASET SPLIT RECOMMENDATIONS")
print("=" * 80)

print("""
MAESTRO Dataset:
  ✓ Already has official train/validation/test splits
  ✓ Train:       962 files (159.2 hours)
  ✓ Validation:  137 files (estimated ~20 hours based on file ratio)
  ✓ Test:        177 files (20 hours)
  
  For your prototype, the training script currently uses:
  - Training split: All 962 train files chunked into small time segments
  - Validation split: Every 5th validation file (reduces from 137 to ~27 files)
  
MAPS Dataset:
  ✓ Has 4 categories: ISOL, MUS, RAND, UCHO
  ✓ 9 different pianos for generalization
  ✓ Suitable for both training and evaluation
  
  Suggested split strategy:
  - Use for augmentation/additional training if needed
  - Good for testing model generalization across pianos
  
Current Training Configuration:
  ✓ Uses MAESTRO v3 training split (official split)
  ✓ Validation reduced by 5x for faster validation
  ✓ Easy to extend with MAPS data for additional training
""")

print("=" * 80)
