#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
Training launcher script with memory optimizations for Windows.
Run this instead of directly running 1_train_onsets_velocities.py
"""

import sys
import os

# Apply memory optimizations BEFORE importing torch
from memory_optimizations import optimize_windows_memory, check_memory_requirements

print("Checking memory requirements...")
check_memory_requirements()

print("\nApplying memory optimizations...")
optimize_windows_memory()

# Now we can safely import and run the training script
print("Launching training script...")
print("-" * 60)

# Import and run training
if __name__ == "__main__":
    import subprocess
    
    # Run the actual training script
    result = subprocess.run(
        [sys.executable, "1_train_onsets_velocities.py"] + sys.argv[1:],
        cwd=os.path.dirname(__file__)
    )
    
    sys.exit(result.returncode)
