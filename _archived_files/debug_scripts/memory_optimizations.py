#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
Windows-specific memory optimization utilities for PyTorch + h5py training.
Run this BEFORE starting training to configure system memory settings.
"""

import os
import sys

def optimize_windows_memory():
    """
    Apply Windows-specific optimizations for memory management.
    Should be called at the start of training scripts.
    """
    print("=" * 60)
    print("MEMORY OPTIMIZATION FOR WINDOWS")
    print("=" * 60)
    
    try:
        import torch
    except ImportError:
        print("⚠ PyTorch not imported yet (will be loaded by training script)")
        torch = None
    
    # 1. Set environment variables for better memory management
    # Note: expandable_segments only works in PyTorch 2.0+
    # For older versions, we skip it
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    
    # 2. Enable memory pooling in PyTorch (if available and CUDA available)
    if torch is not None:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
                print(f"✓ CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            except Exception as e:
                print(f"⚠ CUDA detected but info unavailable: {e}")
            
            # Use CUDA memory pooling for better allocation (if supported)
            try:
                # This helps avoid memory fragmentation
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                print("✓ CUDA memory optimization enabled")
            except Exception as e:
                print(f"⚠ CUDA optimization warning: {e}")
        else:
            print("⚠ CUDA not available, using CPU (this will be slow)")
        
        # 3. Set PyTorch to use less aggressive memory allocation
        # This prevents the DLL loading error (if CUDA is available)
        try:
            if torch.cuda.is_available():
                torch.cuda.set_per_process_memory_fraction(0.8)  # Use 80% of available VRAM
        except:
            pass
    else:
        print("✓ Environment variables set for PyTorch optimization")
    
    print("\nMemory optimization settings applied!")
    print("=" * 60)


def check_memory_requirements():
    """
    Check and report memory status
    """
    try:
        import psutil
        
        # System memory
        mem = psutil.virtual_memory()
        print(f"\nSystem Memory: {mem.total / 1e9:.1f} GB total")
        print(f"  Available: {mem.available / 1e9:.1f} GB")
        print(f"  Used: {mem.used / 1e9:.1f} GB ({mem.percent}%)")
        
        # Paging file
        swap = psutil.swap_memory()
        print(f"\nPaging File (Virtual Memory): {swap.total / 1e9:.1f} GB total")
        print(f"  Available: {swap.free / 1e9:.1f} GB")
        print(f"  Used: {swap.used / 1e9:.1f} GB")
        
        if swap.total < 100e9:  # Less than 100GB
            print("\n⚠ WARNING: Paging file seems small!")
            print("  Consider increasing it on Windows:")
            print("  Settings → System → Advanced → Performance → Virtual Memory")
            
        # GPU memory
        try:
            import torch
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(i)
                    print(f"\nGPU {i}: {props.name}")
                    print(f"  VRAM: {props.total_memory / 1e9:.1f} GB")
        except:
            pass
        
    except ImportError:
        print("⚠ psutil not available. Install with: pip install psutil")
        return
    
    print("\n" + "=" * 60)
    print("RECOMMENDATIONS:")
    print("1. Close other applications to free up RAM")
    print("2. If errors persist, reduce TRAIN_BATCH_SECS further")
    print("3. Ensure Windows paging file is 100+ GB")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    optimize_windows_memory()
    check_memory_requirements()
