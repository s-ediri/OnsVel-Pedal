#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
Checkpoint compatibility verification script.
Ensures that saved model checkpoints properly handle pedal weights.
"""

import torch
import sys
from ov_piano.models.ov import OnsetsAndVelocities


def verify_checkpoint_structure(checkpoint_path=None):
    """
    Verify that a checkpoint (or new model) contains pedal weights.
    
    :param checkpoint_path: Path to .pt checkpoint file. If None, creates new model.
    :returns: Dictionary with verification results
    """
    print("=" * 60)
    print("Checkpoint Verification")
    print("=" * 60)
    
    # Load or create model
    if checkpoint_path is not None:
        print(f"\nLoading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model = OnsetsAndVelocities(
            in_chans=2, in_height=229, out_height=88,
            conv1x1head=(128, 128), bn_momentum=0.95,
            leaky_relu_slope=0.1, dropout_drop_p=0.15)
        model.load_state_dict(checkpoint)
        print("✓ Checkpoint loaded successfully")
    else:
        print("\nCreating fresh model...")
        model = OnsetsAndVelocities(
            in_chans=2, in_height=229, out_height=88,
            conv1x1head=(128, 128), bn_momentum=0.95,
            leaky_relu_slope=0.1, dropout_drop_p=0.15)
        print("✓ Model created")
    
    model.eval()
    
    # Check for pedal_stage
    print("\n" + "-" * 60)
    print("Model Architecture Check")
    print("-" * 60)
    
    has_pedal_stage = hasattr(model, 'pedal_stage')
    print(f"✓ pedal_stage exists: {has_pedal_stage}")
    
    if has_pedal_stage:
        print(f"✓ pedal_stage type: {type(model.pedal_stage)}")
    
    # Check state dict
    print("\n" + "-" * 60)
    print("State Dictionary Check")
    print("-" * 60)
    
    state_dict = model.state_dict()
    pedal_keys = [k for k in state_dict.keys() if 'pedal' in k.lower()]
    
    print(f"✓ Total parameters: {len(state_dict)}")
    print(f"✓ Pedal-related parameters: {len(pedal_keys)}")
    
    if pedal_keys:
        print("\nPedal parameters found:")
        for key in pedal_keys:
            print(f"  - {key}: {state_dict[key].shape}")
    
    # Test forward pass
    print("\n" + "-" * 60)
    print("Forward Pass Test")
    print("-" * 60)
    
    try:
        batch_size = 1
        num_mels = 229
        time_steps = 100
        
        logmels = torch.randn(batch_size, num_mels, time_steps)
        
        with torch.no_grad():
            outputs = model(logmels, trainable_onsets=True)
        
        print(f"✓ Forward pass successful")
        print(f"  Input shape: {logmels.shape}")
        print(f"  Output count: {len(outputs)}")
        
        if len(outputs) >= 3:
            x_stages, velocities, pedals = outputs[0], outputs[1], outputs[2]
            print(f"  - Onset stages: {x_stages.shape if isinstance(x_stages, torch.Tensor) else 'list'}")
            print(f"  - Velocities: {velocities.shape}")
            print(f"  - Pedals: {pedals.shape}")
            
            expected_pedal_shape = (batch_size, 3, time_steps)
            if pedals.shape == expected_pedal_shape:
                print(f"✓ Pedal output shape is correct: {expected_pedal_shape}")
            else:
                print(f"✗ Pedal output shape mismatch!")
                print(f"  Expected: {expected_pedal_shape}")
                print(f"  Got: {pedals.shape}")
                return False
        else:
            print(f"✗ Model returned {len(outputs)} outputs, expected ≥3")
            return False
            
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Summary
    print("\n" + "=" * 60)
    print("Verification Summary")
    print("=" * 60)
    
    results = {
        "has_pedal_stage": has_pedal_stage,
        "pedal_parameters_found": len(pedal_keys) > 0,
        "forward_pass_ok": True,
        "pedal_output_shape_ok": pedals.shape == (batch_size, 3, time_steps)
    }
    
    all_ok = all(results.values())
    
    for key, value in results.items():
        status = "✓" if value else "✗"
        print(f"{status} {key}: {value}")
    
    if all_ok:
        print("\n✓✓✓ ALL CHECKS PASSED ✓✓✓")
        print("Model is ready for training/inference with pedals!")
    else:
        print("\n✗✗✗ SOME CHECKS FAILED ✗✗✗")
        print("Model may have compatibility issues.")
    
    return all_ok


def compare_checkpoints(old_checkpoint, new_checkpoint):
    """
    Compare old checkpoint (pre-pedal) with new one (post-pedal).
    Useful for verifying backward compatibility.
    """
    print("\n" + "=" * 60)
    print("Checkpoint Comparison")
    print("=" * 60)
    
    old_state = torch.load(old_checkpoint, map_location='cpu')
    new_state = torch.load(new_checkpoint, map_location='cpu')
    
    old_keys = set(old_state.keys())
    new_keys = set(new_state.keys())
    
    print(f"\nOld checkpoint parameters: {len(old_keys)}")
    print(f"New checkpoint parameters: {len(new_keys)}")
    
    new_params = new_keys - old_keys
    removed_params = old_keys - new_keys
    common_params = old_keys & new_keys
    
    print(f"\nNew parameters (added): {len(new_params)}")
    if new_params:
        for param in sorted(new_params):
            if 'pedal' in param.lower():
                print(f"  ✓ {param}")
    
    print(f"\nRemoved parameters: {len(removed_params)}")
    if removed_params:
        for param in sorted(removed_params):
            print(f"  - {param}")
    
    print(f"\nCommon parameters: {len(common_params)}")
    
    # Check if common parameters are unchanged
    changed = 0
    for param in common_params:
        if not torch.allclose(old_state[param], new_state[param]):
            changed += 1
    
    print(f"Changed common parameters: {changed}")
    
    return new_params, removed_params, common_params


if __name__ == "__main__":
    if len(sys.argv) > 1:
        checkpoint_path = sys.argv[1]
        verify_checkpoint_structure(checkpoint_path)
    else:
        print("Creating new model to verify pedal support...\n")
        verify_checkpoint_structure(checkpoint_path=None)
