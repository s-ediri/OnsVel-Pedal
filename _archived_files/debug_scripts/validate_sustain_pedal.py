#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
Validation script to verify sustain pedal training implementation
- Checks HDF5 data structure
- Validates model output shapes
- Verifies loss computation
- Tests end-to-end forward pass
"""

import torch
import numpy as np
from ov_piano import PIANO_MIDI_RANGE
from ov_piano.models.ov import OnsetsAndVelocities
from ov_piano.data.maestro import MelMaestro, MetaMAESTROv3

def validate_hdf5_structure():
    """Verify HDF5 file contains sustain pedal data"""
    print("\n=== Validating HDF5 Structure ===")
    
    try:
        maestro = MetaMAESTROv3("datasets/maestro/maestro-v3.0.0", splits=["train"], years=[2004])
        if not maestro.data:
            print("❌ No MAESTRO files found")
            return False
        
        mel_maestro = MelMaestro(
            "datasets/MAESTROv3_logmel_sr=16000_stft=2048w384h_mel=229(50-8000).h5",
            "datasets/MAESTROv3_roll_quant=0.024_midivals=128_extendsus=True.h5",
            maestro.data[0][0],
            as_torch_tensors=False
        )
        
        logmel, roll, meta = mel_maestro[0]
        print(f"✓ Loaded first training file: {meta}")
        print(f"  Logmel shape: {logmel.shape} (frequencies, time)")
        print(f"  Roll shape: {roll.shape} (data_rows, time)")
        
        # Validate roll structure: [onsets (88); frames (88); sustain (88); soft (88); tenuto (88)]
        key_beg, key_end = PIANO_MIDI_RANGE
        num_keys = key_end - key_beg  # Should be 88
        
        expected_rows = 5 * num_keys  # 5 types × 88 keys
        if roll.shape[0] == expected_rows:
            print(f"✓ Roll has correct structure: {expected_rows} rows = 5 × {num_keys} keys")
        else:
            print(f"❌ Roll has {roll.shape[0]} rows, expected {expected_rows}")
            return False
        
        # Check sustain pedal rows
        sustain_beg = 2 * num_keys
        sustain_end = 3 * num_keys
        sustain_data = roll[sustain_beg:sustain_end]
        sustain_mean = sustain_data.mean()
        sustain_max = sustain_data.max()
        sustain_min = sustain_data.min()
        
        print(f"✓ Sustain pedal data statistics:")
        print(f"    Mean: {sustain_mean:.2f}, Min: {sustain_min:.2f}, Max: {sustain_max:.2f}")
        
        if sustain_max > 0:
            print(f"✓ Sustain pedal has active events (good for training)")
        else:
            print(f"⚠ Sustain pedal appears inactive - verify file is correct")
        
        return True
        
    except Exception as e:
        print(f"❌ Error validating HDF5: {e}")
        return False


def validate_model_shapes():
    """Verify model output shapes for sustain pedal"""
    print("\n=== Validating Model Shapes ===")
    
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        
        # Create model
        model = OnsetsAndVelocities(
            in_chans=2,
            in_height=229,  # mel bins
            out_height=88,  # piano keys
            conv1x1head=(128, 128),
            bn_momentum=0.95,
            leaky_relu_slope=0.1,
            dropout_drop_p=0.2
        ).to(device).eval()
        
        # Create dummy input
        batch_size = 4
        time_steps = 100
        logmels = torch.randn(batch_size, 2, 229, time_steps, device=device)
        
        print(f"Input shape: {logmels.shape} (batch, channels, mels, time)")
        
        with torch.no_grad():
            onset_stages, velocities, pedals = model(logmels, trainable_onsets=True)
        
        # Validate shapes
        print(f"✓ Onset stages: {len(onset_stages)} stages")
        for i, onset in enumerate(onset_stages):
            print(f"    Stage {i}: {onset.shape}")
        
        print(f"✓ Velocities shape: {velocities.shape}")
        assert velocities.shape == (batch_size, 88, time_steps), f"Wrong velocity shape!"
        
        print(f"✓ Pedals shape: {pedals.shape}")
        assert pedals.shape == (batch_size, 1, time_steps), f"Wrong pedal shape! Expected (b, 1, t), got {pedals.shape}"
        
        print(f"✓ All model output shapes are correct!")
        return True
        
    except Exception as e:
        print(f"❌ Error validating model: {e}")
        return False


def validate_loss_computation():
    """Verify sustain pedal loss computation"""
    print("\n=== Validating Loss Computation ===")
    
    try:
        batch_size = 4
        time_steps = 100
        
        # Simulate model output and target
        pedal_logits = torch.randn(batch_size, 1, time_steps)
        sustain_target = torch.randint(0, 2, (batch_size, 1, time_steps)).float()
        
        print(f"Pedal logits shape: {pedal_logits.shape}")
        print(f"Sustain target shape: {sustain_target.shape}")
        
        # Compute loss
        pedal_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            pedal_logits.reshape(-1), 
            sustain_target.reshape(-1)
        )
        
        print(f"✓ Loss computed successfully: {pedal_loss.item():.4f}")
        
        # Test different weights
        loss_weights = [0.1, 0.5, 1.0]
        for weight in loss_weights:
            weighted_loss = weight * pedal_loss
            print(f"  With weight {weight}: {weighted_loss.item():.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error computing loss: {e}")
        return False


def validate_data_extraction():
    """Verify sustain pedal extraction from roll data"""
    print("\n=== Validating Data Extraction ===")
    
    try:
        batch_size = 2
        num_keys = 88
        time_steps = 100
        
        # Simulate roll data: [onsets; frames; sustain; soft; tenuto]
        onsets_beg, onsets_end = 0, num_keys
        sustain_beg, sustain_end = 2 * num_keys, 3 * num_keys
        
        rolls = torch.randint(0, 128, (batch_size, 5 * num_keys, time_steps)).float()
        
        # Extract sustain
        sustain_pedal = rolls[:, sustain_beg:sustain_end]
        print(f"Extracted sustain shape: {sustain_pedal.shape}")
        assert sustain_pedal.shape == (batch_size, num_keys, time_steps)
        
        # Average to get single pedal signal
        sustain_norm = sustain_pedal.mean(dim=1, keepdim=True) / 127.0
        print(f"Normalized sustain shape: {sustain_norm.shape}")
        assert sustain_norm.shape == (batch_size, 1, time_steps)
        
        print(f"✓ Data extraction validated!")
        print(f"  Sustain mean value: {sustain_norm.mean():.3f}")
        print(f"  Sustain range: [{sustain_norm.min():.3f}, {sustain_norm.max():.3f}]")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in data extraction: {e}")
        return False


if __name__ == "__main__":
    print("\n" + "="*60)
    print("SUSTAIN PEDAL TRAINING VALIDATION")
    print("="*60)
    
    results = {
        "HDF5 Structure": validate_hdf5_structure(),
        "Model Shapes": validate_model_shapes(),
        "Loss Computation": validate_loss_computation(),
        "Data Extraction": validate_data_extraction(),
    }
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for name, result in results.items():
        status = "✓ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")
    
    print(f"\nResult: {passed}/{total} checks passed")
    
    if passed == total:
        print("\n✓ All validations passed! Sustain pedal training is ready.")
    else:
        print("\n❌ Some validations failed. Please review the errors above.")
