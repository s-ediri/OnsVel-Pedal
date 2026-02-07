#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
Demo script showing how to use the new PedalDecoder for pedal event inference.
This can be called after training to transcribe pedal events from new audio.
"""

import torch
from ov_piano.inference import PedalDecoder


def demo_pedal_decoding():
    """
    Example: Convert raw pedal logits to events
    """
    # Simulated output from model: (batch=1, pedals=3, time=1000)
    batch_size = 1
    num_pedals = 3
    time_steps = 1000
    
    # Random logits (what the model outputs before softmax)
    pedal_logits = torch.randn(batch_size, num_pedals, time_steps)
    
    # Create decoder
    decoder = PedalDecoder(num_pedals=3, threshold=0.5)
    
    # Decode events
    events_df, probs, states = decoder(pedal_logits)
    
    print("Pedal Events Detected:")
    print(events_df.head(20))
    
    print("\nPedal Event Summary:")
    print(f"Total events: {len(events_df)}")
    for pedal_idx, pedal_name in enumerate(["Sustain", "Soft", "Tenuto"]):
        pedal_events = events_df[events_df["pedal_idx"] == pedal_idx]
        print(f"\n{pedal_name} pedal:")
        print(f"  Onsets: {len(pedal_events[pedal_events['event_type'] == 'onset'])}")
        print(f"  Offsets: {len(pedal_events[pedal_events['event_type'] == 'offset'])}")


def demo_pedal_evaluation():
    """
    Example: Evaluate pedal detection accuracy
    """
    import pandas as pd
    from ov_piano.eval import eval_pedal_events
    
    # Simulated ground truth: pedal press at 1.0s (onset), release at 2.0s (offset)
    gt_onsets = [1.0, 2.0]
    gt_states = [1.0, 0.0]  # 1=onset, 0=offset
    
    # Simulated prediction: slightly off in time
    pred_onsets = [1.05, 2.02]
    pred_states = [1.0, 0.0]
    
    precision, recall, f1 = eval_pedal_events(
        gt_onsets, gt_states,
        pred_onsets, pred_states,
        tol_secs=0.1  # 100ms tolerance
    )
    
    print("\nPedal Evaluation Example:")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1 Score:  {f1:.3f}")


if __name__ == "__main__":
    print("=" * 60)
    print("Pedal Decoding Demo")
    print("=" * 60)
    demo_pedal_decoding()
    
    print("\n" + "=" * 60)
    print("Pedal Evaluation Demo")
    print("=" * 60)
    demo_pedal_evaluation()
