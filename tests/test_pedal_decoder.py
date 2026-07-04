import torch
import pandas as pd

from ov_piano.inference import PedalDecoder
from ov_piano.eval import threshold_eval_pedals


def test_pedal_decoder_hysteresis_and_hold_time():
    probs = torch.tensor([
        [[0.10, 0.60, 0.55, 0.40, 0.35, 0.20]],
    ], dtype=torch.float32)

    decoder = PedalDecoder(num_pedals=1, threshold=0.5, hysteresis=0.1, min_hold_steps=2, smoothing_window=1)
    events_df, out_probs, states = decoder(probs)

    assert out_probs.shape == probs.shape
    assert states.shape == probs.shape
    assert len(events_df) == 2
    assert events_df["event_type"].tolist() == ["onset", "offset"]
    assert events_df["t_idx"].tolist() == [1, 4]


def test_threshold_eval_pedals_accepts_tensor_predictions():
    probs = torch.tensor([
        [[0.10, 0.70, 0.75, 0.20, 0.10]],
    ], dtype=torch.float32)
    gt_events = pd.DataFrame({
        "pedal_idx": [0, 0],
        "onset": [1.0, 3.0],
        "event_type": ["onset", "offset"],
    })

    results = threshold_eval_pedals(
        gt_events, probs, secs_per_frame=1.0, thresh=0.5, tol_secs=0.01)

    assert "sustain" in results
    assert results["sustain"]["f1"] == 1.0
    assert results["macro_avg"]["f1"] == 1.0
