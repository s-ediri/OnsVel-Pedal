import torch

from ov_piano.inference import PedalDecoder


def test_pedal_decoder_hysteresis_and_hold_time():
    probs = torch.tensor([
        [[0.10, 0.60, 0.55, 0.40, 0.35, 0.20]],
    ], dtype=torch.float32)

    decoder = PedalDecoder(num_pedals=1, threshold=0.5, hysteresis=0.1, min_hold_steps=2)
    events_df, out_probs, states = decoder(probs)

    assert out_probs.shape == probs.shape
    assert states.shape == probs.shape
    assert len(events_df) == 2
    assert events_df["event_type"].tolist() == ["onset", "offset"]
    assert events_df["t_idx"].tolist() == [1, 3]
