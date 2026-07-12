import torch
import pandas as pd

from ov_piano.data.pedal import sustain_pedal_targets_from_values
from ov_piano.inference import PedalDecoder
from ov_piano.eval import (
    EvaluationCheckpointStore,
    evaluation_fingerprint,
    pedal_grid_search,
    threshold_eval_pedals,
)


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


def test_sustain_pedal_targets_include_first_aligned_transition():
    values = torch.tensor([[[0.0, 8.0, 8.0, 0.0, 10.0]]])

    targets = sustain_pedal_targets_from_values(values, threshold=7, align_to_model_diff=True)

    assert targets.state.tolist() == [[[1.0, 1.0, 0.0, 1.0]]]
    assert targets.onset.tolist() == [[[1.0, 0.0, 0.0, 1.0]]]
    assert targets.offset.tolist() == [[[0.0, 0.0, 1.0, 0.0]]]


def test_pedal_decoder_uses_explicit_transition_heads_directly():
    probs = torch.zeros(1, 3, 6)
    probs[:, 0, :] = 0.10  # State head intentionally stays below threshold.
    probs[:, 1, 2] = 0.95  # Sustain pedal down/onset head.
    probs[:, 2, 4] = 0.95  # Sustain pedal up/offset head.

    decoder = PedalDecoder(num_pedals=1, threshold=0.5, min_hold_steps=1, smoothing_window=1)
    events_df, out_probs, states = decoder(probs)

    assert out_probs.shape == probs.shape
    assert events_df["event_type"].tolist() == ["onset", "offset"]
    assert events_df["t_idx"].tolist() == [2, 4]
    assert states[0, 0].tolist() == [0.0, 0.0, 1.0, 1.0, 0.0, 0.0]


def test_explicit_pedal_decoder_does_not_fallback_to_state_crossings():
    probs = torch.zeros(1, 3, 6)
    probs[:, 0, 2:5] = 0.95  # State-only activity without event-head peaks.

    decoder = PedalDecoder(num_pedals=1, threshold=0.5, min_hold_steps=1, smoothing_window=1)
    events_df, _, states = decoder(probs)

    assert events_df.empty
    assert states[0, 0].tolist() == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


def test_threshold_eval_pedals_accepts_explicit_transition_heads():
    probs = torch.zeros(1, 3, 5)
    probs[:, 1, 1] = 1.0
    probs[:, 2, 3] = 1.0
    gt_events = pd.DataFrame({
        "pedal_idx": [0, 0],
        "onset": [1.0, 3.0],
        "event_type": ["onset", "offset"],
    })

    results = threshold_eval_pedals(
        gt_events, probs, secs_per_frame=1.0, thresh=0.5, tol_secs=0.01)

    assert results["sustain"]["f1"] == 1.0
    assert results["sustain"]["onset_f1"] == 1.0
    assert results["sustain"]["offset_f1"] == 1.0


def test_pedal_grid_search_tunes_threshold_and_shift():
    probs = torch.zeros(1, 3, 6)
    probs[:, 1, 1] = 0.8
    probs[:, 2, 3] = 0.8
    gt_events = pd.DataFrame({
        "pedal_idx": [0, 0],
        "onset": [2.0, 4.0],
        "event_type": ["onset", "offset"],
    })

    summary, best_params, best_metrics = pedal_grid_search(
        [(gt_events, probs)],
        secs_per_frame=1.0,
        thresholds=(0.5, 0.9),
        hysteresis_values=(0.02,),
        smoothing_windows=(1,),
        min_hold_steps_values=(1,),
        shifts=(0.0, 1.0),
        tol_secs=0.01,
    )

    assert len(summary) == 4
    assert best_params == (0.5, 0.02, 1, 1, 1.0)
    assert best_metrics.tolist() == [1.0, 1.0, 1.0]


def test_evaluation_checkpoint_store_round_trip_and_fingerprint_guard(tmp_path):
    fingerprint = evaluation_fingerprint({"stage": "unit", "thresholds": [0.5]})
    path = tmp_path / "eval_checkpoint.pt"

    store = EvaluationCheckpointStore(path, fingerprint, "unit")
    store.upsert(
        "piece-a",
        {
            "status": "ok",
            "frame": pd.DataFrame({"x": [1, 2]}),
            "tensor": torch.tensor([0.25, 0.75]),
        },
    )

    reloaded = EvaluationCheckpointStore(path, fingerprint, "unit")
    entry = reloaded.get("piece-a")

    assert entry["status"] == "ok"
    assert entry["frame"]["x"].tolist() == [1, 2]
    assert torch.equal(entry["tensor"], torch.tensor([0.25, 0.75]))

    stale = EvaluationCheckpointStore(path, "different-fingerprint", "unit")
    assert stale.get("piece-a") is None


def test_pedal_grid_search_uses_checkpoint_store(tmp_path):
    probs = torch.zeros(1, 3, 6)
    probs[:, 1, 1] = 0.8
    probs[:, 2, 3] = 0.8
    gt_events = pd.DataFrame({
        "pedal_idx": [0, 0],
        "onset": [1.0, 3.0],
        "event_type": ["onset", "offset"],
    })
    fingerprint = evaluation_fingerprint({"stage": "pedal-grid", "files": ["piece-a"]})
    store = EvaluationCheckpointStore(tmp_path / "pedal_grid.pt", fingerprint, "pedal-grid")

    summary, best_params, best_metrics = pedal_grid_search(
        [(gt_events, probs)],
        secs_per_frame=1.0,
        thresholds=(0.5,),
        hysteresis_values=(0.02,),
        smoothing_windows=(1,),
        min_hold_steps_values=(1,),
        shifts=(0.0,),
        tol_secs=0.01,
        checkpoint_store=store,
    )

    assert best_params == (0.5, 0.02, 1, 1, 0.0)
    assert best_metrics.tolist() == [1.0, 1.0, 1.0]

    logs = []
    bad_probs = torch.zeros(1, 3, 6)
    summary2, best_params2, best_metrics2 = pedal_grid_search(
        [(gt_events, bad_probs)],
        secs_per_frame=1.0,
        thresholds=(0.5,),
        hysteresis_values=(0.02,),
        smoothing_windows=(1,),
        min_hold_steps_values=(1,),
        shifts=(0.0,),
        tol_secs=0.01,
        logger=logs.append,
        checkpoint_store=store,
    )

    assert set(summary2) == set(summary)
    for key in summary:
        assert torch.allclose(
            torch.tensor(summary2[key], dtype=torch.float32),
            torch.tensor(summary[key], dtype=torch.float32),
        )
    assert best_params2 == best_params
    assert best_metrics2.tolist() == best_metrics.tolist()
    assert any("checkpoint hit" in message for message in logs)
