import torch

from ov_piano.models.ov import OnsetsAndVelocities
from ov_piano.utils import format_load_model_warnings, load_model


def test_load_model_strict_false_reports_checkpoint_incompatibilities(tmp_path):
    model = torch.nn.Sequential(torch.nn.Linear(2, 3))
    checkpoint_path = tmp_path / "partial_checkpoint.torch"
    torch.save(
        {
            "0.weight": torch.ones(4, 2),  # present in model, but wrong shape
            "extra.weight": torch.ones(1),  # not present in model
        },
        checkpoint_path,
    )

    report = load_model(model, checkpoint_path, strict=False, to_cpu=True)

    assert report["path"] == checkpoint_path
    assert report["strict"] is False
    assert report["missing_keys"] == ["0.bias"]
    assert report["unexpected_keys"] == ["extra.weight"]
    assert report["shape_mismatched_keys"] == [
        {
            "key": "0.weight",
            "checkpoint_shape": (4, 2),
            "model_shape": (3, 2),
        }
    ]

    warnings = format_load_model_warnings(report)
    assert len(warnings) == 3
    assert "missing 1 model key(s): 0.bias" in warnings[0]
    assert "unexpected key(s): extra.weight" in warnings[1]
    assert "0.weight checkpoint_shape=(4, 2) model_shape=(3, 2)" in warnings[2]


def test_load_model_strict_false_reports_no_warnings_for_matching_checkpoint(tmp_path):
    model = torch.nn.Sequential(torch.nn.Linear(2, 3))
    checkpoint_path = tmp_path / "matching_checkpoint.torch"
    torch.save(model.state_dict(), checkpoint_path)

    report = load_model(model, checkpoint_path, strict=False, to_cpu=True)

    assert report["missing_keys"] == []
    assert report["unexpected_keys"] == []
    assert report["shape_mismatched_keys"] == []
    assert format_load_model_warnings(report) == []


def test_format_load_model_warnings_can_ignore_expected_missing_pedal_head():
    report = {
        "path": "note_only_checkpoint.torch",
        "strict": False,
        "missing_keys": [
            "pedal_stage.0.weight",
            "pedal_stage.1.weight",
            "pedal_stage.1.bias",
        ],
        "unexpected_keys": [],
        "shape_mismatched_keys": [],
    }

    assert format_load_model_warnings(
        report,
        ignored_missing_key_prefixes=("pedal_stage.",),
    ) == []


def test_format_load_model_warnings_still_reports_unignored_missing_keys():
    report = {
        "path": "partial_checkpoint.torch",
        "strict": False,
        "missing_keys": ["pedal_stage.0.weight", "0.bias"],
        "unexpected_keys": [],
        "shape_mismatched_keys": [],
    }

    warnings = format_load_model_warnings(
        report,
        ignored_missing_key_prefixes=("pedal_stage.",),
    )

    assert len(warnings) == 1
    assert "missing 1 model key(s): 0.bias" in warnings[0]


def test_load_model_migrates_legacy_monolithic_pedal_output_head(tmp_path):
    model = OnsetsAndVelocities(in_chans=2, in_height=16, out_height=8, init_fn=None)
    checkpoint = model.state_dict()
    # The pre-TCN pedal branch appended the monolithic 3-channel output head at
    # index 23. Migration should scan for a compatible legacy head rather than
    # assuming the current pedal_stage length.
    legacy_head_idx = 23
    legacy_weight = torch.randn(3, model.pedal_state_head.in_channels, 1, 1)
    legacy_bias = torch.randn(3)

    checkpoint.pop("pedal_state_head.weight")
    checkpoint.pop("pedal_state_head.bias")
    checkpoint.pop("pedal_onset_head.weight")
    checkpoint.pop("pedal_onset_head.bias")
    checkpoint.pop("pedal_offset_head.weight")
    checkpoint.pop("pedal_offset_head.bias")
    checkpoint[f"pedal_stage.{legacy_head_idx}.weight"] = legacy_weight
    checkpoint[f"pedal_stage.{legacy_head_idx}.bias"] = legacy_bias

    checkpoint_path = tmp_path / "legacy_pedal_head.torch"
    torch.save(checkpoint, checkpoint_path)

    report = load_model(model, checkpoint_path, strict=False, to_cpu=True)

    assert report["missing_keys"] == []
    assert report["unexpected_keys"] == []
    assert torch.equal(model.pedal_state_head.weight, legacy_weight[0:1])
    assert torch.equal(model.pedal_onset_head.weight, legacy_weight[1:2])
    assert torch.equal(model.pedal_offset_head.weight, legacy_weight[2:3])
    assert torch.equal(model.pedal_state_head.bias, legacy_bias[0:1])
    assert torch.equal(model.pedal_onset_head.bias, legacy_bias[1:2])
    assert torch.equal(model.pedal_offset_head.bias, legacy_bias[2:3])