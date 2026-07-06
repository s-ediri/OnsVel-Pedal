import torch

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