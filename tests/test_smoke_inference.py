import torch
import pytest

from ov_piano.models.ov import OnsetsAndVelocities, ResidualTemporalConvBlock
from ov_piano.inference import (
    OnsetVelocityNmsDecoder,
    model_outputs_to_probabilities,
    strided_inference,
)


def test_model_forward_and_decoder_smoke():
    torch.manual_seed(0)
    model = OnsetsAndVelocities(in_chans=2, in_height=32, out_height=32, init_fn=None)
    model.eval()

    x = torch.randn(1, 32, 64)
    with torch.no_grad():
        onset_stages, velocities, pedals = model(x)

    assert len(onset_stages) == 3
    assert onset_stages[-1].shape == (1, 32, 63)
    assert velocities.shape == (1, 32, 63)
    assert pedals.shape == (1, 3, 63)

    onset_probs = torch.sigmoid(onset_stages[-1])
    decoder = OnsetVelocityNmsDecoder(num_keys=32, nms_pool_ksize=3)
    events = decoder(onset_probs, velocities)

    assert set(["batch_idx", "key", "t_idx", "prob", "vel"]).issubset(events.columns)
    assert len(events) >= 0


def test_pedal_head_uses_residual_dilated_temporal_context():
    model = OnsetsAndVelocities(in_chans=2, in_height=16, out_height=8, init_fn=None)

    temporal_blocks = [
        module for module in model.pedal_stage
        if isinstance(module, ResidualTemporalConvBlock)
    ]

    assert [block.dilation for block in temporal_blocks] == [1, 2, 4, 8, 16]
    assert all(block.kernel_width == 5 for block in temporal_blocks)

    features = torch.randn(2, 17, 8, 31)
    with torch.no_grad():
        pedal_logits = model.forward_pedals(features)

    assert pedal_logits.shape == (2, 3, 31)


def test_strided_inference_default_overlap_is_valid():
    x = torch.randn(1, 4, 8)

    def identity_model(chunk):
        return chunk, chunk + 1

    first, second = strided_inference(identity_model, x, chunk_size=16)

    assert torch.equal(first, x)
    assert torch.equal(second, x + 1)


def test_strided_inference_handles_uneven_final_chunk():
    x = torch.arange(11, dtype=torch.float32).view(1, 1, 11)

    def identity_model(chunk):
        return chunk, chunk + 10

    first, second = strided_inference(identity_model, x, chunk_size=4, chunk_overlap=0)

    assert torch.equal(first, x)
    assert torch.equal(second, x + 10)


def test_strided_inference_trims_overlap_boundaries_once():
    x = torch.arange(12, dtype=torch.float32).view(1, 1, 12)

    def identity_model(chunk):
        return chunk, -chunk

    first, second = strided_inference(identity_model, x, chunk_size=6, chunk_overlap=2)

    assert torch.equal(first, x)
    assert torch.equal(second, -x)


def test_strided_inference_preserves_length_with_uneven_final_overlap():
    x = torch.arange(17, dtype=torch.float32).view(1, 1, 17)

    def identity_model(chunk):
        return chunk, chunk + 100

    first, second = strided_inference(identity_model, x, chunk_size=6, chunk_overlap=2)

    assert torch.equal(first, x)
    assert torch.equal(second, x + 100)


@pytest.mark.parametrize("bad_outputs", [None, (), []])
def test_strided_inference_rejects_empty_or_invalid_model_outputs(bad_outputs):
    x = torch.randn(1, 2, 5)

    def bad_model(chunk):
        return bad_outputs

    with pytest.raises(AssertionError):
        strided_inference(bad_model, x, chunk_size=4, chunk_overlap=0)


def test_strided_inference_rejects_single_tensor_return():
    x = torch.randn(1, 2, 5)

    def single_tensor_model(chunk):
        return chunk

    with pytest.raises(AssertionError, match="list or tuple"):
        strided_inference(single_tensor_model, x, chunk_size=4, chunk_overlap=0)


def test_strided_inference_rejects_single_output_tuple():
    x = torch.randn(1, 2, 5)

    def single_output_model(chunk):
        return (chunk,)

    with pytest.raises(AssertionError, match="at least 2 outputs"):
        strided_inference(single_output_model, x, chunk_size=4, chunk_overlap=0)


def test_strided_inference_rejects_mismatched_output_time_dimensions():
    x = torch.randn(1, 2, 5)

    def mismatched_time_model(chunk):
        return chunk[..., :-1], chunk[..., :-1]

    with pytest.raises(AssertionError, match="t_outputs"):
        strided_inference(mismatched_time_model, x, chunk_size=4, chunk_overlap=0)


def test_model_outputs_to_probabilities_pads_with_zero_not_half():
    onset_logits = [torch.zeros(1, 2, 3)]
    velocity_logits = torch.zeros(1, 2, 3)
    pedal_logits = torch.zeros(1, 3, 3)

    onset_probs, velocity_probs, pedal_probs = model_outputs_to_probabilities(
        (onset_logits, velocity_logits, pedal_logits), include_pedals=True)

    assert onset_probs.shape == (1, 2, 4)
    assert velocity_probs.shape == (1, 2, 4)
    assert pedal_probs.shape == (1, 3, 4)
    assert torch.equal(onset_probs[..., 0], torch.zeros_like(onset_probs[..., 0]))
    assert torch.equal(velocity_probs[..., 0], torch.zeros_like(velocity_probs[..., 0]))
    assert torch.equal(pedal_probs[..., 0], torch.zeros_like(pedal_probs[..., 0]))
    assert torch.allclose(onset_probs[..., 1:], torch.full((1, 2, 3), 0.5))
