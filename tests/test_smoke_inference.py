import torch

from ov_piano.models.ov import OnsetsAndVelocities
from ov_piano.inference import OnsetVelocityNmsDecoder, strided_inference


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
    assert pedals.shape == (1, 1, 63)

    onset_probs = torch.sigmoid(onset_stages[-1])
    decoder = OnsetVelocityNmsDecoder(num_keys=32, nms_pool_ksize=3)
    events = decoder(onset_probs, velocities)

    assert set(["batch_idx", "key", "t_idx", "prob", "vel"]).issubset(events.columns)
    assert len(events) >= 0


def test_strided_inference_default_overlap_is_valid():
    x = torch.randn(1, 4, 8)

    def identity_model(chunk):
        return chunk, chunk + 1

    first, second = strided_inference(identity_model, x, chunk_size=16)

    assert torch.equal(first, x)
    assert torch.equal(second, x + 1)
