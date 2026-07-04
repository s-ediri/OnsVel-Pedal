
import os
import sys
import torch
import argparse

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ov_piano import PIANO_MIDI_RANGE
from ov_piano.models.ov import OnsetsAndVelocities
from ov_piano.utils import load_model, torch_load_resample_audio, TorchWavToLogmel
from ov_piano.inference import strided_inference, OnsetVelocityNmsDecoder, PedalDecoder

class AppConfig:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    TARGET_SR = 16_000
    STFT_WINSIZE = 2048
    STFT_HOPSIZE = 384
    MELBINS = 229
    MEL_FMIN = 50
    MEL_FMAX = 8_000
    CONV1X1 = (200, 200)
    LEAKY_RELU_SLOPE = 0.1
    INFERENCE_CHUNK_SIZE_SECS = 20.0
    INFERENCE_CHUNK_OVERLAP_SECS = 1.0
    DECODER_GAUSS_STD = 1.0
    DECODER_GAUSS_KSIZE = 11

CONF = AppConfig()

logmel_fn = TorchWavToLogmel(
    CONF.TARGET_SR, CONF.STFT_WINSIZE, CONF.STFT_HOPSIZE, CONF.MELBINS,
    CONF.MEL_FMIN, CONF.MEL_FMAX
).to(CONF.DEVICE)

key_beg, key_end = PIANO_MIDI_RANGE
NUM_PIANO_KEYS = key_end - key_beg
note_decoder = OnsetVelocityNmsDecoder(
    NUM_PIANO_KEYS, nms_pool_ksize=3,
    gauss_conv_stddev=CONF.DECODER_GAUSS_STD,
    gauss_conv_ksize=CONF.DECODER_GAUSS_KSIZE,
    vel_pad_left=1, vel_pad_right=1
)
pedal_decoder = PedalDecoder(num_pedals=1, threshold=0.5)

SECS_PER_FRAME = CONF.STFT_HOPSIZE / CONF.TARGET_SR
INFERENCE_CHUNK_SIZE_FRAMES = round(CONF.INFERENCE_CHUNK_SIZE_SECS / SECS_PER_FRAME)
INFERENCE_CHUNK_OVERLAP_FRAMES = round(CONF.INFERENCE_CHUNK_OVERLAP_SECS / SECS_PER_FRAME)

def get_model(snapshot_path):
    num_mels = CONF.MELBINS
    model = OnsetsAndVelocities(
        in_chans=2,
        in_height=num_mels,
        out_height=NUM_PIANO_KEYS,
        conv1x1head=CONF.CONV1X1,
        bn_momentum=0,
        leaky_relu_slope=CONF.LEAKY_RELU_SLOPE,
        dropout_drop_p=0
    ).to(CONF.DEVICE)
    load_model(model, snapshot_path, eval_phase=True, to_cpu=(CONF.DEVICE=="cpu"))
    return model

def model_inference(model, x):
    with torch.no_grad():
        probs, vels, pedals = model(x)
        if isinstance(probs, (list, tuple)):
            probs = probs[-1]
        
        probs = torch.sigmoid(torch.nn.functional.pad(probs, (1, 0)))
        vels = torch.sigmoid(torch.nn.functional.pad(vels, (1, 0)))
        pedals = torch.sigmoid(torch.nn.functional.pad(pedals, (1, 0)))
        return probs, vels, pedals

def _run_inference_and_decode(model, logmel):
    onset_pred, vel_pred, pedal_pred = strided_inference(
        lambda x: model_inference(model, x),
        logmel,
        INFERENCE_CHUNK_SIZE_FRAMES,
        INFERENCE_CHUNK_OVERLAP_FRAMES
    )

    pred_df = note_decoder(onset_pred, vel_pred, pthresh=0.5)

    if pedal_pred.dim() == 2:
        pedal_pred = pedal_pred.unsqueeze(0)

    if pedal_pred.dim() != 3:
        pedal_pred = pedal_pred.view(pedal_pred.shape[0], 1, -1)

    events_df, _, _ = pedal_decoder(pedal_pred)
    return pred_df, events_df

def main():
    parser = argparse.ArgumentParser(description='Transcribe audio file.')
    parser.add_argument('audio_path', type=str, help='Path to the audio file.')
    parser.add_argument('model_path', type=str, help='Path to the model checkpoint.')
    args = parser.parse_args()

    import wave
    import numpy as np

    with wave.open(args.audio_path, 'rb') as wf:
        n_channels = wf.getnchannels()
        swidth = wf.getsampwidth()
        framerate = wf.getframerate()
        n_frames = wf.getnframes()
        frames = wf.readframes(n_frames)

    waveform = np.frombuffer(frames, dtype=np.int16)
    if n_channels > 1:
        waveform = waveform.reshape(-1, n_channels).T
    else:
        waveform = waveform.reshape(1, -1)
    waveform = torch.from_numpy(waveform.copy()).float()
    if n_channels > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    from ov_piano.utils import torch_resample_audio
    waveform = torch_resample_audio(waveform, framerate, CONF.TARGET_SR, mono=True, device=CONF.DEVICE)
    logmel = logmel_fn(waveform).unsqueeze(0)

    model = get_model(args.model_path)
    model.eval()
    pred_df, events_df = _run_inference_and_decode(model, logmel)

    print("--- Notes ---")
    print(pred_df)
    print("--- Pedals ---")
    print(events_df)

if __name__ == "__main__":
    main()


