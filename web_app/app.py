"""Flask web application for OnV+Pedal piano transcription."""
import os
import sys
import torch
import wave
import io
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

# Make sure to run `pip install Flask` in your `onsvel` conda environment

# --- Project-specific imports ---
# Support both `python web_app/app.py` and `flask --app web_app.app run`.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from ov_piano import PIANO_MIDI_RANGE
from ov_piano.models.ov import OnsetsAndVelocities
from ov_piano.utils import load_model, torch_resample_audio, TorchWavToLogmel
from ov_piano.inference import strided_inference, OnsetVelocityNmsDecoder, PedalDecoder

# --- Configuration ---
# These parameters should match the ones used for training the model.
# We'll use the parameters from `03_evaluate_pedal_model.py` as a reference.
class AppConfig:
    """Configuration for the Flask application."""
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Audio processing
    TARGET_SR = 16_000
    STFT_WINSIZE = 2048
    STFT_HOPSIZE = 384
    MELBINS = 229
    MEL_FMIN = 50
    MEL_FMAX = 8_000

    # Model architecture (must match the checkpoint)
    CONV1X1 = (200, 200)
    LEAKY_RELU_SLOPE = 0.1

    # Inference
    INFERENCE_CHUNK_SIZE_SECS = 20.0 # Use smaller chunks for web server to avoid long blocking
    INFERENCE_CHUNK_OVERLAP_SECS = 1.0

    # Decoder
    DECODER_GAUSS_STD = 1.0
    DECODER_GAUSS_KSIZE = 11

    # Paths
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    STATIC_ASSETS_DIR = os.path.join(SCRIPT_DIR, "..", "assets")
    MODEL_SNAPSHOTS_DIR = os.path.join(SCRIPT_DIR, "..", "assets")
    UPLOADS_DIR = os.path.join(SCRIPT_DIR, "..", "uploads")

    # Limits
    MAX_FILE_SIZE = 25 * 1024 * 1024  # 25 MB
    MAX_AUDIO_DURATION = 5 * 60  # 5 minutes

CONF = AppConfig()

# --- Global Objects (initialized once) ---
app = Flask(__name__)

# Log-mel spectrogram converter
logmel_fn = TorchWavToLogmel(
    CONF.TARGET_SR, CONF.STFT_WINSIZE, CONF.STFT_HOPSIZE, CONF.MELBINS,
    CONF.MEL_FMIN, CONF.MEL_FMAX
).to(CONF.DEVICE)

# Note and Pedal Decoders
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


# --- Helper Functions ---
def get_model(snapshot_path):
    """Instantiates and loads a model checkpoint."""
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
    load_model(model, snapshot_path, eval_phase=True, to_cpu=(CONF.DEVICE=="cpu"), strict=False)
    return model

def model_inference(model, x):
    """Wrapper around the model for strided_inference."""
    with torch.no_grad():
        probs, vels, pedals = model(x)
        # Use the last onset stage
        if isinstance(probs, (list, tuple)):
            probs = probs[-1]
        
        probs = torch.sigmoid(torch.nn.functional.pad(probs, (1, 0)))
        vels = torch.sigmoid(torch.nn.functional.pad(vels, (1, 0)))
        pedals = torch.sigmoid(torch.nn.functional.pad(pedals, (1, 0)))
        return probs, vels, pedals

# --- Flask Routes ---
@app.route("/")
def index():
    """Serves the main HTML page."""
    return render_template("index.html")

@app.route("/api/models")
def get_models():
    """Returns a list of available model checkpoints."""
    try:
        candidates = []
        for directory in [CONF.MODEL_SNAPSHOTS_DIR]:
            if not os.path.isdir(directory):
                continue
            for filename in os.listdir(directory):
                if filename.endswith(".torch"):
                    candidates.append({
                        "name": filename,
                        "path": os.path.join(directory, filename),
                    })

        models = sorted(
            candidates,
            key=lambda item: os.path.getmtime(item["path"]),
            reverse=True
        )
        return jsonify([item["name"] for item in models])
    except OSError as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/transcribe", methods=["POST"])
def transcribe():
    """Handles audio upload and performs transcription."""
    audio_file, snapshot_path, error_response, error_code = _handle_files(request)
    if error_response:
        return error_response, error_code

    logmel, error_response, error_code = _process_audio(audio_file)
    if error_response:
        return error_response, error_code

    try:

        # 3. Load Model and Run Inference
        model = get_model(snapshot_path)
        model.eval()
        pred_df, events_df = _run_inference_and_decode(model, logmel)

        # 6. Format for Frontend
        return _format_results(pred_df, events_df, logmel)

    except Exception as e:
        # A bit of error logging to the console
        print(f"An error occurred during transcription: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"An internal error occurred during transcription: {e}"}), 500

def _format_results(pred_df, events_df, logmel):
    """Formats the decoded predictions for the frontend."""
    notes = []
    for _, row in pred_df.iterrows():
        notes.append({
            "pitch": int(row["key"] + key_beg),
            "start": float(row["t_idx"] * SECS_PER_FRAME),
            "velocity": float(row["vel"]),
            "duration": 0.4  # The model doesn"t predict duration, so use a fixed value
        })

    pedals = []
    for pedal_idx, group in events_df.groupby("pedal_idx"):
        onsets = sorted(group[group["event_type"] == "onset"]["t_idx"].values)
        offsets = sorted(group[group["event_type"] == "offset"]["t_idx"].values)

        i, j = 0, 0
        while i < len(onsets):
            onset_frame = onsets[i]
            
            # Find the next offset that occurs after the current onset
            next_offset_idx = -1
            for k in range(j, len(offsets)):
                if offsets[k] > onset_frame:
                    next_offset_idx = k
                    break

            if next_offset_idx != -1:
                offset_frame = offsets[next_offset_idx]
                # Find the next onset to check if this offset is valid
                next_onset_frame = onsets[i+1] if (i + 1) < len(onsets) else float("inf")

                # The offset is valid if it occurs before the next onset
                if offset_frame < next_onset_frame:
                    pedals.append({
                        "start": float(onset_frame * SECS_PER_FRAME),
                        "duration": float((offset_frame - onset_frame) * SECS_PER_FRAME)
                    })
                    j = next_offset_idx + 1
            i += 1

    total_duration = float(logmel.shape[-1] * SECS_PER_FRAME)

    return jsonify({
        "notes": notes,
        "pedals": pedals,
        "duration": total_duration
    })

def _run_inference_and_decode(model, logmel):
    """Runs model inference and decodes the predictions."""
    onset_pred, vel_pred, pedal_pred = strided_inference(
        lambda x: model_inference(model, x),
        logmel,
        INFERENCE_CHUNK_SIZE_FRAMES,
        INFERENCE_CHUNK_OVERLAP_FRAMES
    )

    # Decode note predictions
    pred_df = note_decoder(onset_pred, vel_pred, pthresh=0.5)

    # Decode pedal predictions
    if pedal_pred.dim() == 2:
        pedal_pred = pedal_pred.unsqueeze(0)

    if pedal_pred.dim() != 3:
        pedal_pred = pedal_pred.view(pedal_pred.shape[0], 1, -1)

    events_df, _, _ = pedal_decoder(pedal_pred)
    return pred_df, events_df

def _process_audio(audio_file):
    """Loads, validates, and preprocesses the audio file."""
    # Check file size
    audio_file.seek(0, os.SEEK_END)
    file_length = audio_file.tell()
    audio_file.seek(0, os.SEEK_SET)

    if file_length > CONF.MAX_FILE_SIZE:
        return None, jsonify({"error": f"File size exceeds the limit of {CONF.MAX_FILE_SIZE // 1024 // 1024} MB."}), 413

    # 1. Load and Preprocess Audio
    from pydub import AudioSegment
    import numpy as np

    try:
        audio = AudioSegment.from_file(audio_file)
    except Exception as e:
        return None, jsonify({"error": f"Could not read audio file: {e}. Make sure you have ffmpeg installed and in your PATH."}), 400

    # Export to a temporary in-memory WAV file
    wav_buffer = io.BytesIO()
    audio.export(wav_buffer, format="wav")
    wav_buffer.seek(0)

    with wave.open(wav_buffer, "rb") as wav_file:
        sr = wav_file.getframerate()
        n_channels = wav_file.getnchannels()
        n_frames = wav_file.getnframes()
        frames = wav_file.readframes(n_frames)

    waveform = np.frombuffer(frames, dtype=np.int16)
    if n_channels > 1:
        waveform = waveform.reshape(-1, n_channels).T
    else:
        waveform = waveform.reshape(1, -1)
    waveform = torch.from_numpy(waveform.copy()).float()
    if n_channels > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    waveform = torch_resample_audio(waveform, sr, CONF.TARGET_SR, mono=True, device=CONF.DEVICE)





    # Check audio duration
    duration = waveform.shape[1] / CONF.TARGET_SR
    if duration > CONF.MAX_AUDIO_DURATION:
        return None, jsonify({"error": f"Audio duration exceeds the limit of {CONF.MAX_AUDIO_DURATION // 60} minutes."}), 413


    if waveform.shape[-1] == 0:
        return None, jsonify({"error": "Empty or invalid audio file"}), 400

    # 2. Get Log-mel Spectrogram
    logmel = logmel_fn(waveform).unsqueeze(0) # Add batch dimension
    return logmel, None, None


# --- Main --- #


def _handle_files(request):
    """Handles file uploads and model selection."""
    if 'audio' not in request.files:
        return None, None, jsonify({"error": "No audio file in request"}), 400

    snapshot_path = None
    if 'model_file' in request.files:
        model_file = request.files['model_file']
        if model_file.filename != '':
            filename = secure_filename(model_file.filename)
            if not os.path.exists(CONF.UPLOADS_DIR):
                os.makedirs(CONF.UPLOADS_DIR)
            snapshot_path = os.path.join(CONF.UPLOADS_DIR, filename)
            model_file.save(snapshot_path)

    if not snapshot_path:
        snapshot_name = request.form.get("model")
        if not snapshot_name:
            return None, None, jsonify({"error": "No model selected"}), 400
        snapshot_path = _resolve_model_path(snapshot_name)

    if not os.path.exists(snapshot_path):
        return None, None, jsonify({"error": f"Model checkpoint not found: {snapshot_path}"}), 404

    audio_file = request.files['audio']
    return audio_file, snapshot_path, None, None


def _resolve_model_path(snapshot_name):
    """Find a checkpoint by name in generated output first, then static assets."""
    # safe_name = secure_filename(snapshot_name)
    for directory in (CONF.MODEL_SNAPSHOTS_DIR, CONF.STATIC_ASSETS_DIR):
        candidate = os.path.join(directory, snapshot_name)
        if os.path.exists(candidate):
            return candidate
    return os.path.join(CONF.MODEL_SNAPSHOTS_DIR, snapshot_name)


if __name__ == "__main__":
    # Check for model directory
    if not os.path.isdir(CONF.MODEL_SNAPSHOTS_DIR):
        print("="*80)
        print("WARNING: Model directory is empty or not found.")
        print(f"Expected location: {os.path.abspath(CONF.MODEL_SNAPSHOTS_DIR)}")
        print("="*80)

    app.run(host="127.0.0.1", port=5001, debug=True)
