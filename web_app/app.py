"""Flask web application for OnV+Pedal piano transcription."""
from collections import OrderedDict
import os
import sys
from threading import Lock
from flask import Flask, render_template, request, jsonify
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename

# Make sure to run `pip install Flask` in your `onsvel` conda environment

# --- Project-specific imports ---
# Support both `python web_app/app.py` and `flask --app web_app.app run`.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from ov_piano.transcription import (
    AudioPreprocessingError,
    PianoTranscriber,
    TranscriptionConfig,
)

# --- Configuration ---
# These parameters should match the ones used for training the model.
# We'll use the parameters from `03_evaluate_pedal_model.py` as a reference.
class AppConfig:
    """Configuration for the Flask application."""
    # Paths
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    STATIC_ASSETS_DIR = os.path.join(SCRIPT_DIR, "..", "assets")
    MODEL_SNAPSHOTS_DIR = os.path.join(SCRIPT_DIR, "..", "out", "model_snapshots")
    UPLOADS_DIR = os.path.join(SCRIPT_DIR, "..", "uploads")

    # Limits
    MAX_FILE_SIZE = 25 * 1024 * 1024  # 25 MB
    MAX_AUDIO_DURATION = 5 * 60  # 5 minutes
    MODEL_CACHE_SIZE = 2
    MAX_CONTENT_LENGTH = MAX_FILE_SIZE

    # Loading arbitrary uploaded PyTorch checkpoints is unsafe because PyTorch
    # deserialization can execute code. Keep this disabled for normal/server use;
    # opt in only for trusted local development.
    ALLOW_MODEL_UPLOADS = os.environ.get("ONSVEL_ALLOW_MODEL_UPLOADS", "").lower() in {
        "1",
        "true",
        "yes",
    }
    ALLOWED_UPLOADED_MODEL_EXTENSIONS = {".torch", ".pt", ".pth"}

CONF = AppConfig()
TRANSCRIPTION_CONF = TranscriptionConfig(
    # Use smaller chunks for the web server to avoid long blocking.
    inference_chunk_size_secs=20.0,
    inference_chunk_overlap_secs=1.0,
)

# --- Global Objects (initialized once) ---
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = CONF.MAX_CONTENT_LENGTH
transcriber = PianoTranscriber(TRANSCRIPTION_CONF)
_model_cache = OrderedDict()
_model_cache_lock = Lock()

# --- Flask Routes ---
@app.route("/")
def index():
    """Serves the main HTML page."""
    return render_template("index.html")

@app.route("/api/models")
def get_models():
    """Returns a list of available model checkpoints."""
    try:
        return jsonify(list(_available_checkpoints().keys()))
    except OSError as e:
        return jsonify({"error": str(e)}), 500


@app.errorhandler(RequestEntityTooLarge)
def handle_request_entity_too_large(_error):
    """Return JSON when Flask rejects an oversized upload."""
    limit_mb = CONF.MAX_CONTENT_LENGTH // 1024 // 1024
    return jsonify({"error": f"Request exceeds the upload limit of {limit_mb} MB."}), 413


@app.route("/api/transcribe", methods=["POST"])
def transcribe():
    """Handles audio upload and performs transcription."""
    audio_file, snapshot_path, error_response, error_code = _handle_files(request)
    if error_response:
        return error_response, error_code

    try:
        logmel = _process_audio(audio_file)
    except AudioPreprocessingError as e:
        return jsonify({"error": str(e)}), e.status_code

    try:
        model = _get_cached_model(snapshot_path)
        result = transcriber.run_inference_and_decode(model, logmel)

        return _format_results(result.notes, result.pedal_events, result.logmel)

    except Exception as e:
        # A bit of error logging to the console
        print(f"An error occurred during transcription: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"An internal error occurred during transcription: {e}"}), 500


def _model_cache_key(snapshot_path):
    """Build a cache key that changes when the checkpoint file or device changes."""
    abs_path = os.path.abspath(snapshot_path)
    stat = os.stat(abs_path)
    return (abs_path, stat.st_mtime_ns, str(transcriber.config.device))


def _get_cached_model(snapshot_path):
    """Return a cached model for the checkpoint/device, loading it on cache miss.

    The key includes the checkpoint modification time so replacing a checkpoint at
    the same path automatically causes a fresh load. The cache is bounded to avoid
    retaining too many large model instances in memory.
    """
    cache_key = _model_cache_key(snapshot_path)
    with _model_cache_lock:
        model = _model_cache.get(cache_key)
        if model is not None:
            _model_cache.move_to_end(cache_key)
            return model

        _evict_stale_model_cache_entries(cache_key)
        model = transcriber.load_model(cache_key[0])
        _model_cache[cache_key] = model
        _model_cache.move_to_end(cache_key)

        _enforce_model_cache_size_limit()

        return model


def _evict_stale_model_cache_entries(cache_key):
    """Drop older cached versions for the same checkpoint path and device."""
    abs_path, mtime_ns, device = cache_key
    stale_keys = [
        key for key in _model_cache
        if key[0] == abs_path and key[2] == device and key[1] != mtime_ns
    ]
    for key in stale_keys:
        _model_cache.pop(key, None)


def _model_cache_size_limit():
    """Return a non-negative model cache size limit."""
    try:
        return max(0, int(CONF.MODEL_CACHE_SIZE))
    except (TypeError, ValueError):
        return 0


def _enforce_model_cache_size_limit():
    """Evict least-recently-used model entries until the cache is within limit."""
    cache_size_limit = _model_cache_size_limit()
    while len(_model_cache) > cache_size_limit:
        _model_cache.popitem(last=False)

def _format_results(pred_df, events_df, logmel):
    """Formats the decoded predictions for the frontend."""
    notes = []
    max_notes = 5000
    for _, row in pred_df.iterrows():
        notes.append({
            "pitch": int(row["key"] + transcriber.key_beg),
            "start": float(row["t_idx"] * transcriber.secs_per_frame),
            "velocity": float(row["vel"]),
            "duration": 0.4  # The model doesn"t predict duration, so use a fixed value
        })
        if len(notes) >= max_notes:
            break

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
                        "start": float(onset_frame * transcriber.secs_per_frame),
                        "duration": float((offset_frame - onset_frame) * transcriber.secs_per_frame)
                    })
                    j = next_offset_idx + 1
            i += 1

    total_duration = float(logmel.shape[-1] * transcriber.secs_per_frame)

    return jsonify({
        "notes": notes,
        "pedals": pedals,
        "duration": total_duration
    })

def _process_audio(audio_file):
    """Loads, validates, and preprocesses the audio file."""
    # Check file size
    audio_file.seek(0, os.SEEK_END)
    file_length = audio_file.tell()
    audio_file.seek(0, os.SEEK_SET)

    if file_length > CONF.MAX_FILE_SIZE:
        raise AudioPreprocessingError(
            f"File size exceeds the limit of {CONF.MAX_FILE_SIZE // 1024 // 1024} MB.",
            status_code=413,
        )

    return transcriber.preprocess_audio(
        audio_file,
        max_duration_secs=CONF.MAX_AUDIO_DURATION,
        decode_with_pydub=True,
    )


# --- Main --- #


def _handle_files(request):
    """Handles file uploads and model selection."""
    if 'audio' not in request.files:
        return None, None, jsonify({"error": "No audio file in request"}), 400

    snapshot_path = None
    if 'model_file' in request.files:
        model_file = request.files['model_file']
        if model_file.filename != '':
            if not CONF.ALLOW_MODEL_UPLOADS:
                return None, None, jsonify({
                    "error": (
                        "Uploaded model checkpoints are disabled. Select a "
                        "server-listed checkpoint instead. For trusted local "
                        "development only, set ONSVEL_ALLOW_MODEL_UPLOADS=1."
                    )
                }), 403
            filename = secure_filename(model_file.filename)
            if not filename:
                return None, None, jsonify({"error": "Invalid model filename"}), 400
            extension = os.path.splitext(filename)[1].lower()
            if extension not in CONF.ALLOWED_UPLOADED_MODEL_EXTENSIONS:
                return None, None, jsonify({"error": "Unsupported model checkpoint extension"}), 400
            os.makedirs(CONF.UPLOADS_DIR, exist_ok=True)
            try:
                snapshot_path = _safe_join_existing_parent(CONF.UPLOADS_DIR, filename)
            except ValueError:
                return None, None, jsonify({"error": "Invalid model upload path"}), 400
            model_file.save(snapshot_path)

    if not snapshot_path:
        snapshot_name = request.form.get("model")
        if not snapshot_name:
            return None, None, jsonify({"error": "No model selected"}), 400
        snapshot_path = _resolve_model_path(snapshot_name)

    if not snapshot_path or not os.path.isfile(snapshot_path):
        return None, None, jsonify({"error": "Model checkpoint not found or not allowed"}), 404

    audio_file = request.files['audio']
    return audio_file, snapshot_path, None, None


def _resolve_model_path(snapshot_name):
    """Resolve only checkpoints returned by the server-side model listing."""
    if not _is_plain_checkpoint_name(snapshot_name):
        return None

    return _available_checkpoints().get(snapshot_name)


def _is_plain_checkpoint_name(snapshot_name):
    """Return True when ``snapshot_name`` is a single safe filename.

    Rejecting both POSIX and Windows separators keeps model selection portable and
    avoids platform-specific traversal surprises such as ``..\\model.torch`` on a
    Linux deployment.
    """
    if not isinstance(snapshot_name, str) or not snapshot_name:
        return False
    if "\x00" in snapshot_name or snapshot_name in {".", ".."}:
        return False
    if os.path.isabs(snapshot_name):
        return False
    if "/" in snapshot_name or "\\" in snapshot_name:
        return False
    return snapshot_name == os.path.basename(snapshot_name)


def _available_checkpoints():
    """Return server-listed checkpoints as ``display_name -> absolute_path``.

    Checkpoints are discovered only from configured server directories. User
    input is later resolved against this map instead of being joined into a path,
    which prevents path traversal and arbitrary file selection.
    """
    candidates = []
    for directory in (CONF.MODEL_SNAPSHOTS_DIR, CONF.STATIC_ASSETS_DIR):
        directory_abs = os.path.abspath(directory)
        if not os.path.isdir(directory_abs):
            continue
        for filename in os.listdir(directory_abs):
            if not filename.endswith(".torch") or not _is_plain_checkpoint_name(filename):
                continue
            path = os.path.abspath(os.path.join(directory_abs, filename))
            if not os.path.isfile(path) or not _is_path_within_directory(path, directory_abs):
                continue
            candidates.append({"name": filename, "path": path})

    models = sorted(
        candidates,
        key=lambda item: os.path.getmtime(item["path"]),
        reverse=True,
    )

    checkpoints = OrderedDict()
    for model in models:
        checkpoints.setdefault(model["name"], model["path"])
    return checkpoints


def _safe_join_existing_parent(directory, filename):
    """Join a sanitized filename to a directory and keep the result contained."""
    directory_abs = os.path.abspath(directory)
    path = os.path.abspath(os.path.join(directory_abs, filename))
    if not _is_path_within_directory(path, directory_abs):
        raise ValueError("Resolved path escapes upload directory")
    return path


def _is_path_within_directory(path, directory):
    """Return True when ``path`` is inside ``directory`` after normalization."""
    path_real = os.path.realpath(os.path.abspath(path))
    directory_real = os.path.realpath(os.path.abspath(directory))
    try:
        return os.path.commonpath([path_real, directory_real]) == directory_real
    except ValueError:
        # Raised on Windows when paths are on different drives.
        return False


if __name__ == "__main__":
    # Check for model directory
    if not os.path.isdir(CONF.MODEL_SNAPSHOTS_DIR):
        print("="*80)
        print("WARNING: Model directory is empty or not found.")
        print(f"Expected location: {os.path.abspath(CONF.MODEL_SNAPSHOTS_DIR)}")
        print("="*80)

    app.run(host="127.0.0.1", port=5001, debug=True)
