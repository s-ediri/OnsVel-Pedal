"""Flask web application for OnV+Pedal piano transcription."""
from collections import OrderedDict
import logging
import math
import os
import sys
from threading import Lock, local
import time
import urllib.request
import uuid
import wave

from flask import Flask, render_template, request, jsonify, send_from_directory
import numpy as np
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
    _configure_pydub_binaries,
    estimate_note_intervals,
    load_audio_waveform,
    paired_pedal_intervals,
)

LOGGER = logging.getLogger(__name__)

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
    GENERATED_AUDIO_DIR = os.path.join(UPLOADS_DIR, "generated_audio")

    # Limits
    MODEL_CACHE_SIZE = 2
    GENERATED_AUDIO_TTL_SECS = 6 * 60 * 60
    GENERATED_AUDIO_SAMPLE_RATE = 22_050
    GENERATED_AUDIO_RELEASE_SECS = 0.35
    GENERATED_AUDIO_MIN_NOTE_SECS = 0.05
    GENERATED_AUDIO_BALANCE_TO_REFERENCE = True
    GENERATED_AUDIO_BALANCE_GATE_FRACTION = 0.01
    GENERATED_AUDIO_BALANCE_MIN_ACTIVE_RMS = 1e-5
    GENERATED_AUDIO_BALANCE_MIN_GAIN = 0.03
    GENERATED_AUDIO_BALANCE_MAX_GAIN = 12.0
    GENERATED_AUDIO_BALANCE_PEAK_HEADROOM = 0.98
    GENERATED_AUDIO_SYNTHETIC_FALLBACK = os.environ.get(
        "ONSVEL_GENERATED_AUDIO_SYNTHETIC_FALLBACK",
        "1",
    ).lower() not in {"0", "false", "no"}
    GRAND_PIANO_SAMPLE_DIR = os.path.join(STATIC_ASSETS_DIR, "grand_piano_samples", "salamander")
    GRAND_PIANO_SAMPLE_BASE_URL = os.environ.get(
        "ONSVEL_GRAND_PIANO_SAMPLE_BASE_URL",
        "https://tonejs.github.io/audio/salamander/",
    )
    GRAND_PIANO_SAMPLE_DOWNLOAD_TIMEOUT_SECS = 20

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
transcriber = PianoTranscriber(TRANSCRIPTION_CONF)
_model_cache = OrderedDict()
_model_cache_lock = Lock()
_GRAND_PIANO_SAMPLE_FILES = OrderedDict([
    (21, "A0.mp3"),
    (24, "C1.mp3"),
    (27, "Ds1.mp3"),
    (30, "Fs1.mp3"),
    (33, "A1.mp3"),
    (36, "C2.mp3"),
    (39, "Ds2.mp3"),
    (42, "Fs2.mp3"),
    (45, "A2.mp3"),
    (48, "C3.mp3"),
    (51, "Ds3.mp3"),
    (54, "Fs3.mp3"),
    (57, "A3.mp3"),
    (60, "C4.mp3"),
    (63, "Ds4.mp3"),
    (66, "Fs4.mp3"),
    (69, "A4.mp3"),
    (72, "C5.mp3"),
    (75, "Ds5.mp3"),
    (78, "Fs5.mp3"),
    (81, "A5.mp3"),
    (84, "C6.mp3"),
    (87, "Ds6.mp3"),
    (90, "Fs6.mp3"),
    (93, "A6.mp3"),
    (96, "C7.mp3"),
    (99, "Ds7.mp3"),
    (102, "Fs7.mp3"),
    (105, "A7.mp3"),
    (108, "C8.mp3"),
])
_grand_piano_sample_cache = {}
_missing_grand_piano_sample_warnings = set()
_generated_audio_render_state = local()

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
    reference_audio_balance = _analyze_reference_audio_balance(audio_file)

    try:
        model = _get_cached_model(snapshot_path)
        result = transcriber.run_inference_and_decode(model, logmel)

        return _format_results(
            result.notes,
            result.pedal_events,
            result.logmel,
            reference_audio_balance=reference_audio_balance,
        )

    except Exception as e:
        # A bit of error logging to the console
        print(f"An error occurred during transcription: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"An internal error occurred during transcription: {e}"}), 500


@app.route("/api/generated-audio/<path:filename>")
def get_generated_audio(filename):
    """Serve generated piano-audio artifacts created for recent transcriptions."""
    if not _is_plain_generated_audio_name(filename):
        return jsonify({"error": "Generated audio not found"}), 404
    return send_from_directory(
        CONF.GENERATED_AUDIO_DIR,
        filename,
        mimetype="audio/wav",
        as_attachment=False,
        max_age=3600,
    )


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

def _format_results(pred_df, events_df, logmel, reference_audio_balance=None):
    """Formats the decoded predictions for the frontend."""
    total_duration = float(logmel.shape[-1] * transcriber.secs_per_frame)
    max_notes = 5000
    notes = estimate_note_intervals(
        pred_df,
        transcriber.secs_per_frame,
        key_beg=transcriber.key_beg,
        total_duration_secs=total_duration,
    )[:max_notes]
    pedals = _paired_pedal_intervals(
        events_df,
        transcriber.secs_per_frame,
        fallback_end_secs=total_duration,
    )

    generated_audio = _generate_piano_audio_artifact(
        notes,
        pedals,
        total_duration,
        reference_audio_balance=reference_audio_balance,
    )

    return jsonify({
        "notes": notes,
        "pedals": pedals,
        "duration": total_duration,
        "generated_audio": generated_audio,
    })


def _generate_piano_audio_artifact(notes, pedals, duration_secs, reference_audio_balance=None):
    """Render the decoded transcription into a timestamp-aligned WAV artifact.

    The generated file starts at timeline ``0`` and writes note samples directly
    at their decoded onset positions using recorded grand-piano samples. Serving
    a pre-rendered PCM WAV avoids the latency and scheduling jitter of triggering
    MIDI notes during playback, so the browser can synchronize the generated
    piano, source audio, and piano-roll playhead using media-element
    ``currentTime``.
    """
    _cleanup_old_generated_audio()
    sample_rate = _generated_audio_sample_rate()
    renderable_notes = [note for note in notes if _is_renderable_note(note)]
    render_duration = _generated_audio_duration(renderable_notes, pedals, duration_secs)
    _generated_audio_render_state.synthetic_fallback_used = False

    try:
        os.makedirs(CONF.GENERATED_AUDIO_DIR, exist_ok=True)
        filename = f"piano_{int(time.time() * 1000)}_{uuid.uuid4().hex[:10]}.wav"
        path = _safe_join_existing_parent(CONF.GENERATED_AUDIO_DIR, filename)
        samples = _render_grand_piano_audio(
            renderable_notes,
            pedals,
            render_duration,
            sample_rate,
        )
        samples, balance_info = _balance_generated_audio_to_reference(
            samples,
            reference_audio_balance,
        )
        _write_mono_pcm_wav(path, samples, sample_rate)
        engine = _generated_audio_engine_name()
    except Exception as exc:
        print(f"Generated piano audio rendering failed: {exc}")
        return {
            "url": None,
            "sample_rate": sample_rate,
            "duration": render_duration,
            "engine": _generated_audio_engine_name(),
            "latency_seconds": 0.0,
            "balance": {"applied": False, "reason": "render_failed"},
            "error": f"Generated grand piano audio could not be rendered: {exc}",
        }

    return {
        "url": f"/api/generated-audio/{filename}",
        "sample_rate": sample_rate,
        "duration": render_duration,
        "engine": engine,
        "latency_seconds": 0.0,
        "balance": balance_info,
    }


def _generated_audio_engine_name():
    """Return the renderer identifier for the current generated-audio request."""
    if getattr(_generated_audio_render_state, "synthetic_fallback_used", False):
        return "server-synthetic-piano-fallback-v1"
    return "server-sampled-grand-piano-salamander-v1"


def _analyze_reference_audio_balance(audio_source):
    """Measure uploaded/source audio loudness for generated-audio matching.

    The transcription path already accepts many audio formats through pydub. This
    helper reuses the same decoding utilities, but any failure is non-fatal: an
    unmeasurable reference should not prevent transcription or generated playback.
    """
    if not getattr(CONF, "GENERATED_AUDIO_BALANCE_TO_REFERENCE", True):
        return {"method": "active_rms", "usable": False, "reason": "disabled"}
    if audio_source is None:
        return {"method": "active_rms", "usable": False, "reason": "missing_reference"}

    try:
        samples, sample_rate = _decode_reference_audio_to_mono_samples(audio_source)
        stats = _audio_level_stats(samples)
        min_active_rms = _generated_audio_balance_min_active_rms()
        usable = stats["active_rms"] > min_active_rms and stats["peak"] > min_active_rms
        return {
            "method": "active_rms",
            "sample_rate": int(sample_rate),
            "usable": bool(usable),
            "rms": stats["rms"],
            "active_rms": stats["active_rms"],
            "peak": stats["peak"],
            "active_ratio": stats["active_ratio"],
            "reason": None if usable else "reference_too_quiet",
        }
    except Exception as exc:
        print(f"Reference audio level analysis failed: {exc}")
        return {
            "method": "active_rms",
            "usable": False,
            "reason": "analysis_failed",
            "error": str(exc),
        }
    finally:
        _rewind_audio_source(audio_source)


def _decode_reference_audio_to_mono_samples(audio_source):
    """Decode an uploaded reference audio source to mono float samples."""
    _rewind_audio_source(audio_source)
    decode_with_pydub = not _reference_source_looks_like_wav(audio_source)
    waveform, sample_rate = load_audio_waveform(
        audio_source,
        decode_with_pydub=decode_with_pydub,
    )

    waveform_np = waveform.detach().cpu().numpy().astype(np.float32, copy=False)
    if waveform_np.ndim == 0:
        mono = waveform_np.reshape(1)
    elif waveform_np.ndim == 1:
        mono = waveform_np
    else:
        mono = np.mean(waveform_np.reshape(waveform_np.shape[0], -1), axis=0)
    return np.asarray(mono, dtype=np.float32), int(sample_rate)


def _reference_source_looks_like_wav(audio_source):
    """Return True when the reference source has a RIFF/WAVE header."""
    prefix = _peek_audio_source_bytes(audio_source, 12)
    return prefix.startswith(b"RIFF") and prefix[8:12] == b"WAVE"


def _peek_audio_source_bytes(audio_source, max_bytes):
    """Read a small prefix from a path or stream without changing its position."""
    if max_bytes <= 0:
        return b""
    if isinstance(audio_source, (str, bytes, os.PathLike)):
        try:
            with open(audio_source, "rb") as file:
                return file.read(max_bytes)
        except OSError:
            return b""

    stream = getattr(audio_source, "stream", audio_source)
    read = getattr(stream, "read", None)
    if not callable(read):
        return b""

    try:
        position = stream.tell()
    except (AttributeError, OSError, ValueError):
        position = None

    try:
        stream.seek(0)
        data = read(max_bytes)
    except (AttributeError, OSError, ValueError):
        return b""
    finally:
        try:
            if position is not None:
                stream.seek(position)
            else:
                stream.seek(0)
        except (AttributeError, OSError, ValueError):
            pass

    return data if isinstance(data, bytes) else b""


def _rewind_audio_source(audio_source):
    """Best-effort rewind for Flask uploads, file-like objects, and paths."""
    for candidate in (audio_source, getattr(audio_source, "stream", None)):
        if candidate is None:
            continue
        seek = getattr(candidate, "seek", None)
        if not callable(seek):
            continue
        try:
            seek(0)
            return
        except (OSError, ValueError):
            continue


def _balance_generated_audio_to_reference(samples, reference_audio_balance):
    """Scale generated audio so its active RMS follows the source reference."""
    samples = np.asarray(samples, dtype=np.float32)
    rendered_stats = _audio_level_stats(samples)
    balance_info = {
        "method": "active_rms_match",
        "applied": False,
        "gain": 1.0,
        "raw_gain": 1.0,
        "reference_active_rms": None,
        "rendered_active_rms_before": rendered_stats["active_rms"],
        "rendered_peak_before": rendered_stats["peak"],
        "rendered_active_rms_after": rendered_stats["active_rms"],
        "rendered_peak_after": rendered_stats["peak"],
    }

    if not reference_audio_balance or not reference_audio_balance.get("usable"):
        balance_info["reason"] = (
            reference_audio_balance or {"reason": "missing_reference"}
        ).get("reason") or "missing_reference"
        return samples, balance_info

    target_rms = _safe_float(reference_audio_balance.get("active_rms"), 0.0)
    min_active_rms = _generated_audio_balance_min_active_rms()
    balance_info["reference_active_rms"] = target_rms
    if target_rms <= min_active_rms:
        balance_info["reason"] = "reference_too_quiet"
        return samples, balance_info
    if rendered_stats["active_rms"] <= min_active_rms or rendered_stats["peak"] <= min_active_rms:
        balance_info["reason"] = "generated_audio_too_quiet"
        return samples, balance_info

    raw_gain = target_rms / rendered_stats["active_rms"]
    gain = _clamp(
        raw_gain,
        _generated_audio_balance_min_gain(),
        _generated_audio_balance_max_gain(),
    )
    peak_headroom = _generated_audio_balance_peak_headroom()
    if rendered_stats["peak"] > 0.0:
        gain = min(gain, peak_headroom / rendered_stats["peak"])
    gain = max(0.0, gain)

    balanced_samples = np.clip(samples * gain, -1.0, 1.0).astype(np.float32)
    balanced_stats = _audio_level_stats(balanced_samples)
    balance_info.update({
        "applied": True,
        "gain": float(gain),
        "raw_gain": float(raw_gain),
        "reason": None,
        "rendered_active_rms_after": balanced_stats["active_rms"],
        "rendered_peak_after": balanced_stats["peak"],
    })
    if abs(gain - raw_gain) > 1e-6:
        balance_info["limited_by"] = "peak_headroom_or_gain_limits"
    return balanced_samples, balance_info


def _audio_level_stats(samples):
    """Return peak, full RMS, and gated active RMS for mono float audio."""
    samples = np.asarray(samples, dtype=np.float32).reshape(-1)
    if samples.size == 0:
        return {"peak": 0.0, "rms": 0.0, "active_rms": 0.0, "active_ratio": 0.0}

    finite_samples = samples[np.isfinite(samples)]
    if finite_samples.size == 0:
        return {"peak": 0.0, "rms": 0.0, "active_rms": 0.0, "active_ratio": 0.0}

    abs_samples = np.abs(finite_samples)
    peak = float(np.max(abs_samples)) if abs_samples.size else 0.0
    rms = float(np.sqrt(np.mean(np.square(finite_samples, dtype=np.float32))))
    if peak <= 0.0:
        return {"peak": peak, "rms": rms, "active_rms": 0.0, "active_ratio": 0.0}

    gate_fraction = _clamp(_safe_float(CONF.GENERATED_AUDIO_BALANCE_GATE_FRACTION, 0.01), 0.0, 1.0)
    gate = max(_generated_audio_balance_min_active_rms(), peak * gate_fraction)
    active_mask = abs_samples >= gate
    active_samples = finite_samples[active_mask]
    # If the gate captures almost nothing, fall back to full-file RMS so short or
    # very soft uploads still receive a stable estimate.
    if active_samples.size < max(1, int(0.001 * finite_samples.size)):
        active_samples = finite_samples
        active_ratio = 1.0
    else:
        active_ratio = float(active_samples.size / finite_samples.size)
    active_rms = float(np.sqrt(np.mean(np.square(active_samples, dtype=np.float32))))
    return {
        "peak": peak,
        "rms": rms,
        "active_rms": active_rms,
        "active_ratio": active_ratio,
    }


def _generated_audio_balance_min_active_rms():
    return max(0.0, _safe_float(CONF.GENERATED_AUDIO_BALANCE_MIN_ACTIVE_RMS, 1e-5))


def _generated_audio_balance_min_gain():
    return max(0.0, _safe_float(CONF.GENERATED_AUDIO_BALANCE_MIN_GAIN, 0.03))


def _generated_audio_balance_max_gain():
    min_gain = _generated_audio_balance_min_gain()
    return max(min_gain, _safe_float(CONF.GENERATED_AUDIO_BALANCE_MAX_GAIN, 12.0))


def _generated_audio_balance_peak_headroom():
    return _clamp(_safe_float(CONF.GENERATED_AUDIO_BALANCE_PEAK_HEADROOM, 0.98), 0.01, 1.0)


def _generated_audio_sample_rate():
    """Return a safe integer sample rate for generated piano WAV files."""
    try:
        sample_rate = int(CONF.GENERATED_AUDIO_SAMPLE_RATE)
    except (TypeError, ValueError):
        sample_rate = 22_050
    return max(8_000, min(sample_rate, 96_000))


def _generated_audio_duration(notes, pedals, duration_secs):
    """Return the WAV duration needed for notes, pedal holds, and releases."""
    duration = max(0.01, _safe_float(duration_secs, 0.0))
    release_secs = max(0.0, _safe_float(CONF.GENERATED_AUDIO_RELEASE_SECS, 0.35))

    for note in notes:
        note_start = max(0.0, _safe_float(note.get("start")))
        note_duration = max(
            _safe_float(CONF.GENERATED_AUDIO_MIN_NOTE_SECS, 0.05),
            _safe_float(note.get("duration")),
        )
        note_end = note_start + note_duration
        sustain_end = _sustain_end_for_note(note_start, note_end, pedals)
        held_until = max(note_end, sustain_end if sustain_end is not None else note_end)
        duration = max(duration, held_until + release_secs)

    for pedal in pedals or []:
        duration = max(duration, _pedal_end_secs(pedal))

    return float(duration)


def _render_grand_piano_audio(notes, pedals, duration_secs, sample_rate):
    """Render note events with recorded grand-piano samples."""
    num_samples = max(1, int(math.ceil(duration_secs * sample_rate)))
    mix = np.zeros(num_samples, dtype=np.float32)

    for note in notes:
        _add_grand_piano_sample_note(mix, note, pedals, sample_rate)

    peak = float(np.max(np.abs(mix))) if mix.size else 0.0
    if peak > 0.0:
        mix = (mix / peak) * 0.92

    return np.clip(mix, -1.0, 1.0)


def _add_grand_piano_sample_note(mix, note, pedals, sample_rate):
    """Add one recorded grand-piano sample note at its decoded timestamp."""
    pitch = int(round(_clamp(_safe_float(note.get("pitch"), 60.0), 21.0, 108.0)))
    velocity = _clamp(_safe_float(note.get("velocity"), 0.8), 0.05, 1.0)
    note_start = max(0.0, _safe_float(note.get("start")))
    note_duration = max(
        _safe_float(CONF.GENERATED_AUDIO_MIN_NOTE_SECS, 0.05),
        _safe_float(note.get("duration")),
    )
    note_end = note_start + note_duration
    sustain_end = _sustain_end_for_note(note_start, note_end, pedals)
    held_until = max(note_end, sustain_end if sustain_end is not None else note_end)
    render_end = held_until + max(0.0, _safe_float(CONF.GENERATED_AUDIO_RELEASE_SECS, 0.35))

    start_idx = int(round(note_start * sample_rate))
    end_idx = min(len(mix), int(round(render_end * sample_rate)))
    if start_idx >= len(mix) or end_idx <= start_idx:
        return

    sample_pitch, sample = _load_nearest_grand_piano_sample(pitch, sample_rate)
    pitched_sample = _resample_by_ratio(sample, 2.0 ** ((pitch - sample_pitch) / 12.0))
    note_audio = pitched_sample[:end_idx - start_idx].astype(np.float32, copy=True)
    if note_audio.size == 0:
        return

    _apply_note_release_envelope(
        note_audio,
        held_until - note_start,
        _safe_float(CONF.GENERATED_AUDIO_RELEASE_SECS, 0.35),
        sample_rate,
    )

    amplitude = 0.95 * (velocity ** 1.25)
    mix[start_idx:start_idx + note_audio.size] += amplitude * note_audio


def _load_nearest_grand_piano_sample(pitch, sample_rate):
    """Load the nearest recorded grand-piano sample for ``pitch``."""
    sample_pitch = min(_GRAND_PIANO_SAMPLE_FILES, key=lambda candidate: abs(candidate - pitch))
    return sample_pitch, _load_grand_piano_sample(sample_pitch, sample_rate)


def _load_grand_piano_sample(sample_pitch, sample_rate):
    """Return a decoded, mono, trimmed grand-piano sample at ``sample_rate``."""
    filename = _GRAND_PIANO_SAMPLE_FILES.get(sample_pitch)
    if not filename:
        raise RuntimeError(f"No grand piano sample is configured for MIDI pitch {sample_pitch}.")

    fallback_cache_key = ("synthetic-grand-piano", int(sample_pitch), int(sample_rate))
    fallback_cached = _grand_piano_sample_cache.get(fallback_cache_key)
    if fallback_cached is not None and not _local_grand_piano_sample_available(filename):
        _mark_generated_audio_synthetic_fallback_used()
        return fallback_cached

    try:
        path = _ensure_grand_piano_sample_file(filename)
        try:
            mtime_ns = os.stat(path).st_mtime_ns
        except OSError:
            mtime_ns = None
        cache_key = (os.path.abspath(path), mtime_ns, int(sample_rate))
        cached = _grand_piano_sample_cache.get(cache_key)
        if cached is not None:
            return cached

        samples, source_rate = _decode_audio_file_to_mono(path)
        samples = _trim_leading_silence(samples)
        if source_rate != sample_rate:
            samples = _resample_audio(samples, source_rate, sample_rate)
        peak = float(np.max(np.abs(samples))) if samples.size else 0.0
        if peak > 0.0:
            samples = (samples / peak).astype(np.float32)
        if samples.size == 0:
            raise RuntimeError(f"Grand piano sample is empty: {path}")

        _grand_piano_sample_cache[cache_key] = samples
        return samples
    except Exception as exc:
        if not _generated_audio_synthetic_fallback_enabled():
            raise
        return _fallback_grand_piano_sample(sample_pitch, sample_rate, filename, exc)


def _generated_audio_synthetic_fallback_enabled():
    """Return True when missing Salamander samples may use a local synth fallback."""
    value = getattr(CONF, "GENERATED_AUDIO_SYNTHETIC_FALLBACK", True)
    if isinstance(value, str):
        return value.lower() not in {"0", "false", "no"}
    return bool(value)


def _local_grand_piano_sample_available(filename):
    """Return True when a non-empty local Salamander sample file exists."""
    if not _is_plain_checkpoint_name(filename):
        return False
    sample_dir = os.path.abspath(CONF.GRAND_PIANO_SAMPLE_DIR)
    path = os.path.abspath(os.path.join(sample_dir, filename))
    if not _is_path_within_directory(path, sample_dir):
        return False
    try:
        return os.path.isfile(path) and os.path.getsize(path) > 0
    except OSError:
        return False


def _fallback_grand_piano_sample(sample_pitch, sample_rate, filename, exc):
    """Return a cached synthetic fallback sample when recorded samples are unavailable."""
    _mark_generated_audio_synthetic_fallback_used()
    _warn_about_grand_piano_sample_fallback_once(filename, exc)

    cache_key = ("synthetic-grand-piano", int(sample_pitch), int(sample_rate))
    cached = _grand_piano_sample_cache.get(cache_key)
    if cached is not None:
        return cached

    samples = _synthesize_grand_piano_sample(sample_pitch, sample_rate)
    _grand_piano_sample_cache[cache_key] = samples
    return samples


def _mark_generated_audio_synthetic_fallback_used():
    _generated_audio_render_state.synthetic_fallback_used = True


def _warn_about_grand_piano_sample_fallback_once(filename, exc):
    warning_key = filename
    if warning_key in _missing_grand_piano_sample_warnings:
        return
    _missing_grand_piano_sample_warnings.add(warning_key)
    sample_dir = os.path.abspath(CONF.GRAND_PIANO_SAMPLE_DIR)
    message = (
        f"Grand piano sample {filename} could not be loaded ({exc}); "
        "using the built-in synthetic piano fallback for generated playback. "
        f"For recorded Salamander piano audio, place samples in {sample_dir} "
        "or allow the configured sample download."
    )
    LOGGER.warning(message)


def _synthesize_grand_piano_sample(sample_pitch, sample_rate):
    """Generate a deterministic piano-like mono sample for offline fallback rendering."""
    safe_sample_rate = max(8_000, int(sample_rate))
    pitch = int(round(_clamp(_safe_float(sample_pitch, 60.0), 21.0, 108.0)))
    frequency = 440.0 * (2.0 ** ((pitch - 69.0) / 12.0))
    duration_secs = _clamp(5.0 - 0.03 * (pitch - 48), 2.2, 6.0)
    num_samples = max(1, int(round(duration_secs * safe_sample_rate)))
    t = np.arange(num_samples, dtype=np.float32) / float(safe_sample_rate)

    nyquist = 0.47 * safe_sample_rate
    tone = np.zeros(num_samples, dtype=np.float32)
    harmonics = (
        (1.0, 1.00, 0.00),
        (2.0, 0.42, 0.23),
        (3.0, 0.24, 0.41),
        (4.0, 0.13, 0.60),
        (5.0, 0.08, 0.79),
    )
    for multiple, amplitude, phase in harmonics:
        partial_frequency = frequency * multiple
        if partial_frequency < nyquist:
            tone += amplitude * np.sin((2.0 * math.pi * partial_frequency * t) + phase).astype(np.float32)

    detuned_frequency = frequency * 1.006
    if detuned_frequency < nyquist:
        tone += 0.12 * np.sin(2.0 * math.pi * detuned_frequency * t).astype(np.float32)

    attack = 1.0 - np.exp(-t / 0.006)
    decay_time = _clamp(3.4 - 0.028 * (pitch - 60), 0.9, 5.2)
    body_envelope = attack * np.exp(-t / decay_time)
    hammer_frequency = min(nyquist, max(frequency * 8.0, 1_200.0))
    hammer = 0.035 * np.sin(2.0 * math.pi * hammer_frequency * t) * np.exp(-t / 0.012)

    samples = ((tone * body_envelope) + hammer).astype(np.float32)
    peak = float(np.max(np.abs(samples))) if samples.size else 0.0
    if peak > 0.0:
        samples = (samples / peak).astype(np.float32)
    return samples


def _ensure_grand_piano_sample_file(filename):
    """Return a local grand-piano sample path, downloading it if necessary."""
    if not _is_plain_checkpoint_name(filename):
        raise RuntimeError("Invalid grand piano sample filename")

    sample_dir = os.path.abspath(CONF.GRAND_PIANO_SAMPLE_DIR)
    os.makedirs(sample_dir, exist_ok=True)
    path = _safe_join_existing_parent(sample_dir, filename)
    if os.path.isfile(path) and os.path.getsize(path) > 0:
        return path

    base_url = str(CONF.GRAND_PIANO_SAMPLE_BASE_URL or "").rstrip("/")
    if not base_url:
        raise RuntimeError(
            f"Grand piano sample {filename} is missing. Place it in {sample_dir}."
        )

    url = f"{base_url}/{filename}"
    tmp_path = f"{path}.tmp"
    try:
        with urllib.request.urlopen(
            url,
            timeout=max(1, int(CONF.GRAND_PIANO_SAMPLE_DOWNLOAD_TIMEOUT_SECS)),
        ) as response:
            data = response.read()
        if not data:
            raise RuntimeError("download returned an empty file")
        with open(tmp_path, "wb") as sample_file:
            sample_file.write(data)
        os.replace(tmp_path, path)
        return path
    except Exception as exc:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except OSError:
            pass
        raise RuntimeError(
            f"Grand piano sample {filename} is unavailable. Place the Salamander "
            f"grand-piano sample in {sample_dir} or allow download from {url}."
        ) from exc


def _decode_audio_file_to_mono(path):
    """Decode a local grand-piano sample file into mono float PCM."""
    try:
        from pydub import AudioSegment
    except ImportError as exc:
        raise RuntimeError("pydub is required to decode grand piano sample files.") from exc

    try:
        _configure_pydub_binaries(AudioSegment)
        audio = AudioSegment.from_file(path).set_channels(1)
    except Exception as exc:
        raise RuntimeError(f"Could not decode grand piano sample {path}: {exc}") from exc

    samples = np.array(audio.get_array_of_samples()).astype(np.float32)
    scale = float(1 << (8 * audio.sample_width - 1))
    if scale <= 0:
        raise RuntimeError(f"Unsupported grand piano sample width: {audio.sample_width}")
    return np.clip(samples / scale, -1.0, 1.0), int(audio.frame_rate)


def _trim_leading_silence(samples, threshold=1e-4, preroll_samples=64):
    """Trim sample-start silence so rendered onsets align to transcription time."""
    if samples.size == 0:
        return samples.astype(np.float32)
    active = np.flatnonzero(np.abs(samples) > threshold)
    if active.size == 0:
        return samples.astype(np.float32)
    start = max(0, int(active[0]) - int(preroll_samples))
    return samples[start:].astype(np.float32)


def _resample_audio(samples, source_rate, target_rate):
    """Resample mono PCM with linear interpolation."""
    if source_rate <= 0 or target_rate <= 0:
        raise RuntimeError("Sample rates must be positive for grand piano rendering.")
    if int(source_rate) == int(target_rate) or samples.size <= 1:
        return samples.astype(np.float32)
    target_length = max(1, int(round(samples.size * (float(target_rate) / float(source_rate)))))
    source_positions = np.arange(samples.size, dtype=np.float32)
    target_positions = np.linspace(0, samples.size - 1, target_length, dtype=np.float32)
    return np.interp(target_positions, source_positions, samples).astype(np.float32)


def _resample_by_ratio(samples, ratio):
    """Play a sample faster/slower for pitch-shifting without changing onset time."""
    safe_ratio = max(0.01, float(ratio))
    if samples.size <= 1:
        return samples.astype(np.float32)
    output_length = max(1, int(math.ceil(samples.size / safe_ratio)))
    source_positions = np.arange(samples.size, dtype=np.float32)
    playback_positions = np.arange(output_length, dtype=np.float32) * safe_ratio
    playback_positions = np.clip(playback_positions, 0, samples.size - 1)
    return np.interp(playback_positions, source_positions, samples).astype(np.float32)


def _apply_note_release_envelope(samples, held_duration_secs, release_secs, sample_rate):
    """Apply note-off fade to recorded samples without changing their attack."""
    release_start = int(round(max(0.0, held_duration_secs) * sample_rate))
    if release_start >= samples.size:
        return
    release_length = max(1, int(round(max(0.01, release_secs) * sample_rate)))
    fade_length = min(release_length, samples.size - release_start)
    samples[release_start:release_start + fade_length] *= np.linspace(
        1.0,
        0.0,
        fade_length,
        dtype=np.float32,
    )
    if release_start + fade_length < samples.size:
        samples[release_start + fade_length:] = 0.0


def _write_mono_pcm_wav(path, samples, sample_rate):
    """Write normalized mono float samples as 16-bit PCM WAV."""
    pcm = (np.asarray(samples, dtype=np.float32) * 32767.0).astype("<i2")
    with wave.open(path, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm.tobytes())


def _is_renderable_note(note):
    """Return True when a note dict has finite pitch, start, and duration."""
    if not isinstance(note, dict):
        return False
    pitch = _safe_float(note.get("pitch"), math.nan)
    start = _safe_float(note.get("start"), math.nan)
    duration = _safe_float(note.get("duration"), math.nan)
    return math.isfinite(pitch) and math.isfinite(start) and math.isfinite(duration) and start >= 0.0


def _sustain_end_for_note(note_start, note_end, pedals):
    """Return the latest sustain-pedal end affecting a note, if any."""
    sustain_end = None
    for pedal in pedals or []:
        if int(round(_safe_float(pedal.get("pedal_idx"), 0.0))) != 0:
            continue
        pedal_start = _pedal_start_secs(pedal)
        pedal_end = _pedal_end_secs(pedal)
        if pedal_end <= pedal_start:
            continue
        overlaps_note = pedal_start <= note_end and pedal_end >= note_start
        pedal_down_at_note_end = pedal_start <= note_end <= pedal_end
        if overlaps_note or pedal_down_at_note_end:
            sustain_end = max(sustain_end if sustain_end is not None else pedal_end, pedal_end)
    return sustain_end


def _pedal_start_secs(pedal):
    """Return a non-negative pedal start time for frontend interval dicts."""
    if not isinstance(pedal, dict):
        return 0.0
    return max(0.0, _safe_float(pedal.get("start"), 0.0))


def _pedal_end_secs(pedal):
    """Return a non-negative pedal end time for frontend interval dicts."""
    if not isinstance(pedal, dict):
        return 0.0
    start = _pedal_start_secs(pedal)
    explicit_end = _safe_float(pedal.get("end"), math.nan)
    if math.isfinite(explicit_end):
        return max(start, explicit_end)
    return start + max(0.0, _safe_float(pedal.get("duration"), 0.0))


def _safe_float(value, default=0.0):
    """Convert ``value`` to a finite float, returning ``default`` otherwise."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _clamp(value, lower, upper):
    """Clamp ``value`` to the inclusive ``[lower, upper]`` range."""
    return min(upper, max(lower, value))


def _cleanup_old_generated_audio(now=None):
    """Remove stale generated WAV files to avoid unbounded upload growth."""
    if not os.path.isdir(CONF.GENERATED_AUDIO_DIR):
        return
    ttl = max(0, int(getattr(CONF, "GENERATED_AUDIO_TTL_SECS", 0) or 0))
    if ttl <= 0:
        return
    cutoff = (time.time() if now is None else now) - ttl
    for filename in os.listdir(CONF.GENERATED_AUDIO_DIR):
        if not _is_plain_generated_audio_name(filename):
            continue
        path = os.path.join(CONF.GENERATED_AUDIO_DIR, filename)
        try:
            if os.path.getmtime(path) < cutoff:
                os.remove(path)
        except OSError:
            pass


def _is_plain_generated_audio_name(filename):
    """Return True for generated WAV basenames served by the Flask route."""
    return (
        _is_plain_checkpoint_name(filename)
        and filename.startswith("piano_")
        and filename.lower().endswith(".wav")
    )


def _paired_pedal_intervals(events_df, secs_per_frame, fallback_end_secs=None):
    """Return frontend-ready sustain-pedal hold intervals.

    The decoder emits discrete pedal-down/onset and pedal-up/offset events. The
    piano roll needs paired intervals, so each onset is matched with the next
    later offset for the same pedal. If the final onset has no offset, cap the
    displayed hold and mark the end as estimated so a missed pedal-up event does
    not make the rest of the piece look sustained.
    """
    return paired_pedal_intervals(
        events_df,
        secs_per_frame,
        fallback_end_secs=fallback_end_secs,
    )

def _process_audio(audio_file):
    """Loads and preprocesses the audio file without upload size/duration caps."""
    return transcriber.preprocess_audio(
        audio_file,
        max_duration_secs=None,
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
