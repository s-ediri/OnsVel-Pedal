#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""Reusable audio-to-event transcription utilities.

This module centralizes the configuration, audio preprocessing, model loading,
strided inference, and note/pedal decoding used by both the Flask application
and the command-line transcription script.
"""

from __future__ import annotations

import io
import logging
import os
import subprocess
import wave
from dataclasses import dataclass, field
from typing import BinaryIO, Callable, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch

from ov_piano import PIANO_MIDI_RANGE
from ov_piano.inference import (
    OnsetVelocityNmsDecoder,
    PedalDecoder,
    model_outputs_to_probabilities,
    strided_inference,
)
from ov_piano.models.ov import OnsetsAndVelocities
from ov_piano.utils import (
    TorchWavToLogmel,
    format_load_model_warnings,
    load_model,
    torch_resample_audio,
)


AudioSource = Union[str, os.PathLike, BinaryIO]
ModelFactory = Callable[..., torch.nn.Module]
LOGGER = logging.getLogger(__name__)
OPUS_SIGNATURE_SCAN_BYTES = 64 * 1024
WINDOWS_DLL_LOAD_FAILURE_CODES = ("3221225781", "-1073741515", "0xc0000135")


def _default_device() -> str:
    """Return the preferred PyTorch device for transcription."""
    return "cuda" if torch.cuda.is_available() else "cpu"


@dataclass(frozen=True)
class TranscriptionConfig:
    """Configuration shared by audio preprocessing, model inference, and decoding."""

    # Runtime
    device: str = field(default_factory=_default_device)

    # Audio feature extraction. These values must match training.
    target_sr: int = 16_000
    stft_winsize: int = 2048
    stft_hopsize: int = 384
    melbins: int = 229
    mel_fmin: int = 50
    mel_fmax: int = 8_000

    # Model architecture. These values must match the checkpoint.
    conv1x1: Tuple[int, int] = (200, 200)
    leaky_relu_slope: float = 0.1

    # Chunked inference.
    inference_chunk_size_secs: float = 20.0
    inference_chunk_overlap_secs: float = 1.0

    # Note decoder.
    note_threshold: float = 0.5
    decoder_gauss_std: Optional[float] = 1.0
    decoder_gauss_ksize: Optional[int] = 11

    # Pedal decoder.
    num_pedals: int = 1
    pedal_threshold: float = 0.5

    @property
    def key_beg(self) -> int:
        return PIANO_MIDI_RANGE[0]

    @property
    def key_end(self) -> int:
        return PIANO_MIDI_RANGE[1]

    @property
    def num_piano_keys(self) -> int:
        return self.key_end - self.key_beg

    @property
    def secs_per_frame(self) -> float:
        return self.stft_hopsize / self.target_sr

    @property
    def inference_chunk_size_frames(self) -> int:
        return round(self.inference_chunk_size_secs / self.secs_per_frame)

    @property
    def inference_chunk_overlap_frames(self) -> int:
        return round(self.inference_chunk_overlap_secs / self.secs_per_frame)

    def validate(self) -> None:
        """Validate values that would otherwise fail later inside inference."""
        if self.target_sr <= 0:
            raise ValueError("target_sr must be positive")
        if self.stft_winsize <= 0:
            raise ValueError("stft_winsize must be positive")
        if self.stft_hopsize <= 0:
            raise ValueError("stft_hopsize must be positive")
        if self.melbins <= 0:
            raise ValueError("melbins must be positive")
        if self.mel_fmin < 0:
            raise ValueError("mel_fmin must be non-negative")
        if self.mel_fmax <= self.mel_fmin:
            raise ValueError("mel_fmax must be greater than mel_fmin")
        if len(self.conv1x1) == 0 or any(width <= 0 for width in self.conv1x1):
            raise ValueError("conv1x1 must contain positive layer widths")
        if self.inference_chunk_size_frames <= 0:
            raise ValueError("inference_chunk_size_secs must produce at least one frame")
        if self.inference_chunk_overlap_frames < 0:
            raise ValueError("inference_chunk_overlap_secs must be non-negative")
        if self.inference_chunk_overlap_frames >= self.inference_chunk_size_frames:
            raise ValueError("inference chunk overlap must be smaller than chunk size")
        if self.inference_chunk_overlap_frames % 2 != 0:
            raise ValueError("inference chunk overlap must produce an even number of frames")
        if not 0 <= self.note_threshold <= 1:
            raise ValueError("note_threshold must be in [0, 1]")
        if self.decoder_gauss_std is not None and self.decoder_gauss_std <= 0:
            raise ValueError("decoder_gauss_std must be positive when set")
        if self.decoder_gauss_ksize is not None:
            if self.decoder_gauss_ksize <= 0 or self.decoder_gauss_ksize % 2 == 0:
                raise ValueError("decoder_gauss_ksize must be a positive odd integer when set")
        if self.num_pedals <= 0:
            raise ValueError("num_pedals must be positive")
        if not 0 <= self.pedal_threshold <= 1:
            raise ValueError("pedal_threshold must be in [0, 1]")


@dataclass
class TranscriptionResult:
    """Decoded transcription output."""

    notes: pd.DataFrame
    pedal_events: pd.DataFrame
    logmel: Optional[torch.Tensor] = None


class AudioPreprocessingError(ValueError):
    """Raised when an audio file cannot be decoded or validated."""

    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.status_code = status_code


def build_logmel_extractor(config: TranscriptionConfig) -> TorchWavToLogmel:
    """Create the log-mel feature extractor for a transcription configuration."""
    return TorchWavToLogmel(
        config.target_sr,
        config.stft_winsize,
        config.stft_hopsize,
        config.melbins,
        config.mel_fmin,
        config.mel_fmax,
    ).to(config.device)


def build_note_decoder(config: TranscriptionConfig) -> OnsetVelocityNmsDecoder:
    """Create the note onset/velocity decoder for a transcription configuration."""
    return OnsetVelocityNmsDecoder(
        config.num_piano_keys,
        nms_pool_ksize=3,
        gauss_conv_stddev=config.decoder_gauss_std,
        gauss_conv_ksize=config.decoder_gauss_ksize,
        vel_pad_left=1,
        vel_pad_right=1,
    )


def build_pedal_decoder(config: TranscriptionConfig) -> PedalDecoder:
    """Create the pedal event decoder for a transcription configuration."""
    return PedalDecoder(num_pedals=config.num_pedals, threshold=config.pedal_threshold)


def build_transcription_model(
    config: TranscriptionConfig,
    model_factory: ModelFactory = OnsetsAndVelocities,
) -> torch.nn.Module:
    """Instantiate the model architecture expected by transcription checkpoints."""
    return model_factory(
        in_chans=2,
        in_height=config.melbins,
        out_height=config.num_piano_keys,
        conv1x1head=config.conv1x1,
        bn_momentum=0,
        leaky_relu_slope=config.leaky_relu_slope,
        dropout_drop_p=0,
    ).to(config.device)


def load_transcription_model(
    snapshot_path: Union[str, os.PathLike],
    config: TranscriptionConfig,
    model_factory: ModelFactory = OnsetsAndVelocities,
) -> torch.nn.Module:
    """Instantiate and load a transcription model checkpoint."""
    model = build_transcription_model(config, model_factory=model_factory)
    load_report = load_model(
        model,
        snapshot_path,
        eval_phase=True,
        to_cpu=str(config.device).startswith("cpu"),
        strict=False,
    )
    for warning in format_load_model_warnings(load_report):
        LOGGER.warning("CHECKPOINT LOAD WARNING: %s", warning)
    return model


def model_inference(model: torch.nn.Module, x: torch.Tensor):
    """Run the model and convert raw outputs to probability maps."""
    with torch.no_grad():
        return model_outputs_to_probabilities(model(x), include_pedals=True)


def _seek_to_start(source: AudioSource) -> None:
    try:
        source.seek(0)  # type: ignore[attr-defined]
    except (AttributeError, OSError, ValueError):
        pass


def _coerce_pathlike(source: AudioSource) -> AudioSource:
    """Convert pathlib-style paths to strings for third-party audio readers."""
    if isinstance(source, os.PathLike):
        return os.fspath(source)
    return source


def _source_filename(source: AudioSource) -> str:
    """Best-effort filename extraction for paths and uploaded file objects."""
    for attr_name in ("filename", "name"):
        value = getattr(source, attr_name, None)
        if isinstance(value, str) and value:
            return value
    if isinstance(source, (str, bytes, os.PathLike)):
        try:
            return os.fsdecode(source)
        except TypeError:
            return ""
    return ""


def _source_content_type(source: AudioSource) -> str:
    """Best-effort MIME/content-type extraction for uploaded file objects."""
    for attr_name in ("content_type", "mimetype", "type"):
        value = getattr(source, attr_name, None)
        if isinstance(value, str) and value:
            return value.lower()
    return ""


def _peek_source_bytes(source: AudioSource, max_bytes: int) -> bytes:
    """Read a small prefix from a path or stream without changing its position."""
    if max_bytes <= 0:
        return b""
    if isinstance(source, (str, bytes, os.PathLike)):
        try:
            with open(source, "rb") as file:
                return file.read(max_bytes)
        except OSError:
            return b""

    stream = getattr(source, "stream", source)
    read = getattr(stream, "read", None)
    if not callable(read):
        return b""

    try:
        position = stream.tell()
    except (AttributeError, OSError, ValueError):
        position = None

    try:
        _seek_to_start(stream)
        data = read(max_bytes)
    except (OSError, ValueError):
        return b""
    finally:
        try:
            if position is not None:
                stream.seek(position)
            else:
                _seek_to_start(stream)
        except (AttributeError, OSError, ValueError):
            pass

    return data if isinstance(data, bytes) else b""


def _source_looks_like_opus(source: AudioSource) -> bool:
    """Return True when metadata or bytes indicate an Opus audio source.

    Pydub normally runs ffprobe before ffmpeg. Some Windows installations have
    ffmpeg available but no working ffprobe, which causes Opus uploads to fail
    with ``JSONDecodeError: Expecting value`` before ffmpeg can decode them. For
    sources that clearly look like Opus, passing ``codec='opus'`` skips pydub's
    ffprobe probe and lets ffmpeg decode directly.
    """
    filename = _source_filename(source).lower()
    if os.path.splitext(filename)[1] == ".opus":
        return True

    content_type = _source_content_type(source)
    if "opus" in content_type:
        return True

    prefix = _peek_source_bytes(source, OPUS_SIGNATURE_SCAN_BYTES)
    return prefix.startswith(b"OggS") and b"OpusHead" in prefix


def _is_missing_pydub_audioop_dependency(exc: Exception) -> bool:
    """Return True when pydub failed because Python audioop support is missing."""
    missing_module_names = {"audioop", "pyaudioop"}
    if isinstance(exc, ModuleNotFoundError) and getattr(exc, "name", None) in missing_module_names:
        return True

    message = str(exc)
    return any(
        f"No module named '{module_name}'" in message
        or f'No module named "{module_name}"' in message
        for module_name in missing_module_names
    )


def _audio_tool_env_vars(executable_name: str) -> Tuple[str, ...]:
    prefix = executable_name.upper()
    return (
        f"ONSVEL_{prefix}_PATH",
        f"{prefix}_PATH",
        f"{prefix}_BINARY",
    )


def _normalize_executable_path(candidate: str, executable_name: str) -> str:
    """Return an executable path from a file or a directory candidate."""
    candidate = os.path.expanduser(os.path.expandvars(candidate.strip().strip('"')))
    if os.path.isdir(candidate):
        filename = f"{executable_name}.exe" if os.name == "nt" else executable_name
        candidate = os.path.join(candidate, filename)
    return candidate


def _audio_tool_filenames(executable_name: str) -> Tuple[str, ...]:
    """Return executable filenames to check for a command name on this OS."""
    if os.path.splitext(executable_name)[1]:
        return (executable_name,)
    if os.name == "nt":
        return (f"{executable_name}.exe", executable_name)
    return (executable_name,)


def _candidate_audio_tool_paths(executable_name: str):
    """Return existing executable candidates from env vars, PATH, and common dirs."""
    candidates = []
    seen = set()

    def add_candidate(path: str, source: str) -> None:
        candidate = _normalize_executable_path(path, executable_name)
        normalized = os.path.normcase(os.path.abspath(candidate))
        if normalized in seen:
            return
        if os.path.isfile(candidate):
            seen.add(normalized)
            candidates.append((candidate, source))

    for env_var in _audio_tool_env_vars(executable_name):
        configured_path = os.environ.get(env_var)
        if not configured_path:
            continue
        candidate = _normalize_executable_path(configured_path, executable_name)
        if os.path.isfile(candidate):
            add_candidate(candidate, env_var)
        else:
            LOGGER.warning(
                "Ignoring %s=%r because %s was not found there",
                env_var,
                configured_path,
                executable_name,
            )

    for directory in os.environ.get("PATH", "").split(os.pathsep):
        if not directory:
            continue
        for filename in _audio_tool_filenames(executable_name):
            add_candidate(os.path.join(directory, filename), "PATH")

    if os.name == "nt":
        for directory in (
            r"C:\Program Files\ffmpeg\bin",
            r"C:\ffmpeg\bin",
        ):
            for filename in _audio_tool_filenames(executable_name):
                add_candidate(os.path.join(directory, filename), "common Windows ffmpeg directory")

    return candidates


def _audio_tool_is_runnable(candidate: str) -> bool:
    """Return True when an ffmpeg/ffprobe executable can start successfully."""
    try:
        completed = subprocess.run(
            [candidate, "-version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        LOGGER.debug("Audio helper executable failed to start: %s", candidate, exc_info=True)
        return False

    return completed.returncode == 0


def _resolve_audio_tool_path(executable_name: str) -> Optional[str]:
    """Find an audio helper executable, preferring explicit env configuration.

    Pydub stores ``ffmpeg`` as a plain command name by default. That can be
    fragile on Windows when the Flask/IDE process has a different PATH than the
    interactive shell. Returning an absolute path makes subprocess startup much
    more deterministic and avoids accidentally picking another bundled ffmpeg.
    Each candidate is also started with ``-version`` so broken Conda/IDE PATH
    entries that fail with ``STATUS_DLL_NOT_FOUND`` are skipped automatically.
    """
    for candidate, source in _candidate_audio_tool_paths(executable_name):
        if _audio_tool_is_runnable(candidate):
            return candidate
        LOGGER.warning(
            "Ignoring %s candidate for %s because it did not start successfully: %s",
            source,
            executable_name,
            candidate,
        )

    return None


def _configure_pydub_binaries(AudioSegment) -> Tuple[Optional[str], Optional[str]]:
    """Configure pydub to launch deterministic ffmpeg/ffprobe executables."""
    ffmpeg_path = _resolve_audio_tool_path("ffmpeg")
    ffprobe_path = _resolve_audio_tool_path("ffprobe")

    if ffmpeg_path:
        AudioSegment.converter = ffmpeg_path
        # Older pydub versions expose this compatibility attribute; setting it is
        # harmless on newer versions and useful for tests/debug logs.
        AudioSegment.ffmpeg = ffmpeg_path

    if ffprobe_path:
        try:
            import pydub.utils as pydub_utils

            pydub_utils.get_prober_name = lambda: ffprobe_path
            AudioSegment.ffprobe = ffprobe_path
        except Exception:
            LOGGER.debug("Could not override pydub ffprobe path", exc_info=True)

    return ffmpeg_path, ffprobe_path


def _is_windows_dll_load_failure(exc: Exception) -> bool:
    """Return True for Windows STATUS_DLL_NOT_FOUND style ffmpeg failures."""
    message = str(exc).lower()
    return any(code in message for code in WINDOWS_DLL_LOAD_FAILURE_CODES)


def _resolved_audio_tool_summary() -> str:
    ffmpeg_path = _resolve_audio_tool_path("ffmpeg") or "not found"
    ffprobe_path = _resolve_audio_tool_path("ffprobe") or "not found"
    return f"Resolved ffmpeg: {ffmpeg_path}; resolved ffprobe: {ffprobe_path}."


def _pydub_decode_error_message(exc: Exception) -> str:
    """Build a helpful error message for pydub/ffmpeg decode failures."""
    details = str(exc).strip() or exc.__class__.__name__
    guidance = "Make sure you have ffmpeg installed and in your PATH."
    if _is_missing_pydub_audioop_dependency(exc):
        guidance = (
            "pydub could not import Python's audioop compatibility module. "
            "If you are using Python 3.13 or newer, install audioop-lts "
            "(`python -m pip install audioop-lts`) or use the supported "
            "environment.yml Conda environment (Python 3.9). For MP3 or other "
            "compressed audio, also make sure ffmpeg is installed and in your PATH."
        )
    elif _is_windows_dll_load_failure(exc):
        guidance = (
            "ffmpeg was found, but Windows could not start it because a required "
            "DLL is missing (STATUS_DLL_NOT_FOUND / 0xC0000135). Install a complete "
            "static ffmpeg build, put its bin directory before other ffmpeg copies "
            "in PATH, or set ONSVEL_FFMPEG_PATH and ONSVEL_FFPROBE_PATH to the full "
            "ffmpeg.exe/ffprobe.exe paths. "
            f"{_resolved_audio_tool_summary()}"
        )
    return f"Could not read audio file: {details}. {guidance}"


def _samples_from_pcm(frames: bytes, sample_width: int) -> np.ndarray:
    """Convert PCM bytes returned by :mod:`wave` into normalized float samples.

    MAESTRO HDF5 features are prepared with :func:`torchaudio.load`, whose
    default behavior returns floating-point audio normalized to roughly
    ``[-1.0, 1.0]``.  Keeping the same scale here is critical: feeding raw PCM
    integer magnitudes into the log-mel extractor shifts features by tens of dB
    and makes the note model massively over-predict.
    """
    if sample_width == 1:
        samples = np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128.0
        return samples / 128.0
    if sample_width == 2:
        return np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    if sample_width == 3:
        raw = np.frombuffer(frames, dtype=np.uint8)
        if raw.size % 3 != 0:
            raise AudioPreprocessingError("Invalid 24-bit PCM byte length")
        raw = raw.reshape(-1, 3)
        samples = (
            raw[:, 0].astype(np.int32)
            | (raw[:, 1].astype(np.int32) << 8)
            | (raw[:, 2].astype(np.int32) << 16)
        )
        samples[raw[:, 2] & 0x80 != 0] -= 1 << 24
        return samples.astype(np.float32) / 8388608.0
    if sample_width == 4:
        return np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
    raise AudioPreprocessingError(f"Unsupported WAV sample width: {sample_width} bytes")


def load_wav_waveform(source: AudioSource) -> Tuple[torch.Tensor, int]:
    """Load a PCM WAV source into a ``(channels, samples)`` float tensor.

    The returned tensor uses the same normalized floating-point scale as
    ``torchaudio.load`` because that is what the training feature-preparation
    pipeline used. Mono conversion and resampling are handled by
    :func:`preprocess_waveform`.
    """
    _seek_to_start(source)
    source = _coerce_pathlike(source)
    try:
        with wave.open(source, "rb") as wav_file:
            sample_rate = wav_file.getframerate()
            num_channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            num_frames = wav_file.getnframes()
            frames = wav_file.readframes(num_frames)
    except (wave.Error, EOFError, OSError) as exc:
        raise AudioPreprocessingError(f"Could not read WAV audio: {exc}") from exc

    samples = _samples_from_pcm(frames, sample_width)
    if num_channels > 1:
        waveform = samples.reshape(-1, num_channels).T
    else:
        waveform = samples.reshape(1, -1)

    return torch.from_numpy(waveform.copy()).float(), sample_rate


def _audio_segment_from_file(AudioSegment, source: AudioSource, **kwargs):
    """Call pydub with the source rewound and path-like values normalized."""
    _seek_to_start(source)
    return AudioSegment.from_file(_coerce_pathlike(source), **kwargs)


def _decode_audio_segment_with_pydub(AudioSegment, source: AudioSource):
    """Decode with pydub, using an Opus-specific fast path when appropriate."""
    if _source_looks_like_opus(source):
        try:
            return _audio_segment_from_file(AudioSegment, source, codec="opus")
        except Exception as opus_exc:
            LOGGER.debug("Opus-specific pydub decode failed; falling back to generic decode", exc_info=True)
            try:
                return _audio_segment_from_file(AudioSegment, source)
            except Exception as generic_exc:
                raise opus_exc from generic_exc

    return _audio_segment_from_file(AudioSegment, source)


def decode_audio_with_pydub(source: AudioSource) -> Tuple[torch.Tensor, int]:
    """Decode an arbitrary audio source through pydub/ffmpeg, then read as WAV."""
    try:
        from pydub import AudioSegment

        _configure_pydub_binaries(AudioSegment)
        audio = _decode_audio_segment_with_pydub(AudioSegment, source)
    except Exception as exc:
        raise AudioPreprocessingError(_pydub_decode_error_message(exc)) from exc

    wav_buffer = io.BytesIO()
    audio.export(wav_buffer, format="wav")
    wav_buffer.seek(0)
    return load_wav_waveform(wav_buffer)


def load_audio_waveform(source: AudioSource, decode_with_pydub: bool = False) -> Tuple[torch.Tensor, int]:
    """Load audio from a path or file-like object.

    Set ``decode_with_pydub=True`` when non-WAV uploads should be accepted.
    """
    if decode_with_pydub:
        return decode_audio_with_pydub(source)
    return load_wav_waveform(source)


def preprocess_waveform(
    waveform: torch.Tensor,
    sample_rate: int,
    config: TranscriptionConfig,
    logmel_fn: Optional[TorchWavToLogmel] = None,
    max_duration_secs: Optional[float] = None,
) -> torch.Tensor:
    """Resample, validate, and convert a waveform to batched log-mel features."""
    if sample_rate <= 0:
        raise AudioPreprocessingError("Audio sample rate must be positive")
    if max_duration_secs is not None and max_duration_secs < 0:
        raise ValueError("max_duration_secs must be non-negative when set")
    if waveform.numel() == 0 or waveform.shape[-1] == 0:
        raise AudioPreprocessingError("Empty or invalid audio file")
    if not waveform.is_floating_point():
        waveform = waveform.float()

    waveform = torch_resample_audio(
        waveform,
        sample_rate,
        config.target_sr,
        mono=True,
        device=config.device,
    )

    if waveform.numel() == 0 or waveform.shape[-1] == 0:
        raise AudioPreprocessingError("Empty or invalid audio file")

    duration = waveform.shape[-1] / config.target_sr
    if max_duration_secs is not None and duration > max_duration_secs:
        raise AudioPreprocessingError(
            f"Audio duration exceeds the limit of {int(max_duration_secs // 60)} minutes.",
            status_code=413,
        )

    if logmel_fn is None:
        logmel_fn = build_logmel_extractor(config)

    with torch.no_grad():
        return logmel_fn(waveform).unsqueeze(0)


def preprocess_audio_file(
    source: AudioSource,
    config: TranscriptionConfig,
    logmel_fn: Optional[TorchWavToLogmel] = None,
    max_duration_secs: Optional[float] = None,
    decode_with_pydub: bool = False,
) -> torch.Tensor:
    """Load an audio source and convert it to batched log-mel features."""
    waveform, sample_rate = load_audio_waveform(source, decode_with_pydub=decode_with_pydub)
    return preprocess_waveform(
        waveform,
        sample_rate,
        config,
        logmel_fn=logmel_fn,
        max_duration_secs=max_duration_secs,
    )


def normalize_pedal_prediction_shape(
    pedal_pred: torch.Tensor,
    num_pedals: int = 1,
    batch_size: Optional[int] = None,
) -> torch.Tensor:
    """Normalize pedal predictions to ``(batch, channels, frames)``.

    Older inference paths occasionally returned 1D or 2D pedal predictions.
    This helper keeps those paths explicit instead of blindly reshaping: 1D is
    treated as a single-batch sequence, 2D is interpreted as ``(batch, frames)``
    when the first dimension matches ``batch_size`` (or when ``num_pedals == 1``
    and it otherwise cannot be a pedal axis), and as ``(pedals, frames)`` when
    the first dimension matches ``num_pedals``.  Newer models may return three
    channels per logical pedal: ``[state, onset, offset]``.
    """
    if num_pedals <= 0:
        raise ValueError("num_pedals must be positive")
    valid_channels = {num_pedals, num_pedals * 3}
    if pedal_pred.dim() == 1:
        if pedal_pred.numel() % num_pedals != 0:
            raise ValueError("1D pedal predictions are not divisible by num_pedals")
        return pedal_pred.reshape(1, num_pedals, -1)
    if pedal_pred.dim() == 2:
        first_dim = pedal_pred.shape[0]
        if batch_size is not None and first_dim == batch_size:
            return pedal_pred.unsqueeze(1)
        if first_dim in valid_channels:
            return pedal_pred.unsqueeze(0)
        if num_pedals == 1:
            return pedal_pred.unsqueeze(1)
        raise ValueError(
            "Cannot infer 2D pedal prediction layout; provide a matching batch_size "
            "or use shape (num_pedals, frames) / (num_pedals*3, frames)."
        )
    if pedal_pred.dim() == 3:
        if batch_size is not None and pedal_pred.shape[0] != batch_size:
            raise ValueError(
                f"Pedal prediction batch size {pedal_pred.shape[0]} does not match input batch size {batch_size}"
            )
        if pedal_pred.shape[1] not in valid_channels:
            raise ValueError(
                f"Expected {num_pedals} pedal state channel(s) or {num_pedals * 3} state/onset/offset channel(s), got {pedal_pred.shape[1]}"
            )
        return pedal_pred
    if pedal_pred.shape[1] in valid_channels:
        return pedal_pred.reshape(pedal_pred.shape[0], num_pedals, -1)
    raise ValueError(f"Unsupported pedal prediction shape: {tuple(pedal_pred.shape)}")


def run_inference_and_decode(
    model: torch.nn.Module,
    logmel: torch.Tensor,
    config: TranscriptionConfig,
    note_decoder: Optional[OnsetVelocityNmsDecoder] = None,
    pedal_decoder: Optional[PedalDecoder] = None,
) -> TranscriptionResult:
    """Run strided model inference and decode notes plus pedal events."""
    config.validate()
    if logmel.dim() != 3:
        raise ValueError(f"Expected logmel shape (batch, mels, frames), got {tuple(logmel.shape)}")
    logmel = logmel.to(config.device)
    outputs = strided_inference(
        lambda x: model_inference(model, x),
        logmel,
        config.inference_chunk_size_frames,
        config.inference_chunk_overlap_frames,
    )
    if len(outputs) < 3:
        raise RuntimeError("Expected onset, velocity, and pedal predictions from the model")

    onset_pred, vel_pred, pedal_pred = outputs[:3]
    if pedal_pred is None:
        raise RuntimeError("Expected pedal predictions from the model")

    if note_decoder is None:
        note_decoder = build_note_decoder(config)
    if pedal_decoder is None:
        pedal_decoder = build_pedal_decoder(config)

    notes_df = note_decoder(onset_pred, vel_pred, pthresh=config.note_threshold)
    pedal_pred = normalize_pedal_prediction_shape(
        pedal_pred,
        num_pedals=config.num_pedals,
        batch_size=logmel.shape[0],
    )
    pedal_events_df, _, _ = pedal_decoder(pedal_pred)

    return TranscriptionResult(notes=notes_df, pedal_events=pedal_events_df, logmel=logmel)


class PianoTranscriber:
    """Stateful reusable transcriber with cached feature extractor and decoders."""

    def __init__(
        self,
        config: Optional[TranscriptionConfig] = None,
        model_factory: ModelFactory = OnsetsAndVelocities,
    ):
        self.config = config or TranscriptionConfig()
        self.config.validate()
        self.model_factory = model_factory
        self.logmel_fn = build_logmel_extractor(self.config)
        self.note_decoder = build_note_decoder(self.config)
        self.pedal_decoder = build_pedal_decoder(self.config)

    @property
    def key_beg(self) -> int:
        return self.config.key_beg

    @property
    def secs_per_frame(self) -> float:
        return self.config.secs_per_frame

    def load_model(self, snapshot_path: Union[str, os.PathLike]) -> torch.nn.Module:
        return load_transcription_model(
            snapshot_path,
            self.config,
            model_factory=self.model_factory,
        )

    def preprocess_audio(
        self,
        source: AudioSource,
        max_duration_secs: Optional[float] = None,
        decode_with_pydub: bool = False,
    ) -> torch.Tensor:
        return preprocess_audio_file(
            source,
            self.config,
            logmel_fn=self.logmel_fn,
            max_duration_secs=max_duration_secs,
            decode_with_pydub=decode_with_pydub,
        )

    def run_inference_and_decode(
        self,
        model: torch.nn.Module,
        logmel: torch.Tensor,
    ) -> TranscriptionResult:
        return run_inference_and_decode(
            model,
            logmel,
            self.config,
            note_decoder=self.note_decoder,
            pedal_decoder=self.pedal_decoder,
        )

    def transcribe_file(
        self,
        audio_source: AudioSource,
        snapshot_path: Union[str, os.PathLike],
        max_duration_secs: Optional[float] = None,
        decode_with_pydub: bool = False,
    ) -> TranscriptionResult:
        """Load audio and a checkpoint, then return decoded note/pedal events."""
        logmel = self.preprocess_audio(
            audio_source,
            max_duration_secs=max_duration_secs,
            decode_with_pydub=decode_with_pydub,
        )
        model = self.load_model(snapshot_path)
        return self.run_inference_and_decode(model, logmel)