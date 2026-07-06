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
    except (AttributeError, OSError):
        pass


def _coerce_pathlike(source: AudioSource) -> AudioSource:
    """Convert pathlib-style paths to strings for third-party audio readers."""
    if isinstance(source, os.PathLike):
        return os.fspath(source)
    return source


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
    return f"Could not read audio file: {details}. {guidance}"


def _samples_from_pcm(frames: bytes, sample_width: int) -> np.ndarray:
    """Convert PCM bytes returned by :mod:`wave` into a flat numeric array."""
    if sample_width == 1:
        return np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128.0
    if sample_width == 2:
        return np.frombuffer(frames, dtype=np.int16)
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
        return samples
    if sample_width == 4:
        return np.frombuffer(frames, dtype=np.int32)
    raise AudioPreprocessingError(f"Unsupported WAV sample width: {sample_width} bytes")


def load_wav_waveform(source: AudioSource) -> Tuple[torch.Tensor, int]:
    """Load a PCM WAV source into a ``(channels, samples)`` float tensor.

    The returned tensor preserves the integer PCM scale used by the previous
    web/CLI implementations; mono conversion and resampling are handled by
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


def decode_audio_with_pydub(source: AudioSource) -> Tuple[torch.Tensor, int]:
    """Decode an arbitrary audio source through pydub/ffmpeg, then read as WAV."""
    try:
        from pydub import AudioSegment

        _seek_to_start(source)
        source = _coerce_pathlike(source)
        audio = AudioSegment.from_file(source)
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
    """Normalize pedal predictions to ``(batch, pedals, frames)``.

    Older inference paths occasionally returned 1D or 2D pedal predictions.
    This helper keeps those paths explicit instead of blindly reshaping: 1D is
    treated as a single-batch sequence, 2D is interpreted as ``(batch, frames)``
    when the first dimension matches ``batch_size`` (or when ``num_pedals == 1``
    and it otherwise cannot be a pedal axis), and as ``(pedals, frames)`` when
    the first dimension matches ``num_pedals``.
    """
    if num_pedals <= 0:
        raise ValueError("num_pedals must be positive")
    if pedal_pred.dim() == 1:
        if pedal_pred.numel() % num_pedals != 0:
            raise ValueError("1D pedal predictions are not divisible by num_pedals")
        return pedal_pred.reshape(1, num_pedals, -1)
    if pedal_pred.dim() == 2:
        first_dim = pedal_pred.shape[0]
        if batch_size is not None and first_dim == batch_size:
            return pedal_pred.unsqueeze(1)
        if first_dim == num_pedals:
            return pedal_pred.unsqueeze(0)
        if num_pedals == 1:
            return pedal_pred.unsqueeze(1)
        raise ValueError(
            "Cannot infer 2D pedal prediction layout; provide a matching batch_size "
            "or use shape (num_pedals, frames)."
        )
    if pedal_pred.dim() == 3:
        if batch_size is not None and pedal_pred.shape[0] != batch_size:
            raise ValueError(
                f"Pedal prediction batch size {pedal_pred.shape[0]} does not match input batch size {batch_size}"
            )
        if pedal_pred.shape[1] != num_pedals:
            raise ValueError(f"Expected {num_pedals} pedal channel(s), got {pedal_pred.shape[1]}")
        return pedal_pred
    if pedal_pred.shape[1] == num_pedals:
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