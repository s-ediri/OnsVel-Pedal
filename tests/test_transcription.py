import io
import json
import os
import wave

import pytest
import torch

from ov_piano.transcription import (
    AudioPreprocessingError,
    TranscriptionConfig,
    _configure_pydub_binaries,
    _decode_audio_segment_with_pydub,
    _pydub_decode_error_message,
    _source_looks_like_opus,
    normalize_pedal_prediction_shape,
    load_wav_waveform,
    PianoTranscriber,
    preprocess_waveform,
    run_inference_and_decode,
)
from scripts.transcribe import result_to_json_payload, write_json_output
from ov_piano.utils import TorchWavToLogmel


def _wav_bytes(samples, sample_rate=16_000, num_channels=1):
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(num_channels)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(samples.to(torch.int16).numpy().tobytes())
    buffer.seek(0)
    return buffer


def _sine_wave(sample_rate=16_000, duration_secs=0.25, frequency=440.0):
    t = torch.arange(int(sample_rate * duration_secs), dtype=torch.float32) / sample_rate
    return torch.sin(2 * torch.pi * frequency * t).unsqueeze(0)


def test_transcription_config_derives_frame_values():
    config = TranscriptionConfig(
        device="cpu",
        target_sr=100,
        stft_hopsize=10,
        inference_chunk_size_secs=1.0,
        inference_chunk_overlap_secs=0.2,
    )

    config.validate()

    assert config.secs_per_frame == 0.1
    assert config.inference_chunk_size_frames == 10
    assert config.inference_chunk_overlap_frames == 2
    assert config.num_piano_keys == 88


def test_transcription_config_rejects_odd_overlap_frames():
    config = TranscriptionConfig(
        device="cpu",
        target_sr=100,
        stft_hopsize=10,
        inference_chunk_size_secs=1.0,
        inference_chunk_overlap_secs=0.1,
    )

    with pytest.raises(ValueError, match="even number of frames"):
        config.validate()


def test_transcription_config_rejects_invalid_decoder_values():
    with pytest.raises(ValueError, match="note_threshold"):
        TranscriptionConfig(device="cpu", note_threshold=1.1).validate()
    with pytest.raises(ValueError, match="decoder_gauss_ksize"):
        TranscriptionConfig(device="cpu", decoder_gauss_ksize=10).validate()
    with pytest.raises(ValueError, match="pedal_threshold"):
        TranscriptionConfig(device="cpu", pedal_threshold=-0.1).validate()


def test_load_wav_waveform_reads_stereo_pcm_as_normalized_float_audio():
    interleaved_samples = torch.tensor(
        [0, 32767, -32768, 16384, 1, -1],
        dtype=torch.int16,
    )

    waveform, sample_rate = load_wav_waveform(
        _wav_bytes(interleaved_samples, sample_rate=8_000, num_channels=2)
    )

    assert sample_rate == 8_000
    assert waveform.shape == (2, 3)
    assert waveform.dtype == torch.float32
    assert torch.allclose(waveform[0], torch.tensor([0.0, -1.0, 1.0 / 32768.0]))
    assert torch.allclose(
        waveform[1],
        torch.tensor([32767.0 / 32768.0, 0.5, -1.0 / 32768.0]),
    )
    assert float(waveform.abs().max()) <= 1.0


def test_pydub_decode_error_message_mentions_audioop_lts_for_python_313():
    exc = ModuleNotFoundError("No module named 'pyaudioop'", name="pyaudioop")

    message = _pydub_decode_error_message(exc)

    assert "audioop-lts" in message
    assert "Python 3.13" in message
    assert "ffmpeg" in message


def test_pydub_decode_error_message_explains_windows_dll_load_failure():
    exc = RuntimeError("Decoding failed. ffmpeg returned error code: 3221225781")

    message = _pydub_decode_error_message(exc)

    assert "0xC0000135" in message
    assert "required DLL is missing" in message
    assert "ONSVEL_FFMPEG_PATH" in message


def test_configure_pydub_binaries_uses_explicit_env_paths(monkeypatch, tmp_path):
    ffmpeg = tmp_path / "ffmpeg.exe"
    ffprobe = tmp_path / "ffprobe.exe"
    ffmpeg.write_text("", encoding="utf-8")
    ffprobe.write_text("", encoding="utf-8")

    monkeypatch.setenv("ONSVEL_FFMPEG_PATH", str(ffmpeg))
    monkeypatch.setenv("ONSVEL_FFPROBE_PATH", str(ffprobe))
    monkeypatch.setattr("ov_piano.transcription._audio_tool_is_runnable", lambda path: True)

    class FakeAudioSegment:
        converter = "ffmpeg"

    import pydub.utils as pydub_utils

    original_get_prober_name = pydub_utils.get_prober_name
    try:
        resolved_ffmpeg, resolved_ffprobe = _configure_pydub_binaries(FakeAudioSegment)

        assert resolved_ffmpeg == str(ffmpeg)
        assert resolved_ffprobe == str(ffprobe)
        assert FakeAudioSegment.converter == str(ffmpeg)
        assert FakeAudioSegment.ffmpeg == str(ffmpeg)
        assert FakeAudioSegment.ffprobe == str(ffprobe)
        assert pydub_utils.get_prober_name() == str(ffprobe)
    finally:
        monkeypatch.setattr(pydub_utils, "get_prober_name", original_get_prober_name)


def test_configure_pydub_binaries_skips_broken_path_candidate(monkeypatch, tmp_path):
    broken_dir = tmp_path / "broken"
    working_dir = tmp_path / "working"
    broken_dir.mkdir()
    working_dir.mkdir()
    broken = broken_dir / "ffmpeg.exe"
    working = working_dir / "ffmpeg.exe"
    broken.write_text("", encoding="utf-8")
    working.write_text("", encoding="utf-8")

    monkeypatch.delenv("ONSVEL_FFMPEG_PATH", raising=False)
    monkeypatch.delenv("FFMPEG_PATH", raising=False)
    monkeypatch.delenv("FFMPEG_BINARY", raising=False)
    monkeypatch.setenv("PATH", os.pathsep.join([str(broken_dir), str(working_dir)]))
    monkeypatch.setattr(
        "ov_piano.transcription._audio_tool_is_runnable",
        lambda path: os.path.abspath(path) == os.path.abspath(working),
    )

    class FakeAudioSegment:
        converter = "ffmpeg"

    resolved_ffmpeg, _ = _configure_pydub_binaries(FakeAudioSegment)

    assert resolved_ffmpeg == str(working)
    assert FakeAudioSegment.converter == str(working)


def test_source_looks_like_opus_uses_extension_mime_and_signature():
    by_extension = io.BytesIO(b"not inspected")
    by_extension.name = "recording.opus"
    assert _source_looks_like_opus(by_extension)

    class UploadedAudio:
        filename = "recording.bin"
        content_type = "audio/ogg; codecs=opus"

        def __init__(self):
            self.stream = io.BytesIO(b"not inspected")

    assert _source_looks_like_opus(UploadedAudio())

    by_signature = io.BytesIO(b"OggS" + (b"\x00" * 100) + b"OpusHead" + b"payload")
    by_signature.seek(7)

    assert _source_looks_like_opus(by_signature)
    assert by_signature.tell() == 7
    assert not _source_looks_like_opus(io.BytesIO(b"OggS" + b"Vorbis"))


def test_pydub_opus_decode_fast_path_skips_generic_probe():
    calls = []

    class FakeAudioSegment:
        @staticmethod
        def from_file(source, **kwargs):
            calls.append((source.tell(), kwargs))
            return "decoded"

    source = io.BytesIO(b"not a real opus file")
    source.name = "clip.opus"
    source.seek(5)

    assert _decode_audio_segment_with_pydub(FakeAudioSegment, source) == "decoded"
    assert calls == [(0, {"codec": "opus"})]


def test_pydub_opus_decode_falls_back_to_generic_decode():
    calls = []

    class FakeAudioSegment:
        @staticmethod
        def from_file(source, **kwargs):
            calls.append(kwargs)
            if kwargs.get("codec") == "opus":
                raise RuntimeError("forced opus decode failure")
            return "decoded generically"

    source = io.BytesIO(b"not a real opus file")
    source.name = "clip.opus"

    assert _decode_audio_segment_with_pydub(FakeAudioSegment, source) == "decoded generically"
    assert calls == [{"codec": "opus"}, {}]


def test_preprocess_waveform_uses_injected_logmel_and_checks_duration():
    config = TranscriptionConfig(device="cpu", target_sr=16_000)
    waveform = torch.ones(1, 16_000)

    def fake_logmel_fn(wave):
        assert wave.shape == waveform.shape
        return torch.zeros(4, 7)

    logmel = preprocess_waveform(
        waveform,
        sample_rate=16_000,
        config=config,
        logmel_fn=fake_logmel_fn,
        max_duration_secs=2.0,
    )

    assert logmel.shape == (1, 4, 7)

    with pytest.raises(AudioPreprocessingError) as exc_info:
        preprocess_waveform(
            waveform,
            sample_rate=16_000,
            config=config,
            logmel_fn=fake_logmel_fn,
            max_duration_secs=0.5,
        )
    assert exc_info.value.status_code == 413


def test_torch_wav_to_logmel_forward_generates_finite_features_on_input_device():
    extractor = TorchWavToLogmel(
        samplerate=16_000,
        winsize=512,
        hopsize=128,
        n_mels=32,
        mel_fmin=50,
        mel_fmax=8_000,
    ).to("cpu")
    waveform = _sine_wave(sample_rate=16_000, duration_secs=0.5)

    with torch.no_grad():
        logmel = extractor(waveform)

    assert logmel.shape[0] == 32
    assert logmel.shape[1] > 0
    assert logmel.device == waveform.device
    assert torch.isfinite(logmel).all()


def test_torch_wav_to_logmel_uses_module_forward_dispatch():
    extractor = TorchWavToLogmel(
        samplerate=16_000,
        winsize=512,
        hopsize=128,
        n_mels=16,
        mel_fmin=50,
        mel_fmax=8_000,
    )
    waveform = _sine_wave(sample_rate=16_000, duration_secs=0.25)

    called = []

    def hook(module, inputs, output):
        called.append((module, inputs, output))

    handle = extractor.register_forward_hook(hook)
    try:
        logmel = extractor(waveform)
    finally:
        handle.remove()

    assert called
    assert called[0][0] is extractor
    assert called[0][2] is logmel


def test_preprocess_waveform_with_generated_stereo_waveform_resamples_and_extracts_logmel():
    input_sample_rate = 8_000
    config = TranscriptionConfig(
        device="cpu",
        target_sr=16_000,
        stft_winsize=512,
        stft_hopsize=128,
        melbins=24,
        mel_fmin=50,
        mel_fmax=8_000,
    )
    left = _sine_wave(input_sample_rate, duration_secs=0.25, frequency=220.0).squeeze(0)
    right = _sine_wave(input_sample_rate, duration_secs=0.25, frequency=440.0).squeeze(0)
    waveform = torch.stack([left, right])

    logmel = preprocess_waveform(waveform, input_sample_rate, config)

    expected_resampled_samples = int(0.25 * config.target_sr)
    expected_frames = 1 + expected_resampled_samples // config.stft_hopsize

    assert logmel.shape == (1, config.melbins, expected_frames)
    assert logmel.device.type == "cpu"
    assert torch.isfinite(logmel).all()


def test_normalize_pedal_prediction_shape():
    assert normalize_pedal_prediction_shape(torch.zeros(5)).shape == (1, 1, 5)
    assert normalize_pedal_prediction_shape(torch.zeros(1, 5)).shape == (1, 1, 5)
    assert normalize_pedal_prediction_shape(torch.zeros(2, 5), batch_size=2).shape == (2, 1, 5)
    assert normalize_pedal_prediction_shape(torch.zeros(3, 5), num_pedals=3, batch_size=1).shape == (1, 3, 5)
    assert normalize_pedal_prediction_shape(torch.zeros(2, 1, 5)).shape == (2, 1, 5)
    with pytest.raises(ValueError, match="divisible"):
        normalize_pedal_prediction_shape(torch.zeros(5), num_pedals=2)


def test_run_inference_and_decode_uses_shared_pipeline_with_fake_model():
    config = TranscriptionConfig(
        device="cpu",
        target_sr=100,
        stft_hopsize=10,
        melbins=4,
        inference_chunk_size_secs=1.0,
        inference_chunk_overlap_secs=0.0,
        decoder_gauss_std=None,
        note_threshold=0.9,
        pedal_threshold=0.5,
    )
    logmel = torch.randn(1, 4, 6)

    class FakeModel(torch.nn.Module):
        def forward(self, x):
            frames = x.shape[-1] - 1
            onsets = torch.full((1, 88, frames), -20.0)
            velocities = torch.zeros(1, 88, frames)
            pedals = torch.full((1, 3, frames), -20.0)
            onsets[0, 0, 2] = 20.0
            velocities[0, 0, 2] = 20.0
            pedals[0, 1, 1] = 20.0
            pedals[0, 2, 3] = 20.0
            return [onsets], velocities, pedals

    result = run_inference_and_decode(FakeModel(), logmel, config)

    assert list(result.notes["key"]) == [0]
    assert list(result.notes["t_idx"]) == [3]
    assert set(result.pedal_events["event_type"]) == {"onset", "offset"}
    assert list(result.pedal_events["t_idx"]) == [2, 4]
    assert result.logmel.shape == logmel.shape


def test_end_to_end_smoke_generated_wav_fake_model_json_schema(tmp_path):
    """Exercise WAV loading, preprocessing, inference/decoding, and JSON output.

    This intentionally uses a deterministic fake model instead of a real checkpoint
    so CI can validate the lightweight pipeline contract without shipping model
    weights.
    """
    sample_rate = 8_000
    samples = (_sine_wave(sample_rate, duration_secs=0.2, frequency=440.0) * 8_000).squeeze(0)
    wav_path = tmp_path / "short.wav"
    with wave.open(str(wav_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(samples.to(torch.int16).numpy().tobytes())

    config = TranscriptionConfig(
        device="cpu",
        target_sr=8_000,
        stft_winsize=256,
        stft_hopsize=80,
        melbins=16,
        mel_fmin=50,
        mel_fmax=4_000,
        inference_chunk_size_secs=1.0,
        inference_chunk_overlap_secs=0.0,
        decoder_gauss_std=None,
        note_threshold=0.9,
        pedal_threshold=0.5,
    )
    transcriber = PianoTranscriber(config)

    class TinyDeterministicModel(torch.nn.Module):
        def forward(self, x):
            frames = x.shape[-1] - 1
            onsets = torch.full((x.shape[0], 88, frames), -20.0, device=x.device)
            velocities = torch.zeros((x.shape[0], 88, frames), device=x.device)
            pedals = torch.full((x.shape[0], 3, frames), -20.0, device=x.device)

            note_frame = min(3, frames - 1)
            pedal_on = min(1, frames - 1)
            pedal_off = min(max(pedal_on + 3, 2), frames)
            onsets[0, 39, note_frame] = 20.0
            velocities[0, 39, note_frame] = 20.0
            pedals[0, 1, pedal_on] = 20.0
            pedals[0, 2, max(pedal_on, pedal_off - 1)] = 20.0
            return [onsets], velocities, pedals

    logmel = transcriber.preprocess_audio(wav_path)
    result = transcriber.run_inference_and_decode(TinyDeterministicModel(), logmel)

    json_path = tmp_path / "transcription.json"
    write_json_output(result, transcriber, json_path)
    payload = json.loads(json_path.read_text(encoding="utf-8"))

    assert payload == result_to_json_payload(result, transcriber)
    assert set(payload) == {"notes", "pedal_events", "duration", "secs_per_frame", "midi_key_offset"}
    assert isinstance(payload["notes"], list) and payload["notes"]
    assert {"batch_idx", "key", "t_idx", "prob", "vel"}.issubset(payload["notes"][0])
    assert payload["notes"][0]["key"] == 39
    assert isinstance(payload["pedal_events"], list) and payload["pedal_events"]
    assert {"batch_idx", "pedal_idx", "t_idx", "event_type"}.issubset(payload["pedal_events"][0])
    assert {event["event_type"] for event in payload["pedal_events"]} == {"onset", "offset"}
    assert isinstance(payload["duration"], float) and payload["duration"] > 0
    assert payload["secs_per_frame"] == pytest.approx(config.secs_per_frame)
    assert payload["midi_key_offset"] == 21