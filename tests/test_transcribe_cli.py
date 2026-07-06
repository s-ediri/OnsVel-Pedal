import json

import mido
import pandas as pd
import pytest
import torch

from ov_piano.transcription import PianoTranscriber, TranscriptionConfig, TranscriptionResult
import scripts.transcribe as transcribe_cli


def _tiny_result():
    return TranscriptionResult(
        notes=pd.DataFrame(
            [
                {"batch_idx": 0, "key": 39, "t_idx": 2, "prob": 0.99, "vel": 0.75},
            ]
        ),
        pedal_events=pd.DataFrame(
            [
                {"batch_idx": 0, "pedal_idx": 0, "t_idx": 1, "event_type": "onset"},
                {"batch_idx": 0, "pedal_idx": 0, "t_idx": 4, "event_type": "offset"},
            ]
        ),
        logmel=torch.zeros(1, 4, 8),
    )


def test_transcribe_cli_smoke_json_and_midi_outputs(tmp_path):
    transcriber = PianoTranscriber(
        TranscriptionConfig(
            device="cpu",
            stft_winsize=512,
            stft_hopsize=160,
            melbins=16,
            decoder_gauss_std=None,
        )
    )
    result = _tiny_result()
    result.notes.loc[0, "prob"] = float("nan")
    json_path = tmp_path / "transcription.json"
    midi_path = tmp_path / "transcription.mid"

    payload = transcribe_cli.result_to_json_payload(result, transcriber)
    transcribe_cli.write_json_output(result, transcriber, json_path)
    transcribe_cli.write_midi_output(result, transcriber, midi_path)

    assert payload["notes"][0]["key"] == 39
    assert payload["notes"][0]["prob"] is None
    assert payload["midi_key_offset"] == 21
    assert json.loads(json_path.read_text(encoding="utf-8"))["pedal_events"][0]["event_type"] == "onset"

    midi = mido.MidiFile(midi_path)
    messages = [message for track in midi.tracks for message in track]
    assert any(message.type == "note_on" and message.note == 60 for message in messages)
    assert any(
        message.type == "control_change" and message.control == 64 and message.value == 127
        for message in messages
    )


def test_transcribe_cli_validates_existing_input_paths(tmp_path):
    existing_audio = tmp_path / "audio.wav"
    existing_model = tmp_path / "model.torch"
    existing_audio.write_bytes(b"placeholder")
    existing_model.write_bytes(b"placeholder")

    args = transcribe_cli.build_parser().parse_args([str(existing_audio), str(existing_model)])

    assert args.audio_path == existing_audio.resolve()
    assert args.model_path == existing_model.resolve()


def test_transcribe_cli_rejects_unsafe_output_paths(tmp_path):
    existing_audio = tmp_path / "audio.wav"
    existing_model = tmp_path / "model.torch"
    existing_audio.write_bytes(b"placeholder")
    existing_model.write_bytes(b"placeholder")
    parser = transcribe_cli.build_parser()

    args = parser.parse_args([str(existing_audio), str(existing_model), "--json-out", str(existing_audio)])

    with pytest.raises(SystemExit):
        transcribe_cli.validate_args(args, parser)

    duplicate_output = tmp_path / "same_output.mid"
    args = parser.parse_args(
        [
            str(existing_audio),
            str(existing_model),
            "--json-out",
            str(duplicate_output),
            "--midi-out",
            str(duplicate_output),
        ]
    )

    with pytest.raises(SystemExit):
        transcribe_cli.validate_args(args, parser)


def test_transcribe_cli_rejects_non_positive_max_duration(tmp_path):
    existing_audio = tmp_path / "audio.wav"
    existing_model = tmp_path / "model.torch"
    existing_audio.write_bytes(b"placeholder")
    existing_model.write_bytes(b"placeholder")

    with pytest.raises(SystemExit):
        transcribe_cli.build_parser().parse_args(
            [str(existing_audio), str(existing_model), "--max-duration-secs", "0"]
        )


def test_transcribe_cli_main_uses_shared_pydub_loader(monkeypatch, tmp_path):
    existing_audio = tmp_path / "audio.mp3"
    existing_model = tmp_path / "model.torch"
    existing_audio.write_bytes(b"placeholder")
    existing_model.write_bytes(b"placeholder")
    calls = {}

    class FakeTranscriber:
        key_beg = 21
        secs_per_frame = 0.01

        def __init__(self, config):
            calls["config"] = config

        def transcribe_file(self, audio_source, snapshot_path, max_duration_secs=None, decode_with_pydub=False):
            calls["audio_source"] = audio_source
            calls["snapshot_path"] = snapshot_path
            calls["max_duration_secs"] = max_duration_secs
            calls["decode_with_pydub"] = decode_with_pydub
            return _tiny_result()

    monkeypatch.setattr(transcribe_cli, "PianoTranscriber", FakeTranscriber)
    monkeypatch.setattr(transcribe_cli, "TranscriptionConfig", lambda **kwargs: kwargs)

    exit_code = transcribe_cli.main(
        [str(existing_audio), str(existing_model), "--device", "cpu", "--max-duration-secs", "1.5"]
    )

    assert exit_code == 0
    assert calls["config"] == {"device": "cpu"}
    assert calls["audio_source"] == str(existing_audio.resolve())
    assert calls["snapshot_path"] == str(existing_model.resolve())
    assert calls["max_duration_secs"] == 1.5
    assert calls["decode_with_pydub"] is True