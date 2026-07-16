from collections import OrderedDict
from io import BytesIO
import os
import wave

import numpy as np
import pandas as pd
import pytest
import torch

from ov_piano.transcription import AudioPreprocessingError, TranscriptionConfig, TranscriptionResult
from web_app import app as web_app


@pytest.fixture(autouse=True)
def clear_model_cache(tmp_path, monkeypatch):
    web_app._model_cache.clear()
    web_app._grand_piano_sample_cache.clear()
    web_app._missing_grand_piano_sample_warnings.clear()
    monkeypatch.setattr(web_app.CONF, "GENERATED_AUDIO_DIR", str(tmp_path / "generated_audio"))
    grand_piano_sample = torch.linspace(0, 0.8, steps=4096).numpy().astype("float32")
    monkeypatch.setattr(
        web_app,
        "_load_nearest_grand_piano_sample",
        lambda pitch, sample_rate: (60, grand_piano_sample),
    )
    yield
    web_app._model_cache.clear()
    web_app._grand_piano_sample_cache.clear()
    web_app._missing_grand_piano_sample_warnings.clear()


@pytest.fixture
def client():
    web_app.app.config.update(TESTING=True)
    return web_app.app.test_client()


def _upload(audio_bytes=b"fake wav bytes", model="model.torch"):
    return {
        "audio": (BytesIO(audio_bytes), "audio.wav"),
        "model": model,
    }


def _wav_bytes(samples, sample_rate=8_000):
    samples = np.asarray(samples, dtype=np.float32).reshape(-1)
    pcm = (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2")
    buffer = BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm.tobytes())
    buffer.seek(0)
    return buffer.getvalue()


def _wav_response_samples(response_data):
    with wave.open(BytesIO(response_data), "rb") as wav_file:
        frames = wav_file.readframes(wav_file.getnframes())
    return np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0


def _tiny_transcription_result():
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


def test_api_models_returns_available_checkpoint_names(client, tmp_path, monkeypatch):
    first_model = tmp_path / "first.torch"
    second_model = tmp_path / "second.torch"
    first_model.write_bytes(b"checkpoint")
    second_model.write_bytes(b"checkpoint")
    monkeypatch.setattr(
        web_app,
        "_available_checkpoints",
        lambda: OrderedDict(
            [
                ("first.torch", str(first_model)),
                ("second.torch", str(second_model)),
            ]
        ),
    )

    response = client.get("/api/models")

    assert response.status_code == 200
    assert response.get_json() == ["first.torch", "second.torch"]


def test_index_shows_piano_roll_without_sheet_music_preview(client):
    response = client.get("/")

    html = response.get_data(as_text=True)
    assert response.status_code == 200
    assert "piano-roll-canvas" in html
    assert "Piano roll" in html
    assert "Pedal onset" in html
    assert "Sustain held" in html
    assert "notation-preview-container" not in html
    assert "Approximate pitch preview" not in html
    assert "abcjs" not in html.lower()
    assert "tone.js" not in html.lower()


def test_api_transcribe_requires_audio_file(client):
    response = client.post("/api/transcribe", data={"model": "model.torch"})

    assert response.status_code == 400
    assert response.get_json()["error"] == "No audio file in request"


def test_api_transcribe_requires_model_selection(client):
    response = client.post(
        "/api/transcribe",
        data={"audio": (BytesIO(b"fake wav bytes"), "audio.wav")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 400
    assert response.get_json()["error"] == "No model selected"


def test_api_transcribe_rejects_nonexistent_model(client, monkeypatch):
    monkeypatch.setattr(web_app, "_available_checkpoints", lambda: OrderedDict())

    response = client.post(
        "/api/transcribe",
        data=_upload(model="missing.torch"),
        content_type="multipart/form-data",
    )

    assert response.status_code == 404
    assert response.get_json()["error"] == "Model checkpoint not found or not allowed"


@pytest.mark.parametrize(
    "snapshot_name",
    [
        "../model.torch",
        "..\\model.torch",
        "subdir/model.torch",
        "subdir\\model.torch",
        "/tmp/model.torch",
        "",
        ".",
        "..",
    ],
)
def test_resolve_model_path_rejects_path_like_names(snapshot_name, tmp_path, monkeypatch):
    model_path = tmp_path / "model.torch"
    model_path.write_bytes(b"checkpoint")
    monkeypatch.setattr(
        web_app,
        "_available_checkpoints",
        lambda: OrderedDict([("model.torch", str(model_path))]),
    )

    assert web_app._resolve_model_path(snapshot_name) is None


def test_resolve_model_path_allows_only_server_listed_checkpoint_names(tmp_path, monkeypatch):
    model_path = tmp_path / "model.torch"
    model_path.write_bytes(b"checkpoint")
    monkeypatch.setattr(
        web_app,
        "_available_checkpoints",
        lambda: OrderedDict([("model.torch", str(model_path))]),
    )

    assert web_app._resolve_model_path("model.torch") == str(model_path)
    assert web_app._resolve_model_path("other.torch") is None


def test_available_checkpoints_excludes_symlink_escape(tmp_path, monkeypatch):
    checkpoint_dir = tmp_path / "checkpoints"
    assets_dir = tmp_path / "assets"
    outside_dir = tmp_path / "outside"
    checkpoint_dir.mkdir()
    assets_dir.mkdir()
    outside_dir.mkdir()

    inside_model = checkpoint_dir / "inside.torch"
    outside_model = outside_dir / "escape-target.torch"
    escape_link = checkpoint_dir / "escape.torch"
    inside_model.write_bytes(b"trusted checkpoint")
    outside_model.write_bytes(b"outside checkpoint")
    try:
        escape_link.symlink_to(outside_model)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"Symlinks are not available in this environment: {exc}")

    monkeypatch.setattr(web_app.CONF, "MODEL_SNAPSHOTS_DIR", str(checkpoint_dir))
    monkeypatch.setattr(web_app.CONF, "STATIC_ASSETS_DIR", str(assets_dir))

    checkpoints = web_app._available_checkpoints()

    assert checkpoints == OrderedDict([("inside.torch", str(inside_model.resolve()))])


def test_api_transcribe_rejects_uploaded_model_checkpoint_by_default(client, monkeypatch):
    monkeypatch.setattr(web_app.CONF, "ALLOW_MODEL_UPLOADS", False)

    response = client.post(
        "/api/transcribe",
        data={
            "audio": (BytesIO(b"fake wav bytes"), "audio.wav"),
            "model_file": (BytesIO(b"checkpoint"), "uploaded.torch"),
        },
        content_type="multipart/form-data",
    )

    assert response.status_code == 403
    assert "Uploaded model checkpoints are disabled" in response.get_json()["error"]


def test_api_transcribe_rejects_invalid_audio(client, tmp_path, monkeypatch):
    model_path = tmp_path / "model.torch"
    model_path.write_bytes(b"checkpoint")
    monkeypatch.setattr(
        web_app,
        "_available_checkpoints",
        lambda: OrderedDict([("model.torch", str(model_path))]),
    )

    def raise_invalid_audio(*_args, **_kwargs):
        raise AudioPreprocessingError("Could not decode audio", status_code=400)

    monkeypatch.setattr(web_app.transcriber, "preprocess_audio", raise_invalid_audio)

    response = client.post(
        "/api/transcribe",
        data=_upload(audio_bytes=b"not real audio"),
        content_type="multipart/form-data",
    )

    assert response.status_code == 400
    assert response.get_json()["error"] == "Could not decode audio"


def test_api_transcribe_accepts_audio_larger_than_previous_size_cap(client, tmp_path, monkeypatch):
    model_path = tmp_path / "model.torch"
    model_path.write_bytes(b"checkpoint")
    model = object()
    logmel = torch.zeros(1, 4, 8)
    monkeypatch.setattr(
        web_app,
        "_available_checkpoints",
        lambda: OrderedDict([("model.torch", str(model_path))]),
    )
    monkeypatch.setattr(web_app.transcriber, "preprocess_audio", lambda *_args, **_kwargs: logmel)
    monkeypatch.setattr(web_app.transcriber, "load_model", lambda snapshot_path: model)
    monkeypatch.setattr(
        web_app.transcriber,
        "run_inference_and_decode",
        lambda loaded_model, processed_logmel: _tiny_transcription_result(),
    )

    response = client.post(
        "/api/transcribe",
        data=_upload(audio_bytes=b"0" * ((25 * 1024 * 1024) + 1)),
        content_type="multipart/form-data",
    )

    assert response.status_code == 200


def test_process_audio_does_not_cap_duration(monkeypatch):
    calls = {}

    def fake_preprocess_audio(audio_file, **kwargs):
        calls["audio_file"] = audio_file
        calls.update(kwargs)
        return torch.zeros(1, 4, 8)

    audio_file = BytesIO(b"fake wav bytes")
    monkeypatch.setattr(web_app.transcriber, "preprocess_audio", fake_preprocess_audio)

    web_app._process_audio(audio_file)

    assert calls["audio_file"] is audio_file
    assert calls["max_duration_secs"] is None
    assert calls["decode_with_pydub"] is True


def test_api_transcribe_success_with_mocked_inference(client, tmp_path, monkeypatch):
    model_path = tmp_path / "model.torch"
    model_path.write_bytes(b"checkpoint")
    model = object()
    logmel = torch.zeros(1, 4, 8)

    monkeypatch.setattr(
        web_app,
        "_available_checkpoints",
        lambda: OrderedDict([("model.torch", str(model_path))]),
    )
    monkeypatch.setattr(web_app.transcriber, "preprocess_audio", lambda *_args, **_kwargs: logmel)
    monkeypatch.setattr(web_app.transcriber, "load_model", lambda snapshot_path: model)
    monkeypatch.setattr(
        web_app.transcriber,
        "run_inference_and_decode",
        lambda loaded_model, processed_logmel: _tiny_transcription_result(),
    )

    response = client.post(
        "/api/transcribe",
        data=_upload(),
        content_type="multipart/form-data",
    )

    payload = response.get_json()
    assert response.status_code == 200
    assert payload["notes"] == [
        {
            "pitch": 60,
            "start": pytest.approx(0.048),
            "velocity": 0.75,
            "duration": 1.0,
            "duration_estimated": True,
        }
    ]
    assert payload["pedals"] == [
        {
            "pedal_idx": 0,
            "start": pytest.approx(0.024),
            "end": pytest.approx(0.096),
            "duration": pytest.approx(0.072),
            "offset_estimated": False,
        }
    ]
    assert payload["duration"] == pytest.approx(0.192)
    assert payload["generated_audio"]["url"].startswith("/api/generated-audio/piano_")
    assert payload["generated_audio"]["url"].endswith(".wav")
    assert payload["generated_audio"]["sample_rate"] == web_app.CONF.GENERATED_AUDIO_SAMPLE_RATE
    assert payload["generated_audio"]["engine"] == "server-sampled-grand-piano-salamander-v1"
    assert payload["generated_audio"]["latency_seconds"] == pytest.approx(0.0)
    assert payload["generated_audio"]["duration"] >= payload["duration"]
    assert payload["generated_audio"]["balance"]["applied"] is False

    audio_response = client.get(payload["generated_audio"]["url"])
    assert audio_response.status_code == 200
    assert audio_response.mimetype == "audio/wav"
    with wave.open(BytesIO(audio_response.data), "rb") as wav_file:
        assert wav_file.getnchannels() == 1
        assert wav_file.getsampwidth() == 2
        assert wav_file.getframerate() == web_app.CONF.GENERATED_AUDIO_SAMPLE_RATE
        assert wav_file.getnframes() > 0


def test_api_transcribe_balances_generated_audio_to_uploaded_wav(client, tmp_path, monkeypatch):
    model_path = tmp_path / "model.torch"
    model_path.write_bytes(b"checkpoint")
    model = object()
    logmel = torch.zeros(1, 4, 8)
    target_rms = 0.04

    monkeypatch.setattr(
        web_app,
        "_available_checkpoints",
        lambda: OrderedDict([("model.torch", str(model_path))]),
    )
    monkeypatch.setattr(web_app.transcriber, "preprocess_audio", lambda *_args, **_kwargs: logmel)
    monkeypatch.setattr(web_app.transcriber, "load_model", lambda snapshot_path: model)
    monkeypatch.setattr(
        web_app.transcriber,
        "run_inference_and_decode",
        lambda loaded_model, processed_logmel: _tiny_transcription_result(),
    )

    response = client.post(
        "/api/transcribe",
        data=_upload(audio_bytes=_wav_bytes(np.full(8_000, target_rms, dtype=np.float32))),
        content_type="multipart/form-data",
    )

    payload = response.get_json()
    assert response.status_code == 200
    balance = payload["generated_audio"]["balance"]
    assert balance["applied"] is True
    assert balance["reference_active_rms"] == pytest.approx(target_rms, rel=0.02)
    assert balance["rendered_active_rms_after"] == pytest.approx(target_rms, rel=0.08)
    assert balance["gain"] < 1.0

    audio_response = client.get(payload["generated_audio"]["url"])
    assert audio_response.status_code == 200
    generated_samples = _wav_response_samples(audio_response.data)
    assert web_app._audio_level_stats(generated_samples)["active_rms"] == pytest.approx(target_rms, rel=0.08)


def test_generate_piano_audio_artifact_writes_zero_latency_wav(tmp_path, monkeypatch):
    monkeypatch.setattr(web_app.CONF, "GENERATED_AUDIO_DIR", str(tmp_path))
    monkeypatch.setattr(web_app.CONF, "GENERATED_AUDIO_SAMPLE_RATE", 8_000)
    notes = [
        {
            "pitch": 60,
            "start": 0.125,
            "duration": 0.25,
            "velocity": 0.8,
        }
    ]
    pedals = [
        {
            "pedal_idx": 0,
            "start": 0.2,
            "end": 0.55,
            "duration": 0.35,
            "offset_estimated": False,
        }
    ]

    generated_audio = web_app._generate_piano_audio_artifact(notes, pedals, duration_secs=0.6)

    assert generated_audio["url"].startswith("/api/generated-audio/piano_")
    assert generated_audio["sample_rate"] == 8_000
    assert generated_audio["engine"] == "server-sampled-grand-piano-salamander-v1"
    assert generated_audio["latency_seconds"] == pytest.approx(0.0)
    filename = generated_audio["url"].rsplit("/", 1)[-1]
    wav_path = tmp_path / filename
    assert wav_path.is_file()
    with wave.open(str(wav_path), "rb") as wav_file:
        assert wav_file.getnchannels() == 1
        assert wav_file.getsampwidth() == 2
        assert wav_file.getframerate() == 8_000
        assert wav_file.getnframes() == pytest.approx(generated_audio["duration"] * 8_000, abs=1)
        frames = wav_file.readframes(wav_file.getnframes())
    assert any(byte != 0 for byte in frames)


def test_load_grand_piano_sample_uses_synthetic_fallback_when_sample_missing(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(web_app.CONF, "GRAND_PIANO_SAMPLE_DIR", str(tmp_path / "missing_samples"))
    monkeypatch.setattr(web_app.CONF, "GRAND_PIANO_SAMPLE_BASE_URL", "")
    monkeypatch.setattr(web_app.CONF, "GENERATED_AUDIO_SYNTHETIC_FALLBACK", True)
    web_app._grand_piano_sample_cache.clear()
    web_app._missing_grand_piano_sample_warnings.clear()
    web_app._generated_audio_render_state.synthetic_fallback_used = False
    caplog.set_level("WARNING", logger="web_app.app")

    sample = web_app._load_grand_piano_sample(96, 8_000)

    assert sample.dtype == np.float32
    assert sample.size > 8_000
    assert float(np.max(np.abs(sample))) == pytest.approx(1.0)
    assert web_app._generated_audio_engine_name() == "server-synthetic-piano-fallback-v1"
    assert "Grand piano sample C7.mp3 could not be loaded" in caplog.text


def test_load_grand_piano_sample_raises_when_synthetic_fallback_disabled(tmp_path, monkeypatch):
    monkeypatch.setattr(web_app.CONF, "GRAND_PIANO_SAMPLE_DIR", str(tmp_path / "missing_samples"))
    monkeypatch.setattr(web_app.CONF, "GRAND_PIANO_SAMPLE_BASE_URL", "")
    monkeypatch.setattr(web_app.CONF, "GENERATED_AUDIO_SYNTHETIC_FALLBACK", False)
    web_app._grand_piano_sample_cache.clear()

    with pytest.raises(RuntimeError, match="Grand piano sample C7.mp3 is missing"):
        web_app._load_grand_piano_sample(96, 8_000)


def test_analyze_reference_audio_balance_measures_wav_and_rewinds():
    target_rms = 0.2
    source = BytesIO(_wav_bytes(np.full(4_000, target_rms, dtype=np.float32)))

    balance = web_app._analyze_reference_audio_balance(source)

    assert balance["usable"] is True
    assert balance["sample_rate"] == 8_000
    assert balance["active_rms"] == pytest.approx(target_rms, rel=0.02)
    assert source.tell() == 0


def test_balance_generated_audio_to_reference_matches_active_rms():
    generated = np.full(1_000, 0.1, dtype=np.float32)
    reference = {"usable": True, "active_rms": 0.25}

    balanced, balance = web_app._balance_generated_audio_to_reference(generated, reference)

    assert balance["applied"] is True
    assert balance["gain"] == pytest.approx(2.5)
    assert web_app._audio_level_stats(balanced)["active_rms"] == pytest.approx(0.25)


def test_format_results_extends_unclosed_pedal_to_transcription_end():
    result = TranscriptionResult(
        notes=pd.DataFrame(columns=["batch_idx", "key", "t_idx", "prob", "vel"]),
        pedal_events=pd.DataFrame(
            [
                {"batch_idx": 0, "pedal_idx": 0, "t_idx": 2, "event_type": "onset"},
            ]
        ),
        logmel=torch.zeros(1, 4, 10),
    )

    with web_app.app.app_context():
        response = web_app._format_results(result.notes, result.pedal_events, result.logmel)

    payload = response.get_json()
    assert payload["duration"] == pytest.approx(0.24)
    assert payload["pedals"] == [
        {
            "pedal_idx": 0,
            "start": pytest.approx(0.048),
            "end": pytest.approx(0.24),
            "duration": pytest.approx(0.192),
            "offset_estimated": True,
        }
    ]


def test_get_cached_model_reuses_same_path_mtime_and_device(tmp_path, monkeypatch):
    model_path = tmp_path / "model.torch"
    model_path.write_bytes(b"checkpoint")
    loaded_models = []

    def fake_load_model(snapshot_path):
        model = object()
        loaded_models.append((snapshot_path, model))
        return model

    monkeypatch.setattr(web_app.transcriber, "load_model", fake_load_model)

    first_model = web_app._get_cached_model(str(model_path))
    second_model = web_app._get_cached_model(str(model_path))

    assert first_model is second_model
    assert len(loaded_models) == 1
    assert loaded_models[0][0] == str(model_path.resolve())


def test_get_cached_model_reloads_when_checkpoint_mtime_changes(tmp_path, monkeypatch):
    model_path = tmp_path / "model.torch"
    model_path.write_bytes(b"checkpoint-v1")
    loaded_models = []

    def fake_load_model(_snapshot_path):
        model = object()
        loaded_models.append(model)
        return model

    monkeypatch.setattr(web_app.transcriber, "load_model", fake_load_model)

    first_model = web_app._get_cached_model(str(model_path))
    old_cache_keys = list(web_app._model_cache.keys())

    old_mtime_ns = model_path.stat().st_mtime_ns
    model_path.write_bytes(b"checkpoint-v2")
    new_mtime_ns = old_mtime_ns + 1_000_000_000
    os.utime(model_path, ns=(new_mtime_ns, new_mtime_ns))

    second_model = web_app._get_cached_model(str(model_path))

    assert second_model is not first_model
    assert loaded_models == [first_model, second_model]
    assert len(web_app._model_cache) == 1
    assert list(web_app._model_cache.keys()) != old_cache_keys
    assert list(web_app._model_cache.values()) == [second_model]


def test_get_cached_model_keeps_device_specific_entries(tmp_path, monkeypatch):
    model_path = tmp_path / "model.torch"
    model_path.write_bytes(b"checkpoint")
    loaded_models = []

    def fake_load_model(_snapshot_path):
        model = object()
        loaded_models.append(model)
        return model

    monkeypatch.setattr(web_app.transcriber, "load_model", fake_load_model)

    monkeypatch.setattr(web_app.transcriber, "config", TranscriptionConfig(device="cpu"))
    cpu_model = web_app._get_cached_model(str(model_path))

    monkeypatch.setattr(web_app.transcriber, "config", TranscriptionConfig(device="cuda"))
    cuda_model = web_app._get_cached_model(str(model_path))
    cuda_model_again = web_app._get_cached_model(str(model_path))

    assert cuda_model is cuda_model_again
    assert cpu_model is not cuda_model
    assert loaded_models == [cpu_model, cuda_model]
    assert {key[2] for key in web_app._model_cache.keys()} == {"cpu", "cuda"}


def test_get_cached_model_enforces_lru_size_limit(tmp_path, monkeypatch):
    model_paths = []
    for idx in range(3):
        model_path = tmp_path / f"model-{idx}.torch"
        model_path.write_bytes(f"checkpoint-{idx}".encode())
        model_paths.append(model_path)
    loaded_models = []

    def fake_load_model(_snapshot_path):
        model = object()
        loaded_models.append(model)
        return model

    monkeypatch.setattr(web_app.transcriber, "load_model", fake_load_model)
    monkeypatch.setattr(web_app.CONF, "MODEL_CACHE_SIZE", 2)

    first_model = web_app._get_cached_model(str(model_paths[0]))
    second_model = web_app._get_cached_model(str(model_paths[1]))
    # Touch the first entry to make the second entry least-recently-used.
    assert web_app._get_cached_model(str(model_paths[0])) is first_model
    third_model = web_app._get_cached_model(str(model_paths[2]))

    assert len(web_app._model_cache) == 2
    assert list(web_app._model_cache.values()) == [first_model, third_model]
    assert second_model not in web_app._model_cache.values()


def test_get_cached_model_allows_zero_cache_size(tmp_path, monkeypatch):
    model_path = tmp_path / "model.torch"
    model_path.write_bytes(b"checkpoint")
    loaded_models = []

    def fake_load_model(_snapshot_path):
        model = object()
        loaded_models.append(model)
        return model

    monkeypatch.setattr(web_app.transcriber, "load_model", fake_load_model)
    monkeypatch.setattr(web_app.CONF, "MODEL_CACHE_SIZE", 0)

    first_model = web_app._get_cached_model(str(model_path))
    second_model = web_app._get_cached_model(str(model_path))

    assert first_model is not second_model
    assert loaded_models == [first_model, second_model]
    assert web_app._model_cache == OrderedDict()