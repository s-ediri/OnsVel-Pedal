"""Command-line transcription entry point."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ov_piano.transcription import (  # noqa: E402
    AudioPreprocessingError,
    PianoTranscriber,
    TranscriptionConfig,
    TranscriptionResult,
)


def existing_file_path(value: str) -> Path:
    """Return an absolute path for an existing file or raise an argparse error."""
    path = Path(value).expanduser().resolve()
    if not path.exists():
        raise argparse.ArgumentTypeError(f"file does not exist: {value}")
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"path is not a file: {value}")
    return path


def writable_output_path(value: str) -> Path:
    """Return an absolute output path whose parent directory already exists."""
    path = Path(value).expanduser().resolve()
    if path.exists() and path.is_dir():
        raise argparse.ArgumentTypeError(f"output path is a directory: {path}")
    parent = path.parent
    if not parent.exists():
        raise argparse.ArgumentTypeError(f"output directory does not exist: {parent}")
    if not parent.is_dir():
        raise argparse.ArgumentTypeError(f"output parent is not a directory: {parent}")
    return path


def positive_float(value: str) -> float:
    """Parse a finite, positive floating-point CLI value."""
    try:
        number = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"must be a number: {value}") from exc
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be a finite positive number")
    return number


def _dataframe_records(df):
    """Convert a pandas DataFrame to JSON-friendly row dictionaries."""
    records = df.reset_index(drop=True).to_dict(orient="records")
    return [{key: _json_scalar(value) for key, value in row.items()} for row in records]


def _json_scalar(value):
    """Convert numpy/pandas scalar values to builtin JSON scalar types."""
    if hasattr(value, "item"):
        try:
            value = value.item()
        except ValueError:
            pass
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def transcription_duration_secs(result: TranscriptionResult, transcriber: PianoTranscriber) -> float | None:
    """Return the transcription duration in seconds when log-mel frames are available."""
    if result.logmel is None:
        return None
    return float(result.logmel.shape[-1] * transcriber.secs_per_frame)


def result_to_json_payload(result: TranscriptionResult, transcriber: PianoTranscriber) -> dict:
    """Build a stable JSON-serializable payload from a transcription result."""
    return {
        "notes": _dataframe_records(result.notes),
        "pedal_events": _dataframe_records(result.pedal_events),
        "duration": transcription_duration_secs(result, transcriber),
        "secs_per_frame": float(transcriber.secs_per_frame),
        "midi_key_offset": int(transcriber.key_beg),
    }


def write_json_output(result: TranscriptionResult, transcriber: PianoTranscriber, output_path: Path) -> None:
    """Write decoded notes and pedal events to a JSON file."""
    output_path = Path(output_path)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result_to_json_payload(result, transcriber), f, indent=2, allow_nan=False)
        f.write("\n")


def _paired_pedal_intervals(pedal_events_df, secs_per_frame: float, fallback_end_secs: float | None = None):
    """Yield paired sustain-pedal intervals from onset/offset event rows."""
    if pedal_events_df.empty:
        return

    for pedal_idx, group in pedal_events_df.groupby("pedal_idx"):
        onsets = sorted(group[group["event_type"] == "onset"]["t_idx"].tolist())
        offsets = sorted(group[group["event_type"] == "offset"]["t_idx"].tolist())
        offset_cursor = 0
        for onset_frame in onsets:
            while offset_cursor < len(offsets) and offsets[offset_cursor] <= onset_frame:
                offset_cursor += 1
            if offset_cursor >= len(offsets):
                if fallback_end_secs is not None:
                    start = float(onset_frame * secs_per_frame)
                    if fallback_end_secs > start:
                        yield int(pedal_idx), start, fallback_end_secs
                break
            offset_frame = offsets[offset_cursor]
            offset_cursor += 1
            yield int(pedal_idx), float(onset_frame * secs_per_frame), float(offset_frame * secs_per_frame)


def _midi_velocity(value) -> int:
    """Convert a model velocity value to a valid MIDI note-on velocity."""
    try:
        velocity_float = float(value)
    except (TypeError, ValueError):
        return 64
    if not math.isfinite(velocity_float):
        return 64
    return max(1, min(127, round(velocity_float * 127)))


def write_midi_output(result: TranscriptionResult, transcriber: PianoTranscriber, output_path: Path) -> None:
    """Write decoded notes and sustain-pedal intervals to a simple MIDI file."""
    try:
        import mido
    except ImportError as exc:  # pragma: no cover - dependency is declared in project metadata
        raise RuntimeError("MIDI output requires the 'mido' package to be installed") from exc

    output_path = Path(output_path)
    ticks_per_beat = 480
    tempo = mido.bpm2tempo(120)
    mid = mido.MidiFile(ticks_per_beat=ticks_per_beat)
    track = mido.MidiTrack()
    mid.tracks.append(track)
    track.append(mido.MetaMessage("set_tempo", tempo=tempo, time=0))

    events = []
    default_note_duration = 0.4
    for _, row in result.notes.iterrows():
        start = float(row["t_idx"] * transcriber.secs_per_frame)
        end = start + default_note_duration
        note = int(row["key"] + transcriber.key_beg)
        velocity = _midi_velocity(row.get("vel", 1.0))
        events.append((start, 1, mido.Message("note_on", note=note, velocity=velocity, time=0)))
        events.append((end, 0, mido.Message("note_off", note=note, velocity=0, time=0)))

    for pedal_idx, start, end in _paired_pedal_intervals(
        result.pedal_events,
        transcriber.secs_per_frame,
        fallback_end_secs=transcription_duration_secs(result, transcriber),
    ):
        if pedal_idx != 0:
            continue
        events.append((start, 1, mido.Message("control_change", control=64, value=127, time=0)))
        events.append((end, 0, mido.Message("control_change", control=64, value=0, time=0)))

    previous_tick = 0
    for seconds, order, message in sorted(events, key=lambda item: (item[0], item[1])):
        tick = mido.second2tick(max(0.0, seconds), ticks_per_beat, tempo)
        message.time = max(0, int(round(tick - previous_tick)))
        track.append(message)
        previous_tick = tick
    track.append(mido.MetaMessage("end_of_track", time=0))
    mid.save(output_path)


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Transcribe a piano audio file with note and sustain-pedal events."
    )
    parser.add_argument("audio_path", type=existing_file_path, help="Path to the audio file.")
    parser.add_argument("model_path", type=existing_file_path, help="Path to the model checkpoint.")
    parser.add_argument("--device", type=str, default=None, help="PyTorch device to use, e.g. cpu or cuda.")
    parser.add_argument(
        "--json-out",
        type=writable_output_path,
        default=None,
        help="Optional path for JSON output containing decoded note and pedal event rows.",
    )
    parser.add_argument(
        "--midi-out",
        type=writable_output_path,
        default=None,
        help="Optional path for MIDI output with notes and sustain-pedal control changes.",
    )
    parser.add_argument(
        "--max-duration-secs",
        type=positive_float,
        default=None,
        help="Optional maximum accepted audio duration in seconds.",
    )
    return parser


def validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """Validate relationships between parsed CLI paths."""
    output_paths = [path for path in (args.json_out, args.midi_out) if path is not None]
    if len(output_paths) != len(set(output_paths)):
        parser.error("JSON and MIDI output paths must be different")

    protected_inputs = {args.audio_path, args.model_path}
    for output_path in output_paths:
        if output_path in protected_inputs:
            parser.error("output path must not overwrite the input audio or model checkpoint")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_args(args, parser)

    config_kwargs = {}
    if args.device is not None:
        config_kwargs["device"] = args.device
    transcriber = PianoTranscriber(TranscriptionConfig(**config_kwargs))

    try:
        result = transcriber.transcribe_file(
            str(args.audio_path),
            str(args.model_path),
            max_duration_secs=args.max_duration_secs,
            decode_with_pydub=True,
        )
    except AudioPreprocessingError as exc:
        parser.exit(status=2, message=f"Audio preprocessing failed: {exc}\n")

    print("--- Notes ---")
    print(result.notes)
    print("--- Pedals ---")
    print(result.pedal_events)

    if args.json_out is not None:
        try:
            write_json_output(result, transcriber, args.json_out)
        except (OSError, TypeError, ValueError) as exc:
            parser.exit(status=1, message=f"Failed to write JSON output: {exc}\n")
        print(f"Wrote JSON output to {args.json_out}")
    if args.midi_out is not None:
        try:
            write_midi_output(result, transcriber, args.midi_out)
        except (OSError, RuntimeError) as exc:
            parser.exit(status=1, message=f"Failed to write MIDI output: {exc}\n")
        print(f"Wrote MIDI output to {args.midi_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
