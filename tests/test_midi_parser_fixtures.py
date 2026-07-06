import math

import mido
import pytest

from ov_piano.data.key_model import KeyboardStateMachine
from ov_piano.data.midi import MidiToPianoRoll, SingletrackMidiParser


TPB = 480
TEMPO = 500_000


def _write_midi(tmp_path, messages, *, include_tempo=True, ticks_per_beat=TPB):
    midi_path = tmp_path / "fixture.mid"
    mid = mido.MidiFile(type=0, ticks_per_beat=ticks_per_beat)
    track = mido.MidiTrack()
    mid.tracks.append(track)
    if include_tempo:
        track.append(mido.MetaMessage("set_tempo", tempo=TEMPO, time=0))
    track.extend(messages)
    track.append(mido.MetaMessage("end_of_track", time=1))
    mid.save(midi_path)
    return midi_path


def _load_relevant_messages(midi_path):
    mid = SingletrackMidiParser.load_midi(midi_path)
    msgs, meta_msgs = SingletrackMidiParser.parse_midi(mid)
    MidiToPianoRoll._check_midi(msgs, meta_msgs)
    return msgs, meta_msgs


def _parse_key_events(
    midi_path,
    *,
    extend_offsets_sus=True,
    ignore_redundant_keypress=False,
    ignore_redundant_keylift=False,
):
    mid = SingletrackMidiParser.load_midi(midi_path)
    msgs, meta_msgs = SingletrackMidiParser.parse_midi(mid)
    MidiToPianoRoll._check_midi(msgs, meta_msgs)
    sus_t = MidiToPianoRoll.SUS_PEDAL_THRESH if extend_offsets_sus else math.inf
    return SingletrackMidiParser.ksm_parse_midi_messages(
        msgs,
        KeyboardStateMachine(
            sus_t,
            MidiToPianoRoll.TEN_PEDAL_THRESH,
            ignore_redundant_keypress=ignore_redundant_keypress,
            ignore_redundant_keylift=ignore_redundant_keylift,
        ),
    )


def test_sustain_pedal_extends_note_until_pedal_release(tmp_path):
    midi_path = _write_midi(
        tmp_path,
        [
            mido.Message("control_change", control=64, value=127, time=0),
            mido.Message("note_on", note=60, velocity=80, time=0),
            mido.Message("note_off", note=60, velocity=0, time=TPB),
            mido.Message("control_change", control=64, value=0, time=TPB),
        ],
    )

    key_events, sus_states, _, _, largest_ts = _parse_key_events(midi_path)

    assert list(sus_states["val"]) == [127, 0]
    assert key_events.iloc[0].to_dict() == {
        "onset": 0.0,
        "offset": 1.0,
        "key": 60.0,
        "vel": 80.0,
    }
    assert largest_ts == pytest.approx(1.0)


def test_repeated_note_on_can_be_warned_and_sanitized(tmp_path):
    midi_path = _write_midi(
        tmp_path,
        [
            mido.Message("note_on", note=60, velocity=80, time=0),
            mido.Message("note_on", note=60, velocity=96, time=TPB),
            mido.Message("note_off", note=60, velocity=0, time=TPB),
        ],
    )

    with pytest.warns(RuntimeWarning, match="Pressing a pressed key"):
        key_events, _, _, _, _ = _parse_key_events(
            midi_path,
            ignore_redundant_keypress=True,
        )

    assert list(key_events["key"]) == [60, 60]
    assert list(key_events["vel"]) == [80, 96]
    assert key_events.iloc[0]["offset"] == pytest.approx(0.499)
    assert key_events.iloc[1]["onset"] == pytest.approx(0.5)
    assert key_events.iloc[1]["offset"] == pytest.approx(1.0)


def test_unknown_control_change_is_warned_and_ignored(tmp_path):
    midi_path = _write_midi(
        tmp_path,
        [
            mido.Message("control_change", control=1, value=64, time=0),
            mido.Message("note_on", note=62, velocity=70, time=0),
            mido.Message("note_off", note=62, velocity=0, time=TPB),
        ],
    )

    mid = SingletrackMidiParser.load_midi(midi_path)
    with pytest.warns(RuntimeWarning, match="unsupported MIDI control_change control=1"):
        msgs, meta_msgs = SingletrackMidiParser.parse_midi(mid)

    assert [msg[1][0] for msg in msgs] == ["note_on", "note_off"]
    MidiToPianoRoll._check_midi(msgs, meta_msgs)


def test_missing_tempo_warns_and_uses_midi_default_tempo(tmp_path):
    midi_path = _write_midi(
        tmp_path,
        [
            mido.Message("note_on", note=64, velocity=90, time=0),
            mido.Message("note_off", note=64, velocity=0, time=TPB),
        ],
        include_tempo=False,
    )

    mid = SingletrackMidiParser.load_midi(midi_path)
    with pytest.warns(RuntimeWarning, match="no set_tempo"):
        msgs, meta_msgs = SingletrackMidiParser.parse_midi(mid)

    assert meta_msgs[0][1].type == "end_of_track"
    assert msgs[1][0] == pytest.approx(0.5)
    MidiToPianoRoll._check_midi(msgs, meta_msgs)


def test_offset_extension_uses_latest_message_when_sustain_remains_down(tmp_path):
    midi_path = _write_midi(
        tmp_path,
        [
            mido.Message("control_change", control=64, value=127, time=0),
            mido.Message("note_on", note=65, velocity=82, time=0),
            mido.Message("note_off", note=65, velocity=0, time=TPB),
            mido.Message("control_change", control=67, value=45, time=TPB),
        ],
    )

    key_events, sus_states, _, soft_states, largest_ts = _parse_key_events(midi_path)

    assert list(sus_states["val"]) == [127]
    assert list(soft_states["val"]) == [45]
    assert largest_ts == pytest.approx(1.0)
    assert key_events.iloc[0]["offset"] == pytest.approx(1.0)


def test_unsupported_meta_messages_are_warned_and_filtered(tmp_path):
    midi_path = _write_midi(
        tmp_path,
        [
            mido.MetaMessage("time_signature", numerator=3, denominator=4, time=0),
            mido.Message("note_on", note=67, velocity=88, time=0),
            mido.Message("note_off", note=67, velocity=0, time=TPB),
        ],
    )

    mid = SingletrackMidiParser.load_midi(midi_path)
    with pytest.warns(RuntimeWarning, match="unsupported MIDI meta-message 'time_signature'"):
        msgs, meta_msgs = SingletrackMidiParser.parse_midi(mid)

    assert [msg.type for _, msg in meta_msgs] == ["set_tempo", "end_of_track"]
    MidiToPianoRoll._check_midi(msgs, meta_msgs)


def test_metadata_only_midi_returns_empty_events_instead_of_crashing(tmp_path):
    midi_path = _write_midi(tmp_path, [])

    msgs, meta_msgs = _load_relevant_messages(midi_path)
    key_events, sus_states, ten_states, soft_states, largest_ts = (
        SingletrackMidiParser.ksm_parse_midi_messages(
            msgs,
            KeyboardStateMachine(
                MidiToPianoRoll.SUS_PEDAL_THRESH,
                MidiToPianoRoll.TEN_PEDAL_THRESH,
            ),
        )
    )

    assert [msg.type for _, msg in meta_msgs] == ["set_tempo", "end_of_track"]
    assert key_events.empty
    assert sus_states.empty
    assert ten_states.empty
    assert soft_states.empty
    assert largest_ts == 0
