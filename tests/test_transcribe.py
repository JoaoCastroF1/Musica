"""Unit tests for the post-processing and detection helpers in transcribe.py.

These don't invoke Basic Pitch (heavy ML dep). They exercise the pure-Python
note-event manipulation and music21 wrappers with synthetic data.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture(autouse=True)
def _skip_if_deps_missing():
    """Skip the whole module if Basic Pitch / TF isn't installed locally."""
    try:
        import transcribe  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"transcribe module not importable: {exc}", allow_module_level=False)


def test_postprocess_drops_low_confidence():
    from transcribe import _postprocess_note_events

    events = [
        (0.0, 0.5, 60, 120, None),  # conf 0.94
        (0.5, 1.0, 62, 30, None),  # conf 0.24, dropped
        (1.0, 1.5, 64, 80, None),  # conf 0.63
    ]
    out = _postprocess_note_events(events, min_confidence=0.4, merge_gap_seconds=0.05)
    assert [ev[2] for ev in out] == [60, 64]


def test_postprocess_merges_adjacent_same_pitch():
    from transcribe import _postprocess_note_events

    events = [
        (0.0, 0.4, 60, 90, None),
        (0.42, 0.8, 60, 90, None),  # tiny gap → should merge
        (1.5, 1.7, 60, 90, None),  # big gap → stays separate
    ]
    out = _postprocess_note_events(events, min_confidence=0.0, merge_gap_seconds=0.05)
    assert len(out) == 2
    assert out[0][0] == 0.0 and out[0][1] == 0.8
    assert out[1][0] == 1.5


def test_postprocess_preserves_order():
    from transcribe import _postprocess_note_events

    events = [
        (1.0, 1.5, 64, 100, None),
        (0.0, 0.5, 60, 100, None),
        (0.5, 1.0, 62, 100, None),
    ]
    out = _postprocess_note_events(events, min_confidence=0.0, merge_gap_seconds=0.0)
    starts = [ev[0] for ev in out]
    assert starts == sorted(starts)


def test_detect_meter_returns_supported_or_default():
    from transcribe import SUPPORTED_METERS, _detect_meter

    events = [(i * 0.5, i * 0.5 + 0.25, 60, 100, None) for i in range(16)]
    assert _detect_meter(events, bpm=120.0) in SUPPORTED_METERS
    assert _detect_meter([], bpm=120.0) == "4/4"
    assert _detect_meter(events, bpm=0) == "4/4"


def test_detect_meter_prefers_triple_for_waltz():
    from transcribe import _detect_meter

    bpm = 120.0
    spb = 60.0 / bpm
    events = []
    for measure in range(8):
        base = measure * 3 * spb
        for beat in range(3):
            events.append((base + beat * spb, base + (beat + 1) * spb, 60, 100, None))
    meter = _detect_meter(events, bpm=bpm)
    assert meter in ("3/4", "6/8")


def test_rebuild_midi_skips_zero_length_notes():
    from transcribe import _rebuild_midi

    events = [
        (0.0, 0.0, 60, 90, None),  # zero length, dropped
        (0.0, 0.5, 60, 90, None),
    ]
    midi = _rebuild_midi(events)
    assert len(midi.instruments) == 1
    assert len(midi.instruments[0].notes) == 1


def test_detect_tempo_handles_missing_file(tmp_path):
    from transcribe import _detect_tempo

    bpm = _detect_tempo(tmp_path / "nonexistent.wav")
    assert bpm == 120.0


def test_detect_key_falls_back_to_c_on_empty_score():
    from music21 import key as m21_key, stream

    from transcribe import _detect_key

    s = stream.Score()
    detected = _detect_key(s)
    assert isinstance(detected, m21_key.Key)
