"""Unit tests for lyrics.py — no faster-whisper install required.

A fake `faster_whisper` module is injected into sys.modules so
transcribe_lyrics exercises its real logic (validation, caching, result
assembly) against a deterministic stand-in model.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import lyrics  # noqa: E402


class FakeWhisperModel:
    instances = 0

    def __init__(self, model_size, device="cpu", compute_type="int8"):
        FakeWhisperModel.instances += 1
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type

    def transcribe(self, path, language=None, vad_filter=True, beam_size=5):
        segments = [
            SimpleNamespace(start=0.0, end=2.51234, text="  Primeira linha  "),
            SimpleNamespace(start=2.6, end=5.0, text="Segunda linha"),
            SimpleNamespace(start=5.1, end=5.2, text="   "),  # blank, dropped
        ]
        info = SimpleNamespace(language=language or "pt", language_probability=0.93)
        return iter(segments), info


@pytest.fixture
def fake_whisper(monkeypatch, tmp_path):
    module = types.ModuleType("faster_whisper")
    module.WhisperModel = FakeWhisperModel
    monkeypatch.setitem(sys.modules, "faster_whisper", module)
    monkeypatch.setattr(lyrics, "_MODEL_CACHE", {})
    FakeWhisperModel.instances = 0

    audio = tmp_path / "song.wav"
    audio.write_bytes(b"RIFF0000WAVE")
    return audio


def test_happy_path(fake_whisper):
    result = lyrics.transcribe_lyrics(fake_whisper, model_size="small")
    assert result.text == "Primeira linha\nSegunda linha"
    assert result.language == "pt"
    assert result.language_probability == pytest.approx(0.93)
    assert result.model_size == "small"
    assert result.segments == [
        {"start": 0.0, "end": 2.512, "text": "Primeira linha"},
        {"start": 2.6, "end": 5.0, "text": "Segunda linha"},
    ]


def test_forced_language_propagates(fake_whisper):
    result = lyrics.transcribe_lyrics(fake_whisper, language="en")
    assert result.language == "en"


def test_model_cache_reuses_instance(fake_whisper):
    lyrics.transcribe_lyrics(fake_whisper, model_size="base")
    lyrics.transcribe_lyrics(fake_whisper, model_size="base")
    assert FakeWhisperModel.instances == 1
    lyrics.transcribe_lyrics(fake_whisper, model_size="tiny")
    assert FakeWhisperModel.instances == 2


def test_invalid_model_falls_back_to_small(fake_whisper):
    result = lyrics.transcribe_lyrics(fake_whisper, model_size="gigantic")
    assert result.model_size == "small"
    assert "small" in lyrics._MODEL_CACHE


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        lyrics.transcribe_lyrics(tmp_path / "missing.wav")


def test_missing_dependency_raises_runtime_error(monkeypatch, tmp_path):
    audio = tmp_path / "song.wav"
    audio.write_bytes(b"RIFF")
    # sys.modules[name] = None forces `import faster_whisper` to raise ImportError
    monkeypatch.setitem(sys.modules, "faster_whisper", None)
    monkeypatch.setattr(lyrics, "_MODEL_CACHE", {})
    with pytest.raises(RuntimeError, match="pip install faster-whisper"):
        lyrics.transcribe_lyrics(audio)
