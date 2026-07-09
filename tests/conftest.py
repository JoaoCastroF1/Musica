"""Test bootstrap: provide a lightweight stub for the heavy `transcribe` module.

The real transcribe.py pulls in Basic Pitch / TensorFlow / librosa, which are
too heavy for a test environment. The HTTP-layer tests only need the
TranscriptionResult container and a patchable transcribe_audio symbol, so when
the heavy deps are missing we register a minimal stand-in before app.py is
imported. Tests that exercise the real pipeline detect the stub via the
__musica_stub__ marker and skip themselves.
"""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import transcribe  # noqa: F401 — prefer the real module when deps exist
except Exception:
    stub = types.ModuleType("transcribe")
    stub.__musica_stub__ = True

    @dataclass
    class TranscriptionResult:
        job_id: str
        midi_path: Path
        musicxml_path: Path
        pdf_path: Optional[Path] = None
        notes: list = field(default_factory=list)
        tempo_bpm: float = 120.0
        duration_seconds: float = 0.0
        num_notes: int = 0
        key: str = "C major"
        time_signature: str = "4/4"
        num_notes_raw: int = 0

    def transcribe_audio(*_args, **_kwargs):
        raise RuntimeError("stub transcribe_audio called — patch it in tests")

    stub.TranscriptionResult = TranscriptionResult
    stub.transcribe_audio = transcribe_audio
    sys.modules["transcribe"] = stub
