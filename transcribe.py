"""Audio-to-sheet-music transcription pipeline.

Uses Spotify's Basic Pitch (a small CNN trained on multi-instrument data) to
predict MIDI notes from audio, then converts the MIDI into a music21 Score and
exports MusicXML / MIDI / Lilypond-rendered PDF.

Basic Pitch is currently one of the strongest open-source polyphonic
transcription models. It works on monophonic and polyphonic input from any
pitched instrument.
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pretty_midi
from basic_pitch import ICASSP_2022_MODEL_PATH
from basic_pitch.inference import predict
from music21 import converter, environment, instrument, metadata, stream, tempo

logger = logging.getLogger(__name__)


@dataclass
class TranscriptionResult:
    """Paths to the generated artifacts for a single transcription job."""

    job_id: str
    midi_path: Path
    musicxml_path: Path
    pdf_path: Optional[Path] = None
    notes: list[dict] = field(default_factory=list)
    tempo_bpm: float = 120.0
    duration_seconds: float = 0.0
    num_notes: int = 0


def _estimate_tempo(midi: pretty_midi.PrettyMIDI) -> float:
    """Estimate tempo using pretty_midi's beat tracker, with a sane fallback."""
    try:
        tempi = midi.get_tempo_changes()[1]
        if len(tempi) > 0:
            return float(np.median(tempi))
    except Exception:  # noqa: BLE001 — pretty_midi can raise opaque errors
        pass
    try:
        return float(midi.estimate_tempo())
    except Exception:  # noqa: BLE001
        return 120.0


def _midi_to_score(midi_path: Path, title: str) -> stream.Score:
    """Parse a MIDI file into a music21 Score with sensible defaults.

    music21's MIDI parser handles tempo, time-signature inference, and
    quantization to the nearest sensible note value via .makeNotation().
    """
    score = converter.parse(str(midi_path), quantizePost=True)

    score.metadata = metadata.Metadata()
    score.metadata.title = title
    score.metadata.composer = "Transcribed by Musica"

    for part in score.parts:
        if not part.getInstruments(returnDefault=False):
            part.insert(0, instrument.Piano())

    return score


def _render_pdf(score: stream.Score, out_pdf: Path) -> Optional[Path]:
    """Try to render the score to PDF via Lilypond. Returns None if unavailable."""
    if not shutil.which("lilypond"):
        logger.warning("lilypond not installed — skipping PDF rendering")
        return None

    try:
        env = environment.Environment()
        env["lilypondPath"] = shutil.which("lilypond")
        rendered = score.write("lily.pdf", fp=str(out_pdf))
        rendered_path = Path(rendered)
        if rendered_path.exists():
            if rendered_path != out_pdf:
                shutil.move(str(rendered_path), str(out_pdf))
            return out_pdf
    except Exception as exc:  # noqa: BLE001 — Lilypond errors are varied
        logger.warning("Lilypond render failed: %s", exc)
    return None


def transcribe_audio(
    audio_path: str | Path,
    output_dir: str | Path,
    job_id: str,
    title: str = "Transcribed Score",
    onset_threshold: float = 0.5,
    frame_threshold: float = 0.3,
    minimum_note_length_ms: float = 58.0,
    minimum_frequency: Optional[float] = None,
    maximum_frequency: Optional[float] = None,
) -> TranscriptionResult:
    """Run the full audio → MIDI → MusicXML → PDF pipeline.

    Parameters
    ----------
    audio_path:
        Path to a readable audio file (wav/mp3/flac/ogg/m4a).
    output_dir:
        Directory where artifacts will be written. Created if missing.
    job_id:
        Stable identifier used as the artifact filename stem.
    title:
        Score title embedded in the MusicXML metadata.
    onset_threshold, frame_threshold, minimum_note_length_ms:
        Basic Pitch's note-event detection knobs. Tighter onset/frame
        thresholds reduce spurious notes at the cost of recall.
    minimum_frequency, maximum_frequency:
        Optional pitch band to restrict predictions to (Hz).
    """
    audio_path = Path(audio_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not audio_path.exists():
        raise FileNotFoundError(f"audio file not found: {audio_path}")

    logger.info("Transcribing %s (job=%s)", audio_path.name, job_id)

    model_output, midi_data, note_events = predict(
        str(audio_path),
        model_or_model_path=ICASSP_2022_MODEL_PATH,
        onset_threshold=onset_threshold,
        frame_threshold=frame_threshold,
        minimum_note_length=minimum_note_length_ms,
        minimum_frequency=minimum_frequency,
        maximum_frequency=maximum_frequency,
        multiple_pitch_bends=False,
        melodia_trick=True,
    )

    midi_path = output_dir / f"{job_id}.mid"
    midi_data.write(str(midi_path))

    tempo_bpm = _estimate_tempo(midi_data)
    duration = float(midi_data.get_end_time())

    note_list = [
        {
            "start": round(start, 4),
            "end": round(end, 4),
            "pitch_midi": int(pitch),
            "pitch_name": pretty_midi.note_number_to_name(int(pitch)),
            "velocity": int(velocity),
        }
        for (start, end, pitch, velocity, _pitch_bends) in note_events
    ]

    score = _midi_to_score(midi_path, title=title)

    musicxml_path = output_dir / f"{job_id}.musicxml"
    score.write("musicxml", fp=str(musicxml_path))

    pdf_path = _render_pdf(score, output_dir / f"{job_id}.pdf")

    return TranscriptionResult(
        job_id=job_id,
        midi_path=midi_path,
        musicxml_path=musicxml_path,
        pdf_path=pdf_path,
        notes=note_list,
        tempo_bpm=round(tempo_bpm, 2),
        duration_seconds=round(duration, 3),
        num_notes=len(note_list),
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Transcribe audio to sheet music.")
    parser.add_argument("audio", help="Path to the input audio file")
    parser.add_argument(
        "-o", "--output", default="./output", help="Output directory"
    )
    parser.add_argument("--title", default="Transcribed Score")
    parser.add_argument("--onset-threshold", type=float, default=0.5)
    parser.add_argument("--frame-threshold", type=float, default=0.3)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    job_id = Path(args.audio).stem
    result = transcribe_audio(
        args.audio,
        args.output,
        job_id=job_id,
        title=args.title,
        onset_threshold=args.onset_threshold,
        frame_threshold=args.frame_threshold,
    )
    print(f"Wrote {result.num_notes} notes (~{result.tempo_bpm} bpm)")
    print(f"  MIDI:     {result.midi_path}")
    print(f"  MusicXML: {result.musicxml_path}")
    if result.pdf_path:
        print(f"  PDF:      {result.pdf_path}")
