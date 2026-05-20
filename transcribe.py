"""Audio-to-sheet-music transcription pipeline.

Uses Spotify's Basic Pitch (a small CNN trained on multi-instrument data) to
predict MIDI notes from audio, then layers a few CPU-friendly refinements on
top: librosa-based tempo detection, music21 Krumhansl-Schmuckler key analysis,
a simple meter heuristic, and confidence-based filtering of spurious notes.
The output is rendered as MusicXML, MIDI, and (when Lilypond is available) PDF.
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import librosa
import numpy as np
import pretty_midi
from basic_pitch import ICASSP_2022_MODEL_PATH
from basic_pitch.inference import predict
from music21 import (
    analysis,
    converter,
    environment,
    instrument,
    key as m21_key,
    metadata,
    meter as m21_meter,
    stream,
    tempo as m21_tempo,
)

logger = logging.getLogger(__name__)

SUPPORTED_METERS: tuple[str, ...] = ("4/4", "3/4", "2/4", "6/8")


@dataclass
class TranscriptionResult:
    """Paths to the generated artifacts and metadata for a single job."""

    job_id: str
    midi_path: Path
    musicxml_path: Path
    pdf_path: Optional[Path] = None
    notes: list[dict] = field(default_factory=list)
    tempo_bpm: float = 120.0
    duration_seconds: float = 0.0
    num_notes: int = 0
    key: str = "C major"
    time_signature: str = "4/4"
    num_notes_raw: int = 0


def _detect_tempo(audio_path: Path) -> float:
    """Estimate tempo (BPM) from the raw audio with librosa's beat tracker."""
    try:
        y, sr = librosa.load(str(audio_path), sr=22050, mono=True)
        tempo, _beats = librosa.beat.beat_track(y=y, sr=sr)
        bpm = float(np.atleast_1d(tempo)[0])
        if not np.isfinite(bpm) or bpm <= 0:
            return 120.0
        return bpm
    except Exception as exc:  # noqa: BLE001 — librosa errors vary by codec
        logger.warning("librosa tempo detection failed: %s", exc)
        return 120.0


def _detect_key(score: stream.Score) -> m21_key.Key:
    """Estimate the score's key using music21's Krumhansl-Schmuckler analyzer."""
    try:
        analyzer = analysis.discrete.KrumhanslSchmuckler()
        result = analyzer.getSolution(score)
        if isinstance(result, m21_key.Key):
            return result
    except Exception as exc:  # noqa: BLE001 — music21 raises various types
        logger.warning("key detection failed: %s", exc)
    return m21_key.Key("C")


def _detect_meter(note_events: list[tuple], bpm: float) -> str:
    """Pick a likely time signature from the onset distribution.

    The heuristic projects onsets onto beats (using the detected BPM) and
    measures how peaky their position is modulo 2, 3, and 4. The peakiest
    grouping wins. Defaults to 4/4 when the signal is too weak to be sure.
    """
    if not note_events or bpm <= 0:
        return "4/4"

    onsets_sec = np.array([float(ev[0]) for ev in note_events], dtype=np.float64)
    onsets_beats = onsets_sec * (bpm / 60.0)

    def peakiness(period: int) -> float:
        mod = onsets_beats % period
        hist, _ = np.histogram(mod, bins=period * 4, range=(0, period))
        if hist.sum() == 0:
            return 0.0
        return float(hist.max() / hist.sum())

    scores = {
        "4/4": peakiness(4),
        "3/4": peakiness(3),
        "2/4": peakiness(2),
    }
    best, best_score = max(scores.items(), key=lambda kv: kv[1])
    if best_score < 0.35:
        return "4/4"
    return best


def _postprocess_note_events(
    note_events: list[tuple],
    min_confidence: float,
    merge_gap_seconds: float,
) -> list[tuple]:
    """Drop low-confidence notes and merge fragments of the same pitch.

    Basic Pitch encodes its per-note confidence as the MIDI velocity (0-127),
    so we treat ``velocity / 127`` as confidence. Notes below the threshold
    are dropped; adjacent same-pitch notes separated by a gap smaller than
    ``merge_gap_seconds`` are merged into a single sustained note.
    """
    threshold = min_confidence * 127.0
    filtered = [ev for ev in note_events if ev[3] >= threshold]
    filtered.sort(key=lambda ev: (ev[2], ev[0]))

    merged: list[tuple] = []
    for ev in filtered:
        start, end, pitch, velocity, bends = ev
        if merged:
            p_start, p_end, p_pitch, p_vel, p_bends = merged[-1]
            if p_pitch == pitch and start - p_end <= merge_gap_seconds:
                merged[-1] = (
                    p_start,
                    max(p_end, end),
                    p_pitch,
                    int(max(p_vel, velocity)),
                    p_bends,
                )
                continue
        merged.append(ev)
    merged.sort(key=lambda ev: ev[0])
    return merged


def _rebuild_midi(
    note_events: list[tuple], program: int = 0
) -> pretty_midi.PrettyMIDI:
    """Build a fresh PrettyMIDI object from a post-processed event list."""
    midi = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=program)
    for start, end, pitch, velocity, _bends in note_events:
        if end <= start:
            continue
        inst.notes.append(
            pretty_midi.Note(
                velocity=int(np.clip(velocity, 1, 127)),
                pitch=int(pitch),
                start=float(start),
                end=float(end),
            )
        )
    midi.instruments.append(inst)
    return midi


def _midi_to_score(
    midi_path: Path,
    title: str,
    key_obj: Optional[m21_key.Key] = None,
    time_signature: Optional[str] = None,
    bpm: Optional[float] = None,
) -> stream.Score:
    """Parse a MIDI file into a music21 Score and inject key/meter/tempo.

    music21's MIDI parser handles tempo and quantization to the nearest
    sensible note value via ``quantizePost=True``; we then override its
    inferred key/meter/tempo with the ones we detected ourselves, since
    they're typically more accurate than music21's MIDI-only heuristics.
    """
    score = converter.parse(str(midi_path), quantizePost=True)

    score.metadata = metadata.Metadata()
    score.metadata.title = title
    score.metadata.composer = "Transcribed by Musica"

    parts = list(score.parts) or [score]
    first_part = parts[0]

    if bpm is not None and bpm > 0:
        for existing in list(first_part.getElementsByClass(m21_tempo.MetronomeMark)):
            first_part.remove(existing)
        first_part.insert(0, m21_tempo.MetronomeMark(number=round(bpm)))

    if time_signature:
        for existing in list(first_part.getElementsByClass(m21_meter.TimeSignature)):
            first_part.remove(existing)
        try:
            first_part.insert(0, m21_meter.TimeSignature(time_signature))
        except Exception as exc:  # noqa: BLE001 — bad meter strings raise plainly
            logger.warning("invalid time signature %r: %s", time_signature, exc)

    if key_obj is not None:
        for existing in list(first_part.getElementsByClass(m21_key.Key)):
            first_part.remove(existing)
        first_part.insert(0, key_obj)

    for part in parts:
        if not part.getInstruments(returnDefault=False):
            part.insert(0, instrument.Piano())

    try:
        score.makeNotation(inPlace=True)
    except Exception as exc:  # noqa: BLE001 — makeNotation is occasionally fragile
        logger.warning("makeNotation failed: %s", exc)

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
    min_confidence: float = 0.5,
    merge_gap_ms: float = 50.0,
    bpm_override: Optional[float] = None,
    key_override: Optional[str] = None,
    time_signature_override: Optional[str] = None,
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
    min_confidence:
        Drop notes whose Basic Pitch velocity is below ``min_confidence``
        of the maximum (0.0 keeps everything).
    merge_gap_ms:
        Merge adjacent same-pitch notes separated by less than this gap.
    bpm_override, key_override, time_signature_override:
        Skip automatic detection and force these values into the score.
        ``key_override`` accepts strings like ``"G"``, ``"f# minor"``.
    """
    audio_path = Path(audio_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not audio_path.exists():
        raise FileNotFoundError(f"audio file not found: {audio_path}")

    logger.info("Transcribing %s (job=%s)", audio_path.name, job_id)

    _model_output, _bp_midi, note_events_raw = predict(
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

    num_notes_raw = len(note_events_raw)

    note_events = _postprocess_note_events(
        note_events_raw,
        min_confidence=min_confidence,
        merge_gap_seconds=merge_gap_ms / 1000.0,
    )

    midi_data = _rebuild_midi(note_events)
    midi_path = output_dir / f"{job_id}.mid"
    midi_data.write(str(midi_path))

    bpm = bpm_override if bpm_override else _detect_tempo(audio_path)
    duration = float(midi_data.get_end_time())

    time_signature = (
        time_signature_override
        if time_signature_override in SUPPORTED_METERS
        else _detect_meter(note_events, bpm)
    )

    key_obj: Optional[m21_key.Key] = None
    if key_override:
        try:
            key_obj = m21_key.Key(key_override)
        except Exception as exc:  # noqa: BLE001 — bad key strings raise plainly
            logger.warning("invalid key override %r: %s", key_override, exc)

    note_list = [
        {
            "start": round(start, 4),
            "end": round(end, 4),
            "pitch_midi": int(pitch),
            "pitch_name": pretty_midi.note_number_to_name(int(pitch)),
            "velocity": int(velocity),
            "confidence": round(min(1.0, float(velocity) / 127.0), 3),
        }
        for (start, end, pitch, velocity, _pitch_bends) in note_events
    ]

    score = _midi_to_score(
        midi_path,
        title=title,
        key_obj=key_obj,
        time_signature=time_signature,
        bpm=bpm,
    )

    if key_obj is None:
        key_obj = _detect_key(score)
        first_part = next(iter(score.parts), score)
        for existing in list(first_part.getElementsByClass(m21_key.Key)):
            first_part.remove(existing)
        first_part.insert(0, key_obj)

    musicxml_path = output_dir / f"{job_id}.musicxml"
    score.write("musicxml", fp=str(musicxml_path))

    pdf_path = _render_pdf(score, output_dir / f"{job_id}.pdf")

    return TranscriptionResult(
        job_id=job_id,
        midi_path=midi_path,
        musicxml_path=musicxml_path,
        pdf_path=pdf_path,
        notes=note_list,
        tempo_bpm=round(bpm, 2),
        duration_seconds=round(duration, 3),
        num_notes=len(note_list),
        num_notes_raw=num_notes_raw,
        key=str(key_obj) if key_obj else "C major",
        time_signature=time_signature,
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
    parser.add_argument("--min-confidence", type=float, default=0.5)
    parser.add_argument("--bpm", type=float, default=None, help="Override BPM")
    parser.add_argument("--key", default=None, help="Override key (e.g. 'G' or 'f# minor')")
    parser.add_argument(
        "--time-signature",
        default=None,
        choices=list(SUPPORTED_METERS),
        help="Override time signature",
    )
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
        min_confidence=args.min_confidence,
        bpm_override=args.bpm,
        key_override=args.key,
        time_signature_override=args.time_signature,
    )
    print(
        f"Wrote {result.num_notes} notes (raw {result.num_notes_raw}) — "
        f"{result.key}, {result.time_signature}, {result.tempo_bpm} bpm"
    )
    print(f"  MIDI:     {result.midi_path}")
    print(f"  MusicXML: {result.musicxml_path}")
    if result.pdf_path:
        print(f"  PDF:      {result.pdf_path}")
