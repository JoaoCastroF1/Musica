"""Lyrics transcription via faster-whisper.

Uses faster-whisper (a CTranslate2 reimplementation of OpenAI's Whisper that
runs efficiently on CPU with int8 quantization) to transcribe sung vocals.
Whisper handles Brazilian Portuguese well, but sung lyrics over loud
instrumentals degrade accuracy — the UI presents the result as an editable
draft that the user must review before registering the work.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

ALLOWED_MODELS: tuple[str, ...] = ("tiny", "base", "small", "medium")

_MODEL_CACHE: dict[str, object] = {}


@dataclass
class LyricsResult:
    """Transcribed lyrics plus segment timing and language metadata."""

    text: str
    language: str
    language_probability: float
    segments: list[dict] = field(default_factory=list)
    model_size: str = "small"


def transcribe_lyrics(
    audio_path: str | Path,
    model_size: str = "small",
    language: Optional[str] = None,
) -> LyricsResult:
    """Transcribe the vocal line of an audio file into lyrics text.

    Parameters
    ----------
    audio_path:
        Path to a readable audio file.
    model_size:
        Whisper model size — one of ALLOWED_MODELS. Bigger is more accurate
        and slower; "small" is a good CPU default. Invalid values fall back
        to "small".
    language:
        ISO language code (e.g. "pt") to force, or None to auto-detect.

    Raises
    ------
    FileNotFoundError
        If the audio file does not exist.
    RuntimeError
        If faster-whisper is not installed.
    """
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"audio file not found: {audio_path}")

    if model_size not in ALLOWED_MODELS:
        logger.warning("unknown whisper model %r — falling back to 'small'", model_size)
        model_size = "small"

    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError(
            "faster-whisper is not installed — run: pip install faster-whisper"
        ) from exc

    model = _MODEL_CACHE.get(model_size)
    if model is None:
        logger.info("loading whisper model %r (first use)", model_size)
        model = WhisperModel(model_size, device="cpu", compute_type="int8")
        _MODEL_CACHE[model_size] = model

    segments_iter, info = model.transcribe(
        str(audio_path),
        language=language or None,
        vad_filter=True,
        beam_size=5,
    )

    segments: list[dict] = []
    lines: list[str] = []
    for seg in segments_iter:
        text = seg.text.strip()
        if not text:
            continue
        segments.append(
            {
                "start": round(float(seg.start), 3),
                "end": round(float(seg.end), 3),
                "text": text,
            }
        )
        lines.append(text)

    return LyricsResult(
        text="\n".join(lines),
        language=getattr(info, "language", language or "") or "",
        language_probability=float(getattr(info, "language_probability", 0.0) or 0.0),
        segments=segments,
        model_size=model_size,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Transcribe lyrics from audio.")
    parser.add_argument("audio", help="Path to the input audio file")
    parser.add_argument("--model", default="small", choices=list(ALLOWED_MODELS))
    parser.add_argument("--language", default=None, help="Force language (e.g. pt)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    result = transcribe_lyrics(args.audio, model_size=args.model, language=args.language)
    print(
        f"Language: {result.language} "
        f"(confidence {result.language_probability:.0%})\n"
    )
    print(result.text)
