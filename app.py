"""Flask app exposing the transcription pipeline as a web service."""

from __future__ import annotations

import logging
import os
import secrets
import shutil
import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from flask import Flask, abort, jsonify, render_template, request, send_file
from werkzeug.utils import secure_filename

import registration
from transcribe import TranscriptionResult, transcribe_audio

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "output"
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {"wav", "mp3", "flac", "ogg", "m4a", "aac", "aiff", "aif"}
MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50 MB

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("musica")


class JobStore:
    """Thread-safe in-memory job tracker. Good enough for a single-process app."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jobs: dict[str, dict] = {}

    def create(self, job_id: str, filename: str) -> None:
        with self._lock:
            self._jobs[job_id] = {
                "id": job_id,
                "status": "pending",
                "filename": filename,
                "created_at": time.time(),
                "error": None,
                "result": None,
            }

    def update(self, job_id: str, **fields) -> None:
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id].update(fields)

    def get(self, job_id: str) -> Optional[dict]:
        with self._lock:
            job = self._jobs.get(job_id)
            return dict(job) if job else None


jobs = JobStore()


def _allowed(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def _transcribe_lyrics_step(job_id: str, master_path: Path, params: dict) -> dict:
    """Run the optional lyrics pass; failures never abort the whole job."""
    try:
        from lyrics import transcribe_lyrics
    except Exception as exc:  # noqa: BLE001 — import error means dep missing
        return {"lyrics_error": f"transcrição de letra indisponível: {exc}"}
    try:
        lr = transcribe_lyrics(
            master_path,
            model_size=params.get("whisper_model") or "small",
            language=params.get("lyrics_language"),
        )
        return {
            "lyrics": {
                "text": lr.text,
                "language": lr.language,
                "language_probability": lr.language_probability,
                "segments": lr.segments,
                "model_size": lr.model_size,
            }
        }
    except Exception as exc:  # noqa: BLE001 — surface as warning, not job failure
        log.warning("job %s lyrics failed: %s", job_id, exc)
        return {"lyrics_error": str(exc)}


def _run_job(job_id: str, audio_path: Path, title: str, params: dict) -> None:
    jobs.update(job_id, status="running")
    try:
        # Keep a master copy in output/ — the kit embeds it as the fonogram
        # evidence, and lyrics transcription reads from it after the upload
        # is deleted.
        master_path = OUTPUT_DIR / f"{job_id}_master{audio_path.suffix.lower()}"
        shutil.copy2(audio_path, master_path)
        audio_sha256 = registration.sha256_of(master_path)

        result: TranscriptionResult = transcribe_audio(
            master_path,
            OUTPUT_DIR,
            job_id=job_id,
            title=title,
            onset_threshold=params.get("onset_threshold", 0.5),
            frame_threshold=params.get("frame_threshold", 0.3),
            minimum_note_length_ms=params.get("min_note_length_ms", 58.0),
            minimum_frequency=params.get("min_freq"),
            maximum_frequency=params.get("max_freq"),
            min_confidence=params.get("min_confidence", 0.5),
            merge_gap_ms=params.get("merge_gap_ms", 50.0),
            bpm_override=params.get("bpm_override"),
            key_override=params.get("key_override"),
            time_signature_override=params.get("time_signature_override"),
        )
        serializable = asdict(result)
        for key in ("midi_path", "musicxml_path", "pdf_path"):
            if serializable.get(key) is not None:
                serializable[key] = str(serializable[key])
        serializable["audio_sha256"] = audio_sha256
        serializable["master_path"] = str(master_path)

        if params.get("transcribe_lyrics"):
            serializable.update(_transcribe_lyrics_step(job_id, master_path, params))

        jobs.update(job_id, status="done", result=serializable)
        log.info("job %s done: %d notes", job_id, result.num_notes)
    except Exception as exc:  # noqa: BLE001 — surface any failure to the user
        log.exception("job %s failed", job_id)
        jobs.update(job_id, status="error", error=str(exc))
    finally:
        try:
            audio_path.unlink(missing_ok=True)
        except OSError:
            pass


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/healthz")
def healthz():
    return jsonify({"status": "ok"})


@app.route("/api/transcribe", methods=["POST"])
def api_transcribe():
    if "audio" not in request.files:
        return jsonify({"error": "missing 'audio' file field"}), 400
    f = request.files["audio"]
    if not f.filename:
        return jsonify({"error": "empty filename"}), 400
    if not _allowed(f.filename):
        return jsonify(
            {"error": f"unsupported extension; allowed: {sorted(ALLOWED_EXTENSIONS)}"}
        ), 400

    job_id = secrets.token_urlsafe(12).replace("-", "_")
    safe_name = secure_filename(f.filename)
    ext = safe_name.rsplit(".", 1)[1].lower()
    audio_path = UPLOAD_DIR / f"{job_id}.{ext}"
    f.save(audio_path)

    title = request.form.get("title") or Path(safe_name).stem or "Transcribed Score"

    def _float(name: str, default: Optional[float]) -> Optional[float]:
        raw = request.form.get(name)
        if raw is None or raw == "":
            return default
        try:
            return float(raw)
        except ValueError:
            return default

    def _str(name: str) -> Optional[str]:
        raw = request.form.get(name)
        if raw is None or raw.strip() == "":
            return None
        return raw.strip()

    def _bool(name: str) -> bool:
        return (request.form.get(name) or "").strip().lower() in ("on", "true", "1")

    params = {
        "onset_threshold": _float("onset_threshold", 0.5),
        "frame_threshold": _float("frame_threshold", 0.3),
        "min_note_length_ms": _float("min_note_length_ms", 58.0),
        "min_freq": _float("min_freq", None),
        "max_freq": _float("max_freq", None),
        "min_confidence": _float("min_confidence", 0.5),
        "merge_gap_ms": _float("merge_gap_ms", 50.0),
        "bpm_override": _float("bpm_override", None),
        "key_override": _str("key_override"),
        "time_signature_override": _str("time_signature_override"),
        "transcribe_lyrics": _bool("transcribe_lyrics"),
        "whisper_model": _str("whisper_model") or "small",
        "lyrics_language": _str("lyrics_language"),
    }

    jobs.create(job_id, safe_name)
    threading.Thread(
        target=_run_job, args=(job_id, audio_path, title, params), daemon=True
    ).start()

    return jsonify({"job_id": job_id, "status": "pending"}), 202


@app.route("/api/job/<job_id>")
def api_job(job_id: str):
    job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "unknown job"}), 404
    return jsonify(job)


@app.route("/api/kit/<job_id>", methods=["POST"])
def api_kit(job_id: str):
    """Build the registration kit ZIP from the transcription artifacts."""
    job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "unknown job"}), 404
    if job.get("status") != "done":
        return jsonify({"error": "a transcrição ainda não foi concluída"}), 409

    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "corpo JSON ausente ou inválido"}), 400

    result = job.get("result") or {}
    if not data.get("duration_seconds"):
        data["duration_seconds"] = result.get("duration_seconds")

    try:
        work, phonogram = registration.from_form(data)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    artifacts = {
        "midi": result.get("midi_path"),
        "musicxml": result.get("musicxml_path"),
        "pdf": result.get("pdf_path"),
    }
    try:
        kit_path = registration.build_registration_kit(
            OUTPUT_DIR,
            job_id,
            work,
            phonogram,
            artifacts,
            audio_path=result.get("master_path"),
        )
    except Exception as exc:  # noqa: BLE001 — missing reportlab, IO errors, etc.
        log.exception("kit %s failed", job_id)
        return jsonify({"error": f"falha ao gerar o kit: {exc}"}), 500

    jobs.update(job_id, kit_path=str(kit_path))
    return jsonify({"kit_url": f"/api/download/{job_id}/kit"})


@app.route("/api/download/<job_id>/<fmt>")
def api_download(job_id: str, fmt: str):
    job = jobs.get(job_id)
    if not job or job.get("status") != "done":
        abort(404)
    result = job.get("result") or {}

    # The kit path lives on the job dict (written by api_kit), not in result.
    if fmt == "kit":
        path = job.get("kit_path")
        if not path or not Path(path).exists():
            abort(404)
        return send_file(
            path,
            mimetype="application/zip",
            as_attachment=True,
            download_name=f"{job_id}_kit.zip",
        )

    mapping = {
        "midi": (result.get("midi_path"), "audio/midi", f"{job_id}.mid"),
        "musicxml": (
            result.get("musicxml_path"),
            "application/vnd.recordare.musicxml+xml",
            f"{job_id}.musicxml",
        ),
        "pdf": (result.get("pdf_path"), "application/pdf", f"{job_id}.pdf"),
    }
    if fmt not in mapping:
        abort(400)
    path, mime, download_name = mapping[fmt]
    if not path or not Path(path).exists():
        abort(404)
    return send_file(path, mimetype=mime, as_attachment=True, download_name=download_name)


@app.errorhandler(413)
def too_large(_e):
    return jsonify({"error": "file too large (max 50 MB)"}), 413


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
