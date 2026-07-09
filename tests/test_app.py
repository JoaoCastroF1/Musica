"""Tests for the Flask HTTP layer.

These avoid exercising the Basic Pitch model (slow, heavy deps). We monkey-patch
the transcription pipeline and verify the routing / job-store behavior.
"""

from __future__ import annotations

import io
import sys
import time
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    import importlib

    import app as app_module

    importlib.reload(app_module)
    app_module.app.config["TESTING"] = True
    with app_module.app.test_client() as c:
        yield c, app_module


def test_index_renders(client):
    c, _ = client
    res = c.get("/")
    assert res.status_code == 200
    assert b"Musica" in res.data


def test_healthz(client):
    c, _ = client
    res = c.get("/healthz")
    assert res.status_code == 200
    assert res.get_json() == {"status": "ok"}


def test_transcribe_missing_file(client):
    c, _ = client
    res = c.post("/api/transcribe", data={})
    assert res.status_code == 400


def test_transcribe_bad_extension(client):
    c, _ = client
    data = {"audio": (io.BytesIO(b"x"), "song.txt")}
    res = c.post("/api/transcribe", data=data, content_type="multipart/form-data")
    assert res.status_code == 400


def test_transcribe_pipeline_happy_path(client):
    c, app_module = client

    fake_midi = app_module.OUTPUT_DIR / "fake.mid"
    fake_xml = app_module.OUTPUT_DIR / "fake.musicxml"
    fake_midi.write_bytes(b"MThd")
    fake_xml.write_text("<score/>")

    def fake_transcribe(audio_path, output_dir, job_id, **_kwargs):
        from transcribe import TranscriptionResult

        return TranscriptionResult(
            job_id=job_id,
            midi_path=fake_midi,
            musicxml_path=fake_xml,
            pdf_path=None,
            notes=[
                {
                    "start": 0.0,
                    "end": 0.5,
                    "pitch_midi": 60,
                    "pitch_name": "C4",
                    "velocity": 80,
                }
            ],
            tempo_bpm=120.0,
            duration_seconds=0.5,
            num_notes=1,
        )

    with patch.object(app_module, "transcribe_audio", side_effect=fake_transcribe):
        data = {"audio": (io.BytesIO(b"\0" * 100), "song.wav")}
        res = c.post(
            "/api/transcribe", data=data, content_type="multipart/form-data"
        )
        assert res.status_code == 202
        job_id = res.get_json()["job_id"]

        for _ in range(50):
            j = c.get(f"/api/job/{job_id}").get_json()
            if j["status"] in ("done", "error"):
                break
            time.sleep(0.05)
        assert j["status"] == "done", j
        assert j["result"]["num_notes"] == 1

        midi = c.get(f"/api/download/{job_id}/midi")
        assert midi.status_code == 200
        assert midi.data.startswith(b"MThd")

        xml = c.get(f"/api/download/{job_id}/musicxml")
        assert xml.status_code == 200

        pdf = c.get(f"/api/download/{job_id}/pdf")
        assert pdf.status_code == 404


def test_transcribe_pipeline_error(client):
    c, app_module = client

    def boom(*_args, **_kwargs):
        raise RuntimeError("nope")

    with patch.object(app_module, "transcribe_audio", side_effect=boom):
        data = {"audio": (io.BytesIO(b"\0" * 100), "song.wav")}
        res = c.post(
            "/api/transcribe", data=data, content_type="multipart/form-data"
        )
        job_id = res.get_json()["job_id"]
        for _ in range(50):
            j = c.get(f"/api/job/{job_id}").get_json()
            if j["status"] in ("done", "error"):
                break
            time.sleep(0.05)
        assert j["status"] == "error"
        assert "nope" in j["error"]


def test_unknown_job(client):
    c, _ = client
    res = c.get("/api/job/does_not_exist")
    assert res.status_code == 404


def test_invalid_format(client):
    c, app_module = client
    app_module.jobs.create("abc", "song.wav")
    app_module.jobs.update(
        "abc",
        status="done",
        result={"midi_path": "x", "musicxml_path": "y", "pdf_path": None},
    )
    res = c.get("/api/download/abc/bogus")
    assert res.status_code == 400


def _make_done_job(app_module, job_id="kitjob"):
    master = app_module.OUTPUT_DIR / f"{job_id}_master.wav"
    master.write_bytes(b"RIFF" + b"\0" * 32)
    xml = app_module.OUTPUT_DIR / f"{job_id}.musicxml"
    xml.write_text("<score-partwise/>", encoding="utf-8")
    app_module.jobs.create(job_id, "song.wav")
    app_module.jobs.update(
        job_id,
        status="done",
        result={
            "midi_path": None,
            "musicxml_path": str(xml),
            "pdf_path": None,
            "duration_seconds": 12.3,
            "master_path": str(master),
            "audio_sha256": "deadbeef",
        },
    )
    return job_id


def test_kit_endpoint_happy_path(client):
    c, app_module = client
    job_id = _make_done_job(app_module)

    fake_zip = app_module.OUTPUT_DIR / f"{job_id}_kit.zip"

    def fake_build(output_dir, jid, work, phonogram, artifacts, audio_path=None):
        assert work.title == "Minha Canção"
        assert phonogram.duration_seconds == 12.3  # fell back to job result
        fake_zip.write_bytes(b"PK\x03\x04fakezip")
        return fake_zip

    with patch.object(
        app_module.registration, "build_registration_kit", side_effect=fake_build
    ):
        res = c.post(
            f"/api/kit/{job_id}",
            json={
                "title": "Minha Canção",
                "authors": [{"name": "João Castro", "share_percent": 100}],
            },
        )
        assert res.status_code == 200, res.get_json()
        kit_url = res.get_json()["kit_url"]

    dl = c.get(kit_url)
    assert dl.status_code == 200
    assert dl.data.startswith(b"PK")


def test_kit_unknown_job(client):
    c, _ = client
    res = c.post("/api/kit/nope", json={"title": "x", "authors": [{"name": "a"}]})
    assert res.status_code == 404


def test_kit_job_not_done(client):
    c, app_module = client
    app_module.jobs.create("pending1", "song.wav")
    res = c.post("/api/kit/pending1", json={"title": "x", "authors": [{"name": "a"}]})
    assert res.status_code == 409


def test_kit_requires_title(client):
    c, app_module = client
    job_id = _make_done_job(app_module, job_id="kitjob2")
    res = c.post(f"/api/kit/{job_id}", json={"authors": [{"name": "a"}]})
    assert res.status_code == 400
    assert "título" in res.get_json()["error"]


def test_kit_requires_json_body(client):
    c, app_module = client
    job_id = _make_done_job(app_module, job_id="kitjob3")
    res = c.post(f"/api/kit/{job_id}", data="not json", content_type="text/plain")
    assert res.status_code == 400


def test_kit_download_before_generation(client):
    c, app_module = client
    job_id = _make_done_job(app_module, job_id="kitjob4")
    res = c.get(f"/api/download/{job_id}/kit")
    assert res.status_code == 404


def test_audio_master_streaming(client):
    c, app_module = client
    job_id = _make_done_job(app_module, job_id="audiojob")
    res = c.get(f"/api/audio/{job_id}")
    assert res.status_code == 200
    assert res.mimetype == "audio/wav"
    assert res.data.startswith(b"RIFF")


def test_audio_unknown_job(client):
    c, _ = client
    assert c.get("/api/audio/nope").status_code == 404


def test_audio_missing_master(client):
    c, app_module = client
    app_module.jobs.create("nomaster", "song.wav")
    app_module.jobs.update(
        "nomaster", status="done", result={"musicxml_path": "x"}
    )
    assert c.get("/api/audio/nomaster").status_code == 404
