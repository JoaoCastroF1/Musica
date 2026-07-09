"""Unit tests for registration.py (kit generator).

reportlab is a light pure-Python dep; the whole module is skipped when it
isn't installed.
"""

from __future__ import annotations

import hashlib
import json
import sys
import zipfile
from pathlib import Path

import pytest

pytest.importorskip("reportlab")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from registration import (  # noqa: E402
    Author,
    Performer,
    PhonogramMetadata,
    WorkMetadata,
    build_registration_kit,
    from_form,
    sha256_of,
)


def _form_payload(**overrides):
    payload = {
        "title": "Minha Canção",
        "subtitle": "demo",
        "genre": "MPB",
        "language": "pt",
        "year": "2026",
        "lyrics": "Primeira linha\nSegunda linha",
        "authors": [
            {"name": "João Castro", "cpf": "123.456.789-00", "role": "letra e música",
             "share_percent": "50", "association": "UBC", "pseudonym": "JC"},
            {"name": "Maria Silva", "share_percent": 50.0},
            {"name": "   "},  # skipped: empty name
        ],
        "main_performer": "João Castro",
        "performers": [{"name": "Ana Souza", "role": "backing vocal"}, {"name": ""}],
        "producer": "Estúdio X",
        "recording_year": 2026,
        "recording_location": "São Paulo/SP",
        "isrc": "br-abc-26-00001",
        "duration_seconds": "183.5",
    }
    payload.update(overrides)
    return payload


def test_from_form_happy_path():
    work, phonogram = from_form(_form_payload())
    assert work.title == "Minha Canção"
    assert work.year == 2026
    assert [a.name for a in work.authors] == ["João Castro", "Maria Silva"]
    assert work.authors[0].share_percent == pytest.approx(50.0)
    assert work.authors[0].association == "UBC"
    assert work.authors[1].role == "letra e música"
    assert phonogram.main_performer == "João Castro"
    assert [p.name for p in phonogram.performers] == ["Ana Souza"]
    assert phonogram.isrc == "BR-ABC-26-00001"
    assert phonogram.duration_seconds == pytest.approx(183.5)


def test_from_form_requires_title():
    with pytest.raises(ValueError, match="título"):
        from_form(_form_payload(title="   "))


def test_from_form_requires_named_author():
    with pytest.raises(ValueError, match="autor"):
        from_form(_form_payload(authors=[{"name": ""}]))
    with pytest.raises(ValueError, match="autor"):
        from_form(_form_payload(authors=[]))


def test_from_form_tolerates_bad_numbers():
    work, phonogram = from_form(
        _form_payload(
            year="não sei",
            duration_seconds="abc",
            authors=[{"name": "Solo", "share_percent": "banana"}],
        )
    )
    assert work.year is None
    assert phonogram.duration_seconds is None
    assert work.authors[0].share_percent == pytest.approx(100.0)


def test_from_form_accepts_shares_not_summing_100():
    work, _ = from_form(
        _form_payload(authors=[{"name": "A", "share_percent": 70},
                               {"name": "B", "share_percent": 70}])
    )
    assert sum(a.share_percent for a in work.authors) == pytest.approx(140.0)


def test_sha256_of_matches_hashlib(tmp_path):
    f = tmp_path / "blob.bin"
    f.write_bytes(b"conteudo de teste" * 1000)
    assert sha256_of(f) == hashlib.sha256(f.read_bytes()).hexdigest()


def _fake_artifacts(tmp_path):
    midi = tmp_path / "song.mid"
    midi.write_bytes(b"MThd\x00\x00\x00\x06")
    xml = tmp_path / "song.musicxml"
    xml.write_text("<score-partwise/>", encoding="utf-8")
    audio = tmp_path / "master.wav"
    audio.write_bytes(b"RIFF" + b"\x00" * 64)
    return {"midi": midi, "musicxml": xml, "pdf": None}, audio


def test_build_registration_kit_end_to_end(tmp_path):
    artifacts, audio = _fake_artifacts(tmp_path)
    work = WorkMetadata(
        title="Obra Teste",
        authors=[Author(name="João", cpf="123", share_percent=100.0, association="UBC")],
        lyrics="Linha um\n\nLinha dois",
        genre="Rock",
        year=2026,
    )
    phonogram = PhonogramMetadata(
        main_performer="João",
        performers=[Performer(name="Ana")],
        producer="Estúdio",
        recording_year=2026,
        duration_seconds=200.0,
        isrc="BRABC2600001",
    )

    zip_path = build_registration_kit(
        tmp_path / "out", "job123", work, phonogram, artifacts, audio_path=audio
    )
    assert zip_path.exists()
    assert zip_path.name == "job123_kit.zip"

    with zipfile.ZipFile(zip_path) as zf:
        names = set(zf.namelist())
        assert {
            "dossie_registro.pdf",
            "letra.txt",
            "letra.pdf",
            "metadata.json",
            "LEIA-ME.md",
            "partitura.mid",
            "partitura.musicxml",
            "fonograma.wav",
        } <= names
        assert "partitura.pdf" not in names  # artifact was None

        assert zf.read("dossie_registro.pdf").startswith(b"%PDF")
        assert zf.read("letra.pdf").startswith(b"%PDF")
        assert zf.read("letra.txt").decode("utf-8").startswith("Linha um")

        meta = json.loads(zf.read("metadata.json").decode("utf-8"))
        assert meta["obra"]["titulo"] == "Obra Teste"
        assert meta["obra"]["autores"][0]["name"] == "João"
        assert meta["fonograma"]["isrc"] == "BRABC2600001"
        assert meta["fonograma"]["duracao"] == "3:20"
        assert meta["arquivos"]["partitura.mid"] == sha256_of(artifacts["midi"])
        assert meta["arquivos"]["fonograma.wav"] == sha256_of(audio)
        assert "dossie_registro.pdf" in meta["arquivos"]

        leiame = zf.read("LEIA-ME.md").decode("utf-8")
        assert "Biblioteca Nacional" in leiame
        assert "ECAD" in leiame
        assert "ISRC" in leiame

    # staging dir must be cleaned up
    assert not (tmp_path / "out" / "job123_kit_staging").exists()


def test_build_kit_instrumental_work(tmp_path):
    artifacts, audio = _fake_artifacts(tmp_path)
    work = WorkMetadata(title="Instrumental", authors=[Author(name="João")], lyrics="")
    phonogram = PhonogramMetadata()

    zip_path = build_registration_kit(
        tmp_path / "out", "job456", work, phonogram, artifacts, audio_path=audio
    )
    with zipfile.ZipFile(zip_path) as zf:
        names = set(zf.namelist())
        assert "letra.txt" not in names
        assert "letra.pdf" not in names
        assert "dossie_registro.pdf" in names


def test_build_kit_escapes_markup_in_user_text(tmp_path):
    """User text with reportlab mini-HTML must not break PDF generation."""
    artifacts, audio = _fake_artifacts(tmp_path)
    work = WorkMetadata(
        title="<b>Injeção & teste</b>",
        authors=[Author(name="<i>João</i> & Cia")],
        lyrics="Linha com <font size=99> tags & entidades",
    )
    zip_path = build_registration_kit(
        tmp_path / "out", "job789", work, PhonogramMetadata(), artifacts,
        audio_path=audio,
    )
    with zipfile.ZipFile(zip_path) as zf:
        assert zf.read("dossie_registro.pdf").startswith(b"%PDF")
