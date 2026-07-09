"""Brazilian music copyright registration kit generator.

Bundles everything a composer needs to register a work and its phonogram in
Brazil into a single ZIP: a registration dossier (PDF), the lyrics (TXT +
PDF), the sheet-music artifacts already produced by the transcription
pipeline, the audio master, SHA-256 integrity hashes, structured metadata
(JSON), and a step-by-step guide covering Biblioteca Nacional (EDA),
collective-management associations (UBC, ABRAMUS, AMAR, SBACEM, SICAM),
ECAD, and ISRC.

The dossier organizes information and technical evidence; it does not replace
official registration and is not legal advice — that disclaimer is embedded
in every generated document.

Only stdlib + reportlab (pure Python) are used here.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import zipfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from xml.sax.saxutils import escape

logger = logging.getLogger(__name__)

ASSOCIATIONS = ("UBC", "ABRAMUS", "AMAR", "SBACEM", "SICAM")

_DISCLAIMER = (
    "Este dossiê organiza informações e evidências técnicas. Ele não "
    "substitui o registro oficial nem constitui aconselhamento jurídico."
)

_CHECKLIST = [
    (
        "Biblioteca Nacional (EDA) — prova de anterioridade",
        [
            "Acesse o Escritório de Direitos Autorais da Fundação Biblioteca "
            "Nacional em gov.br (busque por “registro de obra intelectual — "
            "Biblioteca Nacional”).",
            "Preencha o requerimento de registro de obra musical (letra e/ou "
            "partitura), pague a GRU e anexe a cópia da obra — use a "
            "partitura e a letra incluídas neste kit.",
            "O registro é declaratório: serve como prova de autoria e "
            "anterioridade em disputas.",
        ],
    ),
    (
        "Associação de gestão coletiva — obra e fonograma",
        [
            "Filie-se a uma associação (UBC, ABRAMUS, AMAR, SBACEM ou SICAM) "
            "como autor e, se for o caso, como intérprete e/ou produtor "
            "fonográfico.",
            "Cadastre a OBRA informando todos os autores e seus percentuais — "
            "use os dados da seção “Autores e participações” deste dossiê.",
            "Cadastre o FONOGRAMA (a gravação): intérprete principal, "
            "participantes, produtor fonográfico, ano de gravação e ISRC.",
        ],
    ),
    (
        "ECAD — royalties de execução pública",
        [
            "O ECAD não atende titulares diretamente: os cadastros feitos na "
            "sua associação alimentam o banco de dados do ECAD.",
            "Rádio, TV, shows, streaming e estabelecimentos pagam ao ECAD, "
            "que distribui às associações e elas repassam aos titulares.",
        ],
    ),
    (
        "ISRC — código do fonograma",
        [
            "O ISRC (ISO 3901) identifica cada gravação e é obrigatório para "
            "rastrear execuções do fonograma.",
            "Solicite à sua associação ou ao produtor fonográfico. Deixe o "
            "campo em branco no cadastro até recebê-lo.",
        ],
    ),
    (
        "Distribuição digital (streaming)",
        [
            "Para Spotify, Deezer, Apple Music etc., contrate uma "
            "distribuidora digital e informe o MESMO ISRC do fonograma, "
            "garantindo que a execução digital seja atribuída a você.",
        ],
    ),
]


@dataclass
class Author:
    """One author of the musical work."""

    name: str
    pseudonym: str = ""
    cpf: str = ""
    role: str = "letra e música"
    share_percent: float = 100.0
    association: str = ""


@dataclass
class Performer:
    """One performer credited on the phonogram."""

    name: str
    role: str = "intérprete"


@dataclass
class WorkMetadata:
    """The musical work (obra) as registered with BN and the associations."""

    title: str
    authors: list[Author] = field(default_factory=list)
    lyrics: str = ""
    subtitle: str = ""
    genre: str = ""
    language: str = "pt"
    year: Optional[int] = None


@dataclass
class PhonogramMetadata:
    """The recording (fonograma) as registered with the association."""

    main_performer: str = ""
    performers: list[Performer] = field(default_factory=list)
    producer: str = ""
    recording_year: Optional[int] = None
    recording_location: str = ""
    isrc: str = ""
    duration_seconds: Optional[float] = None


def sha256_of(path: str | Path) -> str:
    """SHA-256 hex digest of a file, streamed in 1 MiB chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _to_float(value, default: Optional[float]) -> Optional[float]:
    if value is None or value == "":
        return default
    try:
        return float(str(value).replace(",", "."))
    except (TypeError, ValueError):
        return default


def _to_int(value) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return None


def from_form(data: dict) -> tuple[WorkMetadata, PhonogramMetadata]:
    """Parse the JSON body sent by the frontend into metadata objects.

    Raises ValueError with a user-facing PT-BR message when required fields
    are missing. Tolerates and coerces sloppy numeric input; skips author and
    performer entries without a name.
    """
    if not isinstance(data, dict):
        raise ValueError("corpo JSON inválido")

    title = str(data.get("title") or "").strip()
    if not title:
        raise ValueError("título da obra é obrigatório")

    authors: list[Author] = []
    for raw in data.get("authors") or []:
        if not isinstance(raw, dict):
            continue
        name = str(raw.get("name") or "").strip()
        if not name:
            continue
        share = _to_float(raw.get("share_percent"), 100.0)
        authors.append(
            Author(
                name=name,
                pseudonym=str(raw.get("pseudonym") or "").strip(),
                cpf=str(raw.get("cpf") or "").strip(),
                role=str(raw.get("role") or "letra e música").strip(),
                share_percent=share if share is not None else 100.0,
                association=str(raw.get("association") or "").strip(),
            )
        )
    if not authors:
        raise ValueError("informe ao menos um autor com nome")

    total_share = sum(a.share_percent for a in authors)
    if abs(total_share - 100.0) > 0.5:
        logger.warning(
            "author shares sum to %.2f%% (expected 100%%) — keeping as-is",
            total_share,
        )

    performers: list[Performer] = []
    for raw in data.get("performers") or []:
        if not isinstance(raw, dict):
            continue
        name = str(raw.get("name") or "").strip()
        if not name:
            continue
        performers.append(
            Performer(name=name, role=str(raw.get("role") or "intérprete").strip())
        )

    work = WorkMetadata(
        title=title,
        authors=authors,
        lyrics=str(data.get("lyrics") or ""),
        subtitle=str(data.get("subtitle") or "").strip(),
        genre=str(data.get("genre") or "").strip(),
        language=str(data.get("language") or "pt").strip() or "pt",
        year=_to_int(data.get("year")),
    )
    phonogram = PhonogramMetadata(
        main_performer=str(data.get("main_performer") or "").strip(),
        performers=performers,
        producer=str(data.get("producer") or "").strip(),
        recording_year=_to_int(data.get("recording_year")),
        recording_location=str(data.get("recording_location") or "").strip(),
        isrc=str(data.get("isrc") or "").strip().upper(),
        duration_seconds=_to_float(data.get("duration_seconds"), None),
    )
    return work, phonogram


def _format_duration(seconds: Optional[float]) -> str:
    if not seconds or seconds <= 0:
        return "—"
    total = int(round(seconds))
    return f"{total // 60}:{total % 60:02d}"


def _reportlab_available() -> bool:
    try:
        import reportlab  # noqa: F401

        return True
    except ImportError:
        return False


def _pdf_styles():
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet

    base = getSampleStyleSheet()
    styles = {
        "title": ParagraphStyle(
            "KitTitle", parent=base["Title"], fontSize=20, spaceAfter=6
        ),
        "subtitle": ParagraphStyle(
            "KitSubtitle", parent=base["Normal"], fontSize=12, textColor="#444444"
        ),
        "h2": ParagraphStyle(
            "KitH2", parent=base["Heading2"], spaceBefore=14, spaceAfter=6
        ),
        "body": ParagraphStyle("KitBody", parent=base["Normal"], leading=14),
        "lyric": ParagraphStyle(
            "KitLyric", parent=base["Normal"], leading=15, spaceAfter=1
        ),
        "small": ParagraphStyle(
            "KitSmall", parent=base["Normal"], fontSize=8, textColor="#555555"
        ),
        "mono": ParagraphStyle(
            "KitMono", parent=base["Normal"], fontName="Courier", fontSize=6.5
        ),
        "disclaimer": ParagraphStyle(
            "KitDisclaimer",
            parent=base["Normal"],
            fontName="Helvetica-Oblique",
            fontSize=9,
            textColor="#555555",
            spaceBefore=16,
        ),
    }
    return styles


def _table(data, col_widths, header=True):
    from reportlab.lib import colors
    from reportlab.platypus import Table, TableStyle

    table = Table(data, colWidths=col_widths, repeatRows=1 if header else 0)
    style = [
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#999999")),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]
    if header:
        style += [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8e8e8")),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ]
    table.setStyle(TableStyle(style))
    return table


def _hash_paragraph(digest: str, styles):
    from reportlab.platypus import Paragraph

    return Paragraph(f"{escape(digest[:32])}<br/>{escape(digest[32:])}", styles["mono"])


def _kv_rows(pairs, styles):
    from reportlab.platypus import Paragraph

    rows = []
    for label, value in pairs:
        rows.append(
            [
                Paragraph(f"<b>{escape(label)}</b>", styles["body"]),
                Paragraph(escape(str(value) if value not in (None, "") else "—"), styles["body"]),
            ]
        )
    return rows


def _checklist_flowables(styles):
    from reportlab.platypus import Paragraph, Spacer

    flow = []
    for i, (heading, items) in enumerate(_CHECKLIST, start=1):
        flow.append(Paragraph(f"6.{i}. {escape(heading)}", styles["h2"]))
        for item in items:
            flow.append(Paragraph(f"• {escape(item)}", styles["body"]))
        flow.append(Spacer(1, 4))
    return flow


def _render_lyrics_pdf(work: WorkMetadata, out_path: Path) -> None:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import cm
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

    styles = _pdf_styles()
    author_names = ", ".join(a.name for a in work.authors) or "—"
    doc = SimpleDocTemplate(
        str(out_path),
        pagesize=A4,
        title=f"Letra — {work.title}",
        leftMargin=2.5 * cm,
        rightMargin=2.5 * cm,
        topMargin=2.5 * cm,
        bottomMargin=2.5 * cm,
    )
    flow = [
        Paragraph(escape(work.title), styles["title"]),
        Paragraph(f"Letra — {escape(author_names)}", styles["subtitle"]),
        Spacer(1, 18),
    ]
    for line in work.lyrics.splitlines():
        flow.append(Paragraph(escape(line) if line.strip() else "&nbsp;", styles["lyric"]))
    doc.build(flow)


def _render_dossier_pdf(
    work: WorkMetadata,
    phonogram: PhonogramMetadata,
    file_hashes: dict[str, str],
    job_id: str,
    generated_at: str,
    out_path: Path,
) -> None:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import cm
    from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer

    styles = _pdf_styles()
    author_names = ", ".join(a.name for a in work.authors) or "—"

    doc = SimpleDocTemplate(
        str(out_path),
        pagesize=A4,
        title=f"Dossiê de Registro — {work.title}",
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
    )

    flow = [
        Spacer(1, 96),
        Paragraph("Dossiê de Registro de Obra Musical", styles["title"]),
        Spacer(1, 12),
        Paragraph(escape(work.title), styles["h2"]),
    ]
    if work.subtitle:
        flow.append(Paragraph(escape(work.subtitle), styles["subtitle"]))
    flow += [
        Spacer(1, 10),
        Paragraph(f"Autores: {escape(author_names)}", styles["body"]),
        Paragraph(f"Gerado em: {escape(generated_at)} (UTC)", styles["small"]),
        Paragraph(f"Identificador do trabalho: {escape(job_id)}", styles["small"]),
        PageBreak(),
    ]

    flow.append(Paragraph("1. Identificação da obra", styles["h2"]))
    flow.append(
        _table(
            _kv_rows(
                [
                    ("Título", work.title),
                    ("Subtítulo", work.subtitle),
                    ("Gênero", work.genre),
                    ("Idioma", work.language),
                    ("Ano de composição", work.year),
                ],
                styles,
            ),
            col_widths=[4 * cm, 13 * cm],
            header=False,
        )
    )

    flow.append(Paragraph("2. Autores e participações", styles["h2"]))
    author_rows = [["Nome", "Pseudônimo", "CPF", "Função", "%", "Associação"]]
    for a in work.authors:
        author_rows.append(
            [
                escape(a.name),
                escape(a.pseudonym) or "—",
                escape(a.cpf) or "—",
                escape(a.role),
                f"{a.share_percent:g}",
                escape(a.association) or "—",
            ]
        )
    flow.append(
        _table(
            author_rows,
            col_widths=[4.2 * cm, 2.6 * cm, 2.9 * cm, 2.6 * cm, 1.2 * cm, 3.0 * cm],
        )
    )

    flow.append(Paragraph("3. Fonograma", styles["h2"]))
    other_performers = (
        ", ".join(f"{p.name} ({p.role})" for p in phonogram.performers) or "—"
    )
    flow.append(
        _table(
            _kv_rows(
                [
                    ("Intérprete principal", phonogram.main_performer),
                    ("Demais participantes", other_performers),
                    ("Produtor fonográfico", phonogram.producer),
                    ("Ano de gravação", phonogram.recording_year),
                    ("Local de gravação", phonogram.recording_location),
                    ("Duração", _format_duration(phonogram.duration_seconds)),
                    (
                        "ISRC",
                        phonogram.isrc
                        or "a solicitar via associação de gestão coletiva",
                    ),
                ],
                styles,
            ),
            col_widths=[5 * cm, 12 * cm],
            header=False,
        )
    )

    flow.append(Paragraph("4. Letra", styles["h2"]))
    if work.lyrics.strip():
        for line in work.lyrics.splitlines():
            flow.append(
                Paragraph(escape(line) if line.strip() else "&nbsp;", styles["lyric"])
            )
    else:
        flow.append(Paragraph("(obra instrumental — sem letra)", styles["body"]))

    flow.append(Paragraph("5. Integridade dos arquivos (prova de anterioridade)", styles["h2"]))
    flow.append(
        Paragraph(
            "Os resumos SHA-256 abaixo permitem verificar, a qualquer momento, "
            "que os arquivos deste kit não foram alterados desde a geração "
            "do dossiê.",
            styles["small"],
        )
    )
    flow.append(Spacer(1, 6))
    hash_rows = [["Arquivo", "SHA-256"]]
    for name in sorted(file_hashes):
        hash_rows.append([escape(name), _hash_paragraph(file_hashes[name], styles)])
    flow.append(_table(hash_rows, col_widths=[6 * cm, 11 * cm]))

    flow.append(PageBreak())
    flow.append(
        Paragraph("6. Próximos passos para registro e royalties", styles["h2"])
    )
    flow += _checklist_flowables(styles)
    flow.append(Paragraph(_DISCLAIMER, styles["disclaimer"]))

    doc.build(flow)


def _leiame_markdown(
    work: WorkMetadata, kit_files: list[str], generated_at: str
) -> str:
    lines = [
        f"# Kit de registro — {work.title}",
        "",
        f"Gerado em {generated_at} (UTC) pelo Musica.",
        "",
        "## Conteúdo do kit",
        "",
    ]
    lines += [f"- `{name}`" for name in sorted(kit_files)]
    lines += ["", "## Como registrar e receber royalties", ""]
    for i, (heading, items) in enumerate(_CHECKLIST, start=1):
        lines.append(f"### {i}. {heading}")
        lines.append("")
        lines += [f"1. {item}" for item in items]
        lines.append("")
    lines += ["---", "", f"_{_DISCLAIMER}_", ""]
    return "\n".join(lines)


def build_registration_kit(
    output_dir: str | Path,
    job_id: str,
    work: WorkMetadata,
    phonogram: PhonogramMetadata,
    artifacts: dict,
    audio_path: str | Path | None = None,
) -> Path:
    """Assemble the registration kit ZIP and return its path.

    ``artifacts`` maps {"musicxml": path|None, "midi": path|None,
    "pdf": path|None}; entries that are missing on disk are skipped.
    ``audio_path`` is the audio master to embed as the fonogram evidence.
    """
    if not _reportlab_available():
        raise RuntimeError("reportlab não está instalado — execute: pip install reportlab")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    staging = output_dir / f"{job_id}_kit_staging"
    staging.mkdir(exist_ok=True)

    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    try:
        # arcname -> source path for every file that goes into the zip
        members: dict[str, Path] = {}

        for key, arcname in (
            ("musicxml", "partitura.musicxml"),
            ("midi", "partitura.mid"),
            ("pdf", "partitura.pdf"),
        ):
            src = artifacts.get(key)
            if src and Path(src).exists():
                members[arcname] = Path(src)

        if audio_path and Path(audio_path).exists():
            suffix = Path(audio_path).suffix.lower() or ".bin"
            members[f"fonograma{suffix}"] = Path(audio_path)

        if work.lyrics.strip():
            letra_txt = staging / "letra.txt"
            letra_txt.write_text(work.lyrics.strip() + "\n", encoding="utf-8")
            members["letra.txt"] = letra_txt

            letra_pdf = staging / "letra.pdf"
            _render_lyrics_pdf(work, letra_pdf)
            members["letra.pdf"] = letra_pdf

        file_hashes = {arc: sha256_of(path) for arc, path in members.items()}

        dossier = staging / "dossie_registro.pdf"
        _render_dossier_pdf(work, phonogram, file_hashes, job_id, generated_at, dossier)
        members["dossie_registro.pdf"] = dossier
        file_hashes["dossie_registro.pdf"] = sha256_of(dossier)

        metadata = {
            "gerado_em": generated_at,
            "obra": {
                "titulo": work.title,
                "subtitulo": work.subtitle,
                "genero": work.genre,
                "idioma": work.language,
                "ano": work.year,
                "autores": [asdict(a) for a in work.authors],
                "letra": work.lyrics,
            },
            "fonograma": {
                "interprete_principal": phonogram.main_performer,
                "participantes": [asdict(p) for p in phonogram.performers],
                "produtor_fonografico": phonogram.producer,
                "ano_gravacao": phonogram.recording_year,
                "local_gravacao": phonogram.recording_location,
                "isrc": phonogram.isrc,
                "duracao_segundos": phonogram.duration_seconds,
                "duracao": _format_duration(phonogram.duration_seconds),
            },
            "arquivos": file_hashes,
        }
        meta_path = staging / "metadata.json"
        meta_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        members["metadata.json"] = meta_path

        leiame_path = staging / "LEIA-ME.md"
        leiame_path.write_text(
            _leiame_markdown(work, list(members) + ["LEIA-ME.md"], generated_at),
            encoding="utf-8",
        )
        members["LEIA-ME.md"] = leiame_path

        zip_path = output_dir / f"{job_id}_kit.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for arcname in sorted(members):
                zf.write(members[arcname], arcname)

        logger.info(
            "kit %s built: %d files, %d bytes",
            zip_path.name,
            len(members),
            zip_path.stat().st_size,
        )
        return zip_path
    finally:
        shutil.rmtree(staging, ignore_errors=True)
