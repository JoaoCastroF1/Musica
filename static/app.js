"use strict";

const form = document.getElementById("upload-form");
const dropzone = document.getElementById("dropzone");
const audioInput = document.getElementById("audio-input");
const submitBtn = document.getElementById("submit-btn");
const statusCard = document.getElementById("status-card");
const statusText = document.getElementById("status-text");
const resultCard = document.getElementById("result-card");
const osmdContainer = document.getElementById("osmd-container");

const statNotes = document.getElementById("stat-notes");
const statTempo = document.getElementById("stat-tempo");
const statKey = document.getElementById("stat-key");
const statMeter = document.getElementById("stat-meter");
const statDuration = document.getElementById("stat-duration");
const noteList = document.getElementById("note-list");

const dlMusicXml = document.getElementById("dl-musicxml");
const dlMidi = document.getElementById("dl-midi");
const dlPdf = document.getElementById("dl-pdf");

let osmd = null;

function show(el) {
  el.classList.remove("hidden");
}
function hide(el) {
  el.classList.add("hidden");
}

function setStatus(text, isError = false) {
  statusText.textContent = text;
  statusText.classList.toggle("error", isError);
}

function updateDropzoneLabel() {
  const label = dropzone.querySelector(".dropzone-label strong");
  if (audioInput.files && audioInput.files[0]) {
    label.textContent = audioInput.files[0].name;
  } else {
    label.textContent = "Clique";
  }
}

audioInput.addEventListener("change", updateDropzoneLabel);

["dragenter", "dragover"].forEach((ev) => {
  dropzone.addEventListener(ev, (e) => {
    e.preventDefault();
    dropzone.classList.add("dragover");
  });
});
["dragleave", "drop"].forEach((ev) => {
  dropzone.addEventListener(ev, (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
  });
});
dropzone.addEventListener("drop", (e) => {
  if (e.dataTransfer.files && e.dataTransfer.files[0]) {
    audioInput.files = e.dataTransfer.files;
    updateDropzoneLabel();
  }
});

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  if (!audioInput.files || !audioInput.files[0]) return;

  const data = new FormData(form);

  submitBtn.disabled = true;
  hide(resultCard);
  show(statusCard);
  setStatus("Enviando áudio…");

  let jobId;
  try {
    const res = await fetch("/api/transcribe", { method: "POST", body: data });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.error || `HTTP ${res.status}`);
    }
    const json = await res.json();
    jobId = json.job_id;
  } catch (err) {
    setStatus("Erro: " + err.message, true);
    submitBtn.disabled = false;
    return;
  }

  setStatus("Transcrevendo (pode levar alguns segundos por minuto de áudio)…");

  try {
    const job = await pollJob(jobId);
    await renderResult(job);
  } catch (err) {
    setStatus("Erro: " + err.message, true);
  } finally {
    submitBtn.disabled = false;
  }
});

async function pollJob(jobId) {
  while (true) {
    await new Promise((r) => setTimeout(r, 1200));
    const res = await fetch(`/api/job/${jobId}`);
    if (!res.ok) throw new Error("não foi possível consultar o trabalho");
    const job = await res.json();
    if (job.status === "done") return job;
    if (job.status === "error") throw new Error(job.error || "falha desconhecida");
  }
}

async function renderResult(job) {
  hide(statusCard);
  show(resultCard);

  const result = job.result || {};
  const totalLabel =
    result.num_notes_raw && result.num_notes_raw !== result.num_notes
      ? `${result.num_notes} / ${result.num_notes_raw}`
      : (result.num_notes ?? "—");
  statNotes.textContent = totalLabel;
  statTempo.textContent = result.tempo_bpm ? `${result.tempo_bpm} bpm` : "—";
  statKey.textContent = result.key || "—";
  statMeter.textContent = result.time_signature || "—";
  statDuration.textContent = result.duration_seconds
    ? `${result.duration_seconds.toFixed(1)} s`
    : "—";

  configureDownload(dlMusicXml, `/api/download/${job.id}/musicxml`, "MusicXML");
  configureDownload(dlMidi, `/api/download/${job.id}/midi`, "MIDI");
  if (result.pdf_path) {
    configureDownload(dlPdf, `/api/download/${job.id}/pdf`, "PDF");
  } else {
    dlPdf.hidden = true;
  }

  renderNoteList(result.notes || []);
  await renderScore(job.id);
}

function configureDownload(el, href, label) {
  el.href = href;
  el.textContent = `Baixar ${label}`;
  el.hidden = false;
  el.setAttribute("download", "");
}

function confidenceColor(c) {
  if (c >= 0.8) return "#34d399";
  if (c >= 0.6) return "#fbbf24";
  return "#f87171";
}

function renderNoteList(notes) {
  if (!notes.length) {
    noteList.textContent = "(nenhuma nota detectada)";
    return;
  }
  const rows = notes
    .slice(0, 500)
    .map((n) => {
      const dur = (n.end - n.start).toFixed(2);
      const conf = n.confidence ?? n.velocity / 127;
      const color = confidenceColor(conf);
      return `<div class="note-row"><span>${n.start.toFixed(2)}s</span><span>${dur}s</span><strong style="color:${color}">${n.pitch_name}</strong><span>${conf.toFixed(2)}</span></div>`;
    })
    .join("");
  const trailing =
    notes.length > 500
      ? `<div class="note-row"><span>…</span><span>${notes.length - 500} mais</span></div>`
      : "";
  noteList.innerHTML = rows + trailing;
}

async function renderScore(jobId) {
  const res = await fetch(`/api/download/${jobId}/musicxml`);
  if (!res.ok) return;
  const xml = await res.text();

  if (!osmd) {
    osmd = new opensheetmusicdisplay.OpenSheetMusicDisplay(osmdContainer, {
      autoResize: true,
      backend: "svg",
      drawTitle: true,
      drawComposer: true,
    });
  }
  try {
    await osmd.load(xml);
    osmd.render();
  } catch (err) {
    osmdContainer.innerHTML =
      '<p style="color:#666;padding:20px">A partitura não pôde ser renderizada (' +
      err.message +
      '). Use os botões de download para abri-la em outro programa.</p>';
  }
}
