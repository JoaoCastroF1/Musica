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

const lyricsCard = document.getElementById("lyrics-card");
const lyricsText = document.getElementById("lyrics-text");
const lyricsLang = document.getElementById("lyrics-lang");

const kitCard = document.getElementById("kit-card");
const kitForm = document.getElementById("kit-form");
const kitStatus = document.getElementById("kit-status");
const kitSubmit = document.getElementById("kit-submit");
const authorsList = document.getElementById("authors-list");
const addAuthorBtn = document.getElementById("add-author");
const dlKit = document.getElementById("dl-kit");

const playBtn = document.getElementById("play-btn");
const stopBtn = document.getElementById("stop-btn");
const originalAudio = document.getElementById("original-audio");

let osmd = null;
let currentJobId = null;
let currentResult = null;

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
  currentJobId = job.id;
  currentResult = result;
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
  renderLyrics(result);
  showKitCard(result);
  setupOriginalAudio(job.id, result);
  stopPlayback();
  await renderScore(job.id);
}

function setupOriginalAudio(jobId, result) {
  if (result.master_path) {
    originalAudio.src = `/api/audio/${jobId}`;
    originalAudio.hidden = false;
  } else {
    originalAudio.removeAttribute("src");
    originalAudio.hidden = true;
  }
}

// ---- Transcription playback: a tiny WebAudio synth over the detected note
// events, with the OSMD cursor following the score in time. ----

let playback = null; // { ctx, timer, t0, duration }

function midiToFreq(m) {
  return 440 * Math.pow(2, (m - 69) / 12);
}

function stopPlayback() {
  if (!playback) {
    stopBtn.disabled = true;
    playBtn.disabled = !currentResult;
    return;
  }
  clearInterval(playback.timer);
  playback.ctx.close().catch(() => {});
  playback = null;
  playBtn.disabled = false;
  stopBtn.disabled = true;
  if (osmd && osmd.cursor) {
    try {
      osmd.cursor.reset();
      osmd.cursor.hide();
    } catch (_e) {
      /* cursor may not be initialized when rendering failed */
    }
  }
}

function startPlayback() {
  const notes = (currentResult && currentResult.notes) || [];
  if (!notes.length) return;

  const AudioCtx = window.AudioContext || window.webkitAudioContext;
  const ctx = new AudioCtx();
  const t0 = ctx.currentTime + 0.15;
  const duration = Math.max(...notes.map((n) => n.end)) + 0.5;

  const master = ctx.createGain();
  master.gain.value = 0.9;
  master.connect(ctx.destination);

  notes.forEach((n) => {
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    const vol = 0.25 * ((n.velocity || 90) / 127);
    const start = t0 + n.start;
    const end = t0 + Math.max(n.end, n.start + 0.05);

    osc.type = "triangle";
    osc.frequency.value = midiToFreq(n.pitch_midi);
    gain.gain.setValueAtTime(0, start);
    gain.gain.linearRampToValueAtTime(vol, start + 0.015);
    gain.gain.setValueAtTime(vol, Math.max(start + 0.015, end - 0.06));
    gain.gain.linearRampToValueAtTime(0, end);
    osc.connect(gain).connect(master);
    osc.start(start);
    osc.stop(end + 0.05);
  });

  // Cursor sync: OSMD iterator timestamps are in whole-note units;
  // seconds-per-whole-note = 4 beats * 60 / bpm.
  let cursorReady = false;
  const bpm = currentResult.tempo_bpm || 120;
  const secondsPerWhole = 240 / bpm;
  if (osmd && osmd.cursor) {
    try {
      osmd.cursor.reset();
      osmd.cursor.show();
      cursorReady = true;
    } catch (_e) {
      cursorReady = false;
    }
  }

  const timer = setInterval(() => {
    if (!playback) return;
    const t = ctx.currentTime - t0;
    if (cursorReady) {
      try {
        const it = osmd.cursor.Iterator;
        while (!it.EndReached && it.currentTimeStamp.RealValue * secondsPerWhole <= t) {
          osmd.cursor.next();
        }
      } catch (_e) {
        cursorReady = false;
      }
    }
    if (t >= duration) stopPlayback();
  }, 50);

  playback = { ctx, timer, t0, duration };
  playBtn.disabled = true;
  stopBtn.disabled = false;
}

playBtn.addEventListener("click", startPlayback);
stopBtn.addEventListener("click", stopPlayback);

function renderLyrics(result) {
  if (result.lyrics && result.lyrics.text) {
    show(lyricsCard);
    lyricsText.value = result.lyrics.text;
    const prob = Math.round((result.lyrics.language_probability || 0) * 100);
    lyricsLang.textContent = `Idioma detectado: ${result.lyrics.language || "?"} (${prob}% de confiança) · modelo ${result.lyrics.model_size}`;
    lyricsLang.classList.remove("error");
  } else if (result.lyrics_error) {
    show(lyricsCard);
    lyricsText.value = "";
    lyricsLang.textContent = `Letra não transcrita: ${result.lyrics_error}`;
    lyricsLang.classList.add("error");
  } else {
    hide(lyricsCard);
    lyricsText.value = "";
  }
}

function showKitCard(result) {
  show(kitCard);
  dlKit.hidden = true;
  kitStatus.textContent = "";

  const titleInput = form.querySelector('input[name="title"]');
  const fileName = audioInput.files && audioInput.files[0] ? audioInput.files[0].name : "";
  const fallback = fileName.replace(/\.[^.]+$/, "");
  const kitTitle = document.getElementById("kit-title");
  if (!kitTitle.value) {
    kitTitle.value = (titleInput && titleInput.value.trim()) || fallback;
  }
  const year = new Date().getFullYear();
  const kitYear = document.getElementById("kit-year");
  const kitRecYear = document.getElementById("kit-rec-year");
  if (!kitYear.value) kitYear.value = year;
  if (!kitRecYear.value) kitRecYear.value = year;

  if (!authorsList.querySelector(".author-row")) {
    addAuthorRow();
  }
}

function addAuthorRow() {
  const row = document.createElement("div");
  row.className = "author-row";
  row.innerHTML = `
    <label>Nome<input type="text" data-f="name" placeholder="nome completo" /></label>
    <label>Pseudônimo<input type="text" data-f="pseudonym" /></label>
    <label>CPF<input type="text" data-f="cpf" placeholder="000.000.000-00" /></label>
    <label>Função
      <select data-f="role">
        <option value="letra e música" selected>Letra e música</option>
        <option value="letra">Letra</option>
        <option value="música">Música</option>
      </select>
    </label>
    <label>%<input type="number" data-f="share" min="0" max="100" step="0.5" value="100" /></label>
    <label>Associação
      <select data-f="association">
        <option value="">Nenhuma</option>
        <option value="UBC">UBC</option>
        <option value="ABRAMUS">ABRAMUS</option>
        <option value="AMAR">AMAR</option>
        <option value="SBACEM">SBACEM</option>
        <option value="SICAM">SICAM</option>
        <option value="Outra">Outra</option>
      </select>
    </label>
    <button type="button" class="remove" title="remover autor">✕</button>
  `;
  row.querySelector("button.remove").addEventListener("click", () => {
    if (authorsList.querySelectorAll(".author-row").length > 1) {
      row.remove();
    } else {
      row.querySelectorAll("input").forEach((i) => (i.value = i.dataset.f === "share" ? "100" : ""));
    }
  });
  authorsList.appendChild(row);
}

addAuthorBtn.addEventListener("click", addAuthorRow);

function collectAuthors() {
  return [...authorsList.querySelectorAll(".author-row")]
    .map((row) => ({
      name: row.querySelector('[data-f="name"]').value.trim(),
      pseudonym: row.querySelector('[data-f="pseudonym"]').value.trim(),
      cpf: row.querySelector('[data-f="cpf"]').value.trim(),
      role: row.querySelector('[data-f="role"]').value,
      share_percent: parseFloat(row.querySelector('[data-f="share"]').value) || 0,
      association: row.querySelector('[data-f="association"]').value,
    }))
    .filter((a) => a.name);
}

kitForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  if (!currentJobId) return;

  const authors = collectAuthors();
  if (!authors.length) {
    kitStatus.textContent = "Informe ao menos um autor com nome.";
    kitStatus.classList.add("error");
    return;
  }

  const participants = document
    .getElementById("kit-participants")
    .value.split(",")
    .map((s) => s.trim())
    .filter(Boolean)
    .map((name) => ({ name, role: "participação" }));

  const payload = {
    title: document.getElementById("kit-title").value.trim(),
    subtitle: document.getElementById("kit-subtitle").value.trim(),
    genre: document.getElementById("kit-genre").value.trim(),
    language: document.getElementById("kit-language").value.trim() || "pt",
    year: parseInt(document.getElementById("kit-year").value, 10) || null,
    lyrics: lyricsText.value,
    authors,
    main_performer: document.getElementById("kit-performer").value.trim(),
    performers: participants,
    producer: document.getElementById("kit-producer").value.trim(),
    recording_year: parseInt(document.getElementById("kit-rec-year").value, 10) || null,
    recording_location: document.getElementById("kit-rec-location").value.trim(),
    isrc: document.getElementById("kit-isrc").value.trim(),
    duration_seconds: currentResult ? currentResult.duration_seconds : null,
  };

  kitSubmit.disabled = true;
  kitStatus.classList.remove("error");
  kitStatus.textContent = "Gerando kit…";
  dlKit.hidden = true;

  try {
    const res = await fetch(`/api/kit/${currentJobId}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const json = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(json.error || `HTTP ${res.status}`);
    dlKit.href = json.kit_url;
    dlKit.hidden = false;
    dlKit.setAttribute("download", "");
    kitStatus.textContent = "Kit pronto — bom registro!";
  } catch (err) {
    kitStatus.textContent = "Erro: " + err.message;
    kitStatus.classList.add("error");
  } finally {
    kitSubmit.disabled = false;
  }
});

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
