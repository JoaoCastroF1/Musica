# Musica — do áudio ao registro da sua música

App web que recebe um arquivo de áudio (WAV/MP3/FLAC/OGG/M4A) e devolve:

1. **Partitura** em MusicXML, MIDI e (opcionalmente) PDF, renderizada no
   navegador;
2. **Letra transcrita** (Whisper) em rascunho editável;
3. **Kit de registro** (.zip) com tudo o que você precisa para registrar a
   obra e o fonograma no Brasil: dossiê em PDF, letra, partitura, áudio
   master, hashes SHA-256 de integridade, metadados estruturados e um guia
   passo a passo (Biblioteca Nacional/EDA, associações UBC/ABRAMUS/AMAR/
   SBACEM/SICAM, ECAD e ISRC).

A transcrição musical usa o [Basic Pitch](https://github.com/spotify/basic-pitch)
da Spotify — um modelo CNN leve, polifônico e multi-instrumento. A saída MIDI
é convertida em partitura via [music21](https://www.music21.org/) e renderizada
no navegador com [OpenSheetMusicDisplay](https://opensheetmusicdisplay.org/).
A letra usa [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
(Whisper em CTranslate2, roda bem em CPU).

## Por que não é "perfeito"

Transcrição automática perfeita é um problema em aberto na pesquisa em MIR
(Music Information Retrieval). Mesmo modelos comerciais erram em:

- harmonias densas (acordes com várias notas próximas);
- instrumentos com vibrato/portamento intenso (voz, violino);
- percussão não-afinada e gravações com ruído;
- ritmos muito sincopados ou em compassos irregulares;
- letra cantada sobre instrumental alto (para o Whisper).

Para o melhor resultado, use gravações **limpas, monoinstrumentais e bem
afinadas**. O app expõe os parâmetros do Basic Pitch e do Whisper para você
refinar, e a letra é sempre apresentada como rascunho editável — revise antes
de registrar.

## Como acessar o site

O app é auto-hospedado: você o roda no seu computador ou publica numa nuvem.
Três caminhos, do mais simples ao mais completo:

### Opção A — rodar no seu computador (Docker, recomendado)

1. Instale o [Docker Desktop](https://www.docker.com/products/docker-desktop/)
   (Windows/Mac) ou `docker` (Linux).
2. No terminal:

   ```bash
   git clone https://github.com/JoaoCastroF1/Musica.git
   cd Musica
   docker build -t musica .
   docker run --rm -p 5000:5000 musica
   ```

3. Abra **http://localhost:5000** no navegador. Pronto — a imagem já inclui
   Lilypond (PDF da partitura) e ffmpeg.

### Opção B — rodar com Python direto

Pré-requisitos: Python 3.9–3.11 (Basic Pitch/TensorFlow ainda não suportam
3.12+) e, opcionalmente, [Lilypond](https://lilypond.org/) para o PDF.

```bash
git clone https://github.com/JoaoCastroF1/Musica.git
cd Musica
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt  # demora alguns minutos (TensorFlow)
python app.py
# abra http://localhost:5000
```

### Opção C — publicar na internet de graça (Hugging Face Spaces)

Para ter um **site com URL pública** sem pagar servidor:

1. Crie uma conta gratuita em [huggingface.co](https://huggingface.co).
2. Clique em **New Space** → nome (ex.: `musica`) → SDK: **Docker** →
   visibilidade pública ou privada → Create.
3. Envie os arquivos deste repositório para o Space (via git ou upload na
   própria página; o `Dockerfile` já está pronto — o Space usa a porta
   definida em `PORT` automaticamente).
4. Aguarde o build (~10 min na primeira vez). Seu site fica em
   `https://huggingface.co/spaces/<seu-usuario>/musica`.

Dicas para o plano gratuito (2 vCPU): use o modelo Whisper `tiny` ou `base`
para a letra, e áudios de até ~3 minutos para transcrições rápidas. Na
primeira letra transcrita o Space baixa o modelo Whisper (fica em cache).

Healthcheck: `GET /healthz` → `{"status": "ok"}`.

Em produção o container usa gunicorn (1 worker + threads — a fila de
trabalhos vive na memória do processo).

### Linha de comando

```bash
python transcribe.py minha_musica.mp3 -o ./output --title "Minha música"
python lyrics.py minha_musica.mp3 --model small --language pt
```

## Fluxo completo: da gravação ao royalty

1. **Envie o áudio** — o app gera partitura, detecta tom/tempo/compasso e
   transcreve a letra.
2. **Confira de ouvido** — o botão "▶ Tocar transcrição" toca as notas
   detectadas com o cursor acompanhando a partitura; o player logo abaixo
   toca o áudio original para comparação A/B.
3. **Revise a letra** no editor (as correções entram no kit).
3. **Preencha os dados** da obra (autores, CPF, percentuais, associação) e do
   fonograma (intérprete, produtor, ano/local de gravação).
4. **Gere o kit (.zip)** e siga o guia incluído:
   - **Biblioteca Nacional (EDA)** — registro declaratório da obra (letra +
     partitura) via gov.br: prova de autoria e anterioridade;
   - **Associação de gestão coletiva** (UBC, ABRAMUS, AMAR, SBACEM, SICAM) —
     filie-se e cadastre a obra (autores/percentuais) e o fonograma;
   - **ECAD** — recebe os cadastros via associação e distribui os royalties de
     execução pública (rádio, TV, shows, streaming);
   - **ISRC** — código do fonograma, solicitado pela associação/produtor;
   - **Distribuidora digital** — para streaming, com o mesmo ISRC.

> O kit organiza informações e evidências técnicas (incluindo hashes SHA-256
> dos arquivos). Ele **não substitui o registro oficial** nem constitui
> aconselhamento jurídico.

## Endpoints HTTP

| Método | Rota                           | Função                                  |
| ------ | ------------------------------ | --------------------------------------- |
| GET    | `/`                            | Interface web                           |
| POST   | `/api/transcribe`              | Recebe `audio` (multipart) → `job_id`   |
| GET    | `/api/job/<job_id>`            | Status (`pending`/`running`/`done`/`error`) |
| POST   | `/api/kit/<job_id>`            | Gera o kit de registro (corpo JSON)     |
| GET    | `/api/download/<job_id>/<fmt>` | `fmt` ∈ `midi` · `musicxml` · `pdf` · `kit` |

Parâmetros opcionais do POST `/api/transcribe` (form-data):

- `title` — título da partitura
- `onset_threshold` — limiar de detecção de onset (0–1, default 0.5)
- `frame_threshold` — limiar de sustentação (0–1, default 0.3)
- `min_note_length_ms` — duração mínima de uma nota (default 58 ms)
- `min_freq` / `max_freq` — faixa de pitch em Hz
- `min_confidence` — descarta notas abaixo desta confiança (0–1, default 0.5)
- `merge_gap_ms` — funde notas adjacentes do mesmo pitch (default 50 ms)
- `bpm_override` — força BPM em vez de detectar
- `key_override` — força tom (`G`, `f# minor`, etc.)
- `time_signature_override` — força compasso (`4/4`, `3/4`, `2/4`, `6/8`)
- `transcribe_lyrics` — `on`/`true`/`1` para transcrever a letra
- `whisper_model` — `tiny` · `base` · `small` (default) · `medium`
- `lyrics_language` — força idioma (ex.: `pt`); vazio = detectar

Corpo JSON do POST `/api/kit/<job_id>`:

```json
{
  "title": "Minha Canção",
  "subtitle": "", "genre": "MPB", "language": "pt", "year": 2026,
  "lyrics": "letra revisada...",
  "authors": [
    {"name": "João Castro", "pseudonym": "", "cpf": "000.000.000-00",
     "role": "letra e música", "share_percent": 100, "association": "UBC"}
  ],
  "main_performer": "João Castro",
  "performers": [{"name": "Ana Souza", "role": "participação"}],
  "producer": "Estúdio X",
  "recording_year": 2026, "recording_location": "São Paulo/SP",
  "isrc": "", "duration_seconds": 183.5
}
```

Resposta: `{"kit_url": "/api/download/<job_id>/kit"}`.

## Arquitetura

```
audio ─▶ Basic Pitch (CNN) ─▶ note events ─▶ filtro de confiança ─▶ MIDI
  │                                                                  │
  │        librosa (BPM) + music21 (tom) + heurística (compasso) ────┤
  │                                                                  ▼
  │                                            music21 Score ─▶ MusicXML/PDF
  │
  ├─▶ faster-whisper ─▶ letra (rascunho editável)
  │
  └─▶ master + SHA-256 ─┐
                        ▼
              registration.py ─▶ kit .zip
              (dossiê PDF, letra, partitura, metadados,
               guia BN/associação/ECAD/ISRC)
```

O master do áudio é mantido em `output/` (`<job_id>_master.<ext>`) para
integrar o kit como evidência do fonograma.

## Testes

```bash
pip install pytest flask reportlab
pytest
```

Os testes não exercitam os modelos de ML: a camada HTTP usa um stub do
pipeline (ver `tests/conftest.py`), o Whisper é simulado em
`tests/test_lyrics.py` e o gerador de kit é testado de ponta a ponta com
artefatos sintéticos em `tests/test_registration.py`. Testes que exigem as
deps pesadas se auto-desativam quando elas não estão instaladas.

## Limitações conhecidas

- O app é stateful em memória — em produção, use Redis/banco para a fila.
- Não há autenticação; não exponha à internet sem proxy.
- O modelo Basic Pitch baixa pesos (~17 MB) na primeira execução; o Whisper
  baixa o modelo escolhido (~460 MB no `small`) na primeira letra.
- Tracks de bateria não são transcritas (modelo só prevê notas afinadas).
- A letra transcrita é um rascunho: revise sempre antes do registro.
