# Musica — transcrição de áudio para partitura

App web que recebe um arquivo de áudio (WAV/MP3/FLAC/OGG/M4A) e devolve a
partitura correspondente em **MusicXML**, **MIDI** e (opcionalmente) **PDF**,
renderizando a partitura diretamente no navegador.

A transcrição usa o [Basic Pitch](https://github.com/spotify/basic-pitch) da
Spotify — um modelo CNN leve, polifônico e multi-instrumento que é um dos
melhores em código aberto hoje. A saída MIDI é convertida em partitura via
[music21](https://www.music21.org/) e renderizada no navegador com
[OpenSheetMusicDisplay](https://opensheetmusicdisplay.org/).

## Por que não é "perfeito"

Transcrição automática perfeita é um problema em aberto na pesquisa em MIR
(Music Information Retrieval). Mesmo modelos comerciais erram em:

- harmonias densas (acordes com várias notas próximas);
- instrumentos com vibrato/portamento intenso (voz, violino);
- percussão não-afinada e gravações com ruído;
- ritmos muito sincopados ou em compassos irregulares.

Para o melhor resultado, use gravações **limpas, monoinstrumentais e bem
afinadas**. O app expõe os parâmetros do Basic Pitch (limiares de onset/frame,
duração mínima, faixa de frequência) para que você refine o resultado.

## Como usar

### Pré-requisitos

- Python 3.9–3.11 (Basic Pitch / TensorFlow não suportam 3.12+ ainda)
- (opcional) [Lilypond](https://lilypond.org/) para renderização em PDF

### Instalação

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Servidor web

```bash
python app.py
# abre http://localhost:5000
```

Healthcheck: `GET /healthz` → `{"status": "ok"}`.

### Docker

```bash
docker build -t musica .
docker run --rm -p 5000:5000 musica
```

A imagem inclui Lilypond, então o PDF é gerado out-of-the-box.

### Linha de comando

```bash
python transcribe.py minha_musica.mp3 -o ./output --title "Minha música"
```

Gera `output/minha_musica.mid`, `output/minha_musica.musicxml` e (se Lilypond
estiver instalado) `output/minha_musica.pdf`.

## Endpoints HTTP

| Método | Rota                          | Função                                  |
| ------ | ----------------------------- | --------------------------------------- |
| GET    | `/`                           | Interface web                           |
| POST   | `/api/transcribe`             | Recebe `audio` (multipart) → `job_id`   |
| GET    | `/api/job/<job_id>`           | Status do trabalho (`pending`/`running`/`done`/`error`) |
| GET    | `/api/download/<job_id>/<fmt>`| `fmt` ∈ `midi` · `musicxml` · `pdf`     |

Parâmetros opcionais do POST `/api/transcribe` (form-data):

- `title` — título da partitura
- `onset_threshold` — limiar de detecção de onset (0–1, default 0.5)
- `frame_threshold` — limiar de sustentação (0–1, default 0.3)
- `min_note_length_ms` — duração mínima de uma nota (default 58 ms)
- `min_freq` / `max_freq` — faixa de pitch em Hz

## Arquitetura

```
audio  ─▶  Basic Pitch (CNN)  ─▶  MIDI  ─▶  music21 (quantização)  ─▶  MusicXML
                                                   │
                                                   ├─▶  Lilypond  ─▶  PDF
                                                   └─▶  OpenSheetMusicDisplay (navegador)
```

## Testes

```bash
pip install pytest
pytest
```

Os testes não exercitam o modelo de ML (lento + heavy deps); cobrem a camada
HTTP e o parsing/serialização. Para um teste end-to-end manual, use o arquivo
`examples/scale.wav` (se presente).

## Limitações conhecidas

- O app é stateful em memória — em produção, use Redis/banco para a fila.
- Não há autenticação; não exponha à internet sem proxy.
- O modelo Basic Pitch baixa pesos (~17 MB) na primeira execução.
- Tracks de bateria não são transcritas (modelo só prevê notas afinadas).
