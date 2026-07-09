FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
      libsndfile1 \
      ffmpeg \
      lilypond \
    && rm -rf /var/lib/apt/lists/*

# Non-root user (required by Hugging Face Spaces; good practice elsewhere).
# -m gives it a writable HOME for the Whisper/Basic Pitch model caches.
RUN useradd -m -u 1000 musica

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN mkdir -p uploads output && chown -R musica:musica /app

USER musica
ENV HOME=/home/musica

# HF Spaces injects PORT=7860; default to 5000 for local docker runs.
ENV PORT=5000
EXPOSE 5000 7860

# Single worker: the job queue lives in process memory. gthread keeps the
# HTTP endpoints responsive while transcription runs on background threads.
CMD ["sh", "-c", "gunicorn -w 1 -k gthread --threads 8 --timeout 300 -b 0.0.0.0:${PORT:-5000} app:app"]
