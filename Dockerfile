FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# libgomp1 is the OpenMP runtime. The CPU inference wheels are prebuilt, so no compiler is
# needed, but slim images do not always ship this library. (Precaution: see docs/deployment.md.)
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app ./app
COPY web ./web
COPY scripts ./scripts

# Models are NOT baked into the image: they are large and carry their own licenses, so they are
# downloaded on first start into a volume. See docs/deployment.md.
RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /app/models \
    && chown -R appuser:appuser /app/models
USER appuser
VOLUME /app/models

EXPOSE 8000

# /health answers 200 as soon as the process is up (even while models load), so the start
# period only has to cover interpreter start-up.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"]

# Exec form so uvicorn is PID 1 and receives SIGTERM for a graceful shutdown.
CMD ["uvicorn", "app.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000", "--timeout-graceful-shutdown", "10"]
