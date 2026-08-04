FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt requirements-runtime.lock ./
RUN python -m pip install --upgrade pip \
    && pip install -r requirements-runtime.lock

FROM base AS test

RUN apt-get update \
    && apt-get install -y --no-install-recommends git nodejs \
    && rm -rf /var/lib/apt/lists/*
COPY requirements-dev.txt pyproject.toml ./
RUN pip install -r requirements-dev.txt
COPY personalityrag ./personalityrag
COPY static ./static
COPY assets ./assets
COPY tests ./tests
COPY tools ./tools
COPY run.py ./run.py
COPY Dockerfile ./Dockerfile
COPY docker ./docker
CMD ["python", "-m", "pytest", "-q"]

FROM base AS runtime

ARG PERSONALITYRAG_VERSION=v0.1.2
ARG VCS_REF=unknown
LABEL org.opencontainers.image.title="PersonalityRAG Linux" \
      org.opencontainers.image.version="${PERSONALITYRAG_VERSION}" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.source="https://github.com/Creeper3222/PersonalityRAG/tree/Linux-Docker" \
      io.personalityrag.platform="linux-docker"

COPY personalityrag ./personalityrag
COPY static ./static
COPY assets ./assets
COPY run.py ./run.py
COPY docker/entrypoint.sh /usr/local/bin/personalityrag-entrypoint.sh

RUN chmod +x /usr/local/bin/personalityrag-entrypoint.sh \
    && mkdir -p /app/state/config /app/state/data \
    && python -m compileall -q /app/personalityrag /app/run.py

EXPOSE 8765 8766
VOLUME ["/app/state"]
HEALTHCHECK --interval=20s --timeout=5s --start-period=30s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8765/api/v1/health', timeout=3)" || exit 1

ENTRYPOINT ["/usr/local/bin/personalityrag-entrypoint.sh"]
CMD []
