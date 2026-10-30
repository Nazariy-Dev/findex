# syntax=docker/dockerfile:1
# Етап 1: залежності з uv.lock, потім сам пакет
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS builder
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=0
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --locked --no-install-project --no-dev --extra semantic
COPY README.md ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv uv sync --locked --no-dev --extra semantic

# Етап 2: лише готове оточення, без uv і компіляторів
FROM python:3.13-slim-bookworm
RUN useradd --create-home --uid 10001 findex
WORKDIR /app
COPY --from=builder --chown=findex:findex /app /app
# індекс і корпус кладуться в образ на етапі деплою (див. README); шлях — зі змінної
COPY --chown=findex:findex deploy/ /app/deploy/
ENV PATH="/app/.venv/bin:$PATH" INDEX_PATH=/app/deploy/index.bin PORT=8000
USER findex
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request,os;urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/health')"
CMD ["sh", "-c", "uvicorn findex.web:app --host 0.0.0.0 --port ${PORT} --proxy-headers"]
