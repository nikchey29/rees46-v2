FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --no-dev

COPY configs ./configs
COPY results ./results

EXPOSE 8000
CMD ["uv", "run", "uvicorn", "rees46.serving.api:app", "--host", "0.0.0.0", "--port", "8000"]
