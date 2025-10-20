FROM python:3.12-slim

ENV POETRY_VERSION=2.2.1 \
    POETRY_HOME=/opt/poetry \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=off \
    JAX_PLATFORMS=cpu

ENV PATH="$POETRY_HOME/bin:$PATH"

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        curl \
        libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

RUN curl -sSL https://install.python-poetry.org | python3 - --version "$POETRY_VERSION"

WORKDIR /app

COPY pyproject.toml ./

RUN poetry install --no-root --no-interaction --no-ansi

COPY . .

CMD ["poetry", "run", "python", "analyse_models.py", "--help"]
