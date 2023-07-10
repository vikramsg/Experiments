FROM python:3.8-slim

# Add the path for poetry
ENV PATH $PATH:/root/.local/bin/

RUN apt-get update && apt-get install -y curl && \
    # Install & Configure poetry
    python3 -m pip install --upgrade pip && \
    curl -sSL https://install.python-poetry.org | python3 - --version 1.4.2 && \
    poetry config virtualenvs.create false

COPY poetry.lock pyproject.toml /app/
COPY ./src /app/src
WORKDIR /app
RUN --mount=type=cache,target=/home/.cache/pypoetry/cache \
    --mount=type=cache,target=/home/.cache/pypoetry/artifacts \ 
    poetry install --only main
ENV PYTHONPATH /app

# Expose the port on which the application will run (default: 8000)
EXPOSE 8000