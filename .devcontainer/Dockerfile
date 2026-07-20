FROM mcr.microsoft.com/devcontainers/rust:2-1-bookworm

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libsqlite3-dev \
        python3 \
        sqlite3 \
    && rm -rf /var/lib/apt/lists/*
