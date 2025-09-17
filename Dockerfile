# syntax=docker/dockerfile:1
FROM python:3.11-slim

# System deps
RUN apt-get update -y && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install supercronic
ENV SUPERCRONIC_URL=https://github.com/aptible/supercronic/releases/download/v0.2.31/supercronic-linux-amd64 \
    SUPERCRONIC=/usr/local/bin/supercronic
RUN curl -fsSLo ${SUPERCRONIC} ${SUPERCRONIC_URL} \
    && chmod +x ${SUPERCRONIC}

WORKDIR /app

# Copy only files needed to install deps first (better cache)
COPY pyproject.toml README.md /app/
COPY quant_strategies /app/quant_strategies
COPY scripts /app/scripts
COPY crontab /app/crontab

# Install project with notebook extras for plotting/analysis
RUN pip install --upgrade pip \
    && pip install .[notebook]

# Create data and logs directory (data typically mounted from host)
RUN mkdir -p /app/data /app/logs

# Default command: run supercronic with our crontab
CMD ["/usr/local/bin/supercronic", "/app/crontab"]
