# ── Stage 1: Build dependencies ─────────────────────────────────
FROM python:3.13-slim AS builder

WORKDIR /app

# Install build-time system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ── Stage 2: Runtime ────────────────────────────────────────────
FROM python:3.13-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY app/ ./app/

# Expose the API port
EXPOSE 8000

# Health check — hits /health every 30 s
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Run the API (model downloads on first startup if not cached)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
