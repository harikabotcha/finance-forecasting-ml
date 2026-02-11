# Multi-stage build for Finance Forecasting ML
# Stage 1: Build dependencies
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Stage 2: Production image
FROM python:3.11-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Create non-root user
RUN useradd --create-home --shell /bin/bash appuser

# Copy application code
COPY . .

# Create data directories
RUN mkdir -p data/raw data/processed data/models logs && \
    chown -R appuser:appuser /app

USER appuser

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Default command: run the API server
CMD ["python", "-m", "uvicorn", "src.forecast.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
