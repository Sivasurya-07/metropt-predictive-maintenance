# Multi-Stage Production Dockerfile for MetroPT APU System
FROM python:3.11-slim AS builder

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency specifications (we use the slimmed production requirements)
COPY requirements.txt ./

# Install wheels
RUN pip install --no-cache-dir --upgrade pip && \
    pip wheel --no-cache-dir --wheel-dir /app/wheels -r requirements.txt

# Final runtime image
FROM python:3.11-slim AS runner

WORKDIR /app

# Install system dependencies required for ML libraries (OpenMP)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install runtime dependencies only
COPY --from=builder /app/wheels /wheels
COPY --from=builder /app/requirements.txt .
RUN pip install --no-cache-dir --no-index --find-links=/wheels -r requirements.txt

# Copy project source code
COPY src/ ./src/
COPY models/ ./models/
RUN mkdir -p data/raw data/processed

EXPOSE 8000

# Run with raw Uvicorn to prevent Gunicorn's 30s boot timeout from killing the worker on slow free tier CPUs
CMD uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT:-8000}
