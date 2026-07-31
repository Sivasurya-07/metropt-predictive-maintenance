# Multi-Stage Production Dockerfile for MetroPT APU System
FROM python:3.11-slim AS builder

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency specifications
COPY requirements.txt ./

# Install wheels (Install CPU-only PyTorch to shrink image from 5GB to 1GB)
RUN pip install --no-cache-dir --upgrade pip && \
    pip wheel --no-cache-dir --wheel-dir /app/wheels --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

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

# Copy project source code (ARG busts cache so Railway always picks up latest)
ARG CACHEBUST=1
COPY src/ ./src/
COPY models/ ./models/
RUN mkdir -p data/raw data/processed

# Limit threads to prevent deadlocks on strict CPU limits
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1

EXPOSE 8000

# Run with Uvicorn. Use sh -c form to allow Railway/Render to inject the random $PORT variable.
CMD uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT:-8000}
