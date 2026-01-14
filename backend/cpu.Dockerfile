# ============================
# Stage 1: Builder
# ============================
FROM python:3.9-slim AS builder

# Prevent Python from writing .pyc files and enable unbuffered logs
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# System deps for building and pdf2image (poppler)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    poppler-utils \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements (CPU-only requirements file)
COPY cpu-requirements.txt .

# Install Python dependencies into a dedicated prefix
RUN pip install --no-cache-dir --prefix=/install -r cpu-requirements.txt

# ============================
# Stage 2: Runtime
# ============================
FROM python:3.9-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Only runtime system deps (no compiler)
RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Optional: create non-root user
RUN useradd -m appuser
USER appuser

# Copy installed Python packages from builder stage
COPY --from=builder /install /usr/local

# Copy application code
COPY . .

# Expose FastAPI port
EXPOSE 18888

# Run FastAPI with Uvicorn
CMD ["uvicorn", "src.server.main:app", "--host", "0.0.0.0", "--port", "18888"]
