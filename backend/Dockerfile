# ============================
# Stage 1: Builder
# ============================
FROM python:3.10-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# System dependencies needed for building + pdf2image/poppler
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    poppler-utils \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*


RUN pip install --no-cache-dir --prefix=/install \
    --index-url https://download.pytorch.org/whl/cpu \
    --no-deps \
    torch==2.9.1+cpu torchvision==0.24.1+cpu torchaudio==2.9.1+cpu

COPY requirements.txt .

# Then install everything else normally
RUN pip install --no-deps --no-cache-dir --prefix=/install --ignore-installed -r requirements.txt

# ============================
# Stage 2: Runtime
# ============================
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Runtime deps only
RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Optional: non-root user (good practice)
#RUN useradd -m -u 1000 appuser
#USER appuser

# Copy pure python packages from builder
COPY --from=builder /install /usr/local

# Copy your application code
COPY . .

EXPOSE 18888

CMD ["uvicorn", "src.server.main:app", "--host", "0.0.0.0", "--port", "18888"]