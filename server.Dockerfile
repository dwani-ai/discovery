# Stage 1: Build stage
FROM python:3.10-alpine AS builder

WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install build dependencies
RUN apk add --no-cache \
    gcc \
    musl-dev \
    curl \
    libjpeg-turbo-dev \
    zlib-dev \
    libpng-dev poppler-utils

# Install Python dependencies
COPY server-requirements.txt .
RUN pip install --no-cache-dir --user -r server-requirements.txt

# Stage 2: Final stage
FROM python:3.10-alpine

WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install runtime dependencies only
RUN apk add --no-cache \
    libjpeg-turbo \
    zlib \
    libpng poppler-utils\
    && rm -rf /var/cache/apk/*

# Copy installed Python dependencies from builder stage
COPY --from=builder /root/.local /home/appuser/.local

# Copy the application code
COPY . .

# Create appuser and set permissions for /app and /data
RUN adduser -D appuser \
    && mkdir -p /data \
    && chown -R appuser:appuser /app /data

RUN pip install uvicorn
USER appuser
EXPOSE 80

# Command to run the Gradio program
CMD ["uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "18889"]