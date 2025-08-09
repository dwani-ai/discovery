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
    libpng-dev

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

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
    libpng \
    && rm -rf /var/cache/apk/*

# Copy installed Python dependencies from builder stage
COPY --from=builder /root/.local /home/appuser/.local

# Copy the application code
COPY . .

# Create appuser and set permissions for /app and /data
RUN adduser -D appuser \
    && mkdir -p /data \
    && chown -R appuser:appuser /app /data

USER appuser
EXPOSE 80

# Command to run the Gradio program
CMD ["python", "workshop_demo.py"]