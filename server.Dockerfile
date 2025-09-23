# Use official Python slim image as base
FROM python:3.10-slim AS builder

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY server-requirements.txt .
RUN pip install --no-cache-dir -r server-requirements.txt

# Final stage
FROM python:3.10-slim

# Install Nginx and supervisor
RUN apt-get update && apt-get install -y \
    nginx \
    supervisor \
    && rm -rf /var/lib/apt/lists/*

# Copy Python dependencies from builder
COPY --from=builder /usr/local /usr/local

# Set working directory
WORKDIR /app

# Create appuser and set permissions
RUN useradd -ms /bin/bash appuser \
    && mkdir -p /data \
    && chown -R appuser:appuser /app /data /var/log/nginx /var/lib/nginx

# Copy application code
COPY server/ /app/src/

# Copy Nginx configuration
COPY nginx/nginx.conf /etc/nginx/nginx.conf

# Expose port 8080 internally (map to 80 on host via -p 80:8080)
EXPOSE 18889

# Copy supervisor configuration
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Run as non-root user
USER appuser

# Command to run supervisor (manages Nginx and Uvicorn)
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]