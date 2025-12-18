# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies for pdf2image (poppler-utils)
RUN apt-get update && apt-get install -y \
    poppler-utils \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Expose port 8000 for the FastAPI server
EXPOSE 18888

# Command to run the FastAPI application
CMD ["uvicorn", "src.server.main:app", "--host", "0.0.0.0", "--port", "18888"]