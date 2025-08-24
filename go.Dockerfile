# Use multi-stage build for smaller final image

# Builder stage
FROM golang:1.22-bookworm AS builder

# Install system dependencies for go-fitz
RUN apt-get update && apt-get install -y libmupdf-dev pkg-config

# Set working directory
WORKDIR /app

# Copy the Go source code (assume the provided code is in main.go, with fixes for missing imports: add "image", "io/ioutil", "mime/multipart")
COPY main.go .

# Initialize go module and fetch dependencies
RUN go mod init app
RUN go mod tidy

# Build the binary with required tags for external MuPDF
RUN go build -tags extlib,pkgconfig -o server main.go

# Runtime stage
FROM debian:bookworm-slim

# Install runtime dependencies (libmupdf-dev includes the shared library; also ca-certificates for HTTPS calls)
RUN apt-get update && apt-get install -y libmupdf-dev ca-certificates && rm -rf /var/lib/apt/lists/*

# Copy the built binary from builder
COPY --from=builder /app/server /usr/local/bin/server

# Expose the port the app runs on
EXPOSE 8080

# Set environment variable if needed (e.g., VLLM_IP, default is 0.0.0.0)
ENV VLLM_IP=0.0.0.0

# Run the server
CMD ["server"]