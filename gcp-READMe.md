
### Core Replacements

| Original Component | GCP/Vertex AI Replacement |
|--------------------|---------------------------|
| **ChromaDB + bge-small embeddings** | Vertex AI Text Embeddings (text-embedding-004) + ChunkRecord table in AlloyDB/Cloud SQL |
| **pdf2image + OpenAI vision OCR** | Gemini 1.5 Flash multimodal (native PDF ingestion) |
| **AsyncOpenAI chat (gemma3 proxy)** | Gemini 1.5 Pro via Vertex AI SDK |
| **dwani Audio API (custom TTS)** | Google Cloud Text-to-Speech |
| **SQLite** | Cloud SQL PostgreSQL with Cloud SQL Connector |
| **Local file storage** | Cloud Storage (GCS) buckets |
| **fpdf regeneration** | Gemini text extraction (clean PDFs now via GCS signed URLs) |

### Key Features Preserved

✅ **Hybrid RAG**: Semantic search (Vertex embeddings + cosine similarity) + BM25 lexical, fused with RRF  
✅ **Multi-document chat**: Cross-file contradiction detection with Gemini  
✅ **NotebookLM-style podcasts**: Gemini script generation → Google TTS → GCS audio storage  
✅ **Page-aware chunking**: Maintains page metadata for citation  
✅ **Background processing**: FastAPI BackgroundTasks (Cloud Tasks-ready architecture)

## Required Environment Variables

```bash
# GCP Core
GOOGLE_CLOUD_PROJECT=your-project-id
GCP_LOCATION=europe-west4  # Frankfurt

# Cloud SQL / AlloyDB
DB_USER=postgres
DB_PASSWORD=your-password
DB_NAME=files_db
INSTANCE_CONNECTION_NAME=project:region:instance

# GCS Buckets
GCS_BUCKET_PDFS=dwani-pdfs
GCS_BUCKET_CLEAN=dwani-clean-pdfs
GCS_BUCKET_AUDIO=dwani-audio

# Optional: Cloud Tasks (for production background jobs)
CLOUD_TASKS_QUEUE=projects/PROJECT/locations/LOCATION/queues/dwani-tasks
CLOUD_RUN_SERVICE_URL=https://your-service-run.app

# Token limits (unchanged)
MAX_CONTEXT_TOKENS=12000
MAX_HISTORY_TOKENS=3000
```

## Updated `requirements.txt`

```
fastapi[all]==0.109.0
uvicorn[standard]==0.27.0
google-cloud-aiplatform>=1.52.0
google-cloud-storage>=2.16.0
google-cloud-sql-connector[asyncpg]>=1.9.0
google-cloud-texttospeech>=2.16.0
google-cloud-tasks>=2.16.0
pg8000>=1.30.0
psycopg2-binary>=2.9.9
sqlalchemy>=2.0.25
rank-bm25>=0.2.2
pydantic>=2.6.0
python-multipart>=0.0.9
numpy>=1.26.0
```

## Deployment Quick Start

### 1. **Local Development** (with Cloud SQL proxy)
```bash
# Start Cloud SQL proxy
cloud-sql-proxy project:region:instance &

# Set env vars
export GOOGLE_CLOUD_PROJECT=your-project
export INSTANCE_CONNECTION_NAME=project:region:instance
# ... other vars

# Run locally
python main_gcp_vertex.py
```

### 2. **Cloud Run Deployment**

Create `Dockerfile`:
```dockerfile
FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main_gcp_vertex.py main.py
COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
```

Deploy:
```bash
gcloud run deploy dwani-api \
  --source . \
  --region europe-west4 \
  --allow-unauthenticated \
  --set-env-vars GOOGLE_CLOUD_PROJECT=$PROJECT_ID,GCP_LOCATION=europe-west4 \
  --set-secrets DB_PASSWORD=db-password:latest \
  --memory 2Gi \
  --timeout 300
```

## Production Enhancements (Next Steps)

For your **Google CE interview demo**, consider these upgrades to show production readiness:

1. **Vertex AI Vector Search Index**: Replace in-memory cosine similarity with managed index
2. **Cloud Tasks**: Move background jobs (`background_extraction_task`, `background_podcast_task`) to queue-triggered Cloud Run jobs
3. **IAM & signed URLs**: Add proper authentication, use GCS signed URLs for file downloads
4. **Monitoring**: Cloud Logging + Cloud Trace instrumentation
5. **Cost optimization**: Batch embeddings, cache frequent queries, use Spot instances for Cloud Run jobs
