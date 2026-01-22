Discovery - Agentic Intelligence Platform on GCP
================================================
Try Demo: https://app.dwani.ai

## Overview
Agentic PDF assistant: Upload → OCR/digitize → RAG chat → Clean PDF export.
Fully managed on GCP: Vertex AI (models/embeddings), Vector Search (RAG), Cloud Run (API/UI).


| Local Component                          | GCP Replacement                                    | Migration Rationale                                                                            |
| ---------------------------------------- | -------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| FastAPI server (uvicorn)                 | Cloud Run or Vertex AI custom prediction endpoints | Serverless scaling, autoscaling to zero; deploy Docker image directly. colab.research.google+1 |
| ChromaDB (local vector store)            | Vertex AI Vector Search                            | Managed vector DB with hybrid search (semantic + BM25), ACLs, autoscaling. github+1            |
| VLLM/llama.cpp (Gemma3-Vision text-only) | Vertex AI endpoints (Gemma 3 or Qwen VL models)    | GPU‑optimized serving, multi‑model (embeddings + vision + LLM), no local infra. cloud.google+1 |
| SQLite (full text storage)               | Cloud SQL (PostgreSQL) or Firestore                | Managed DB with backups, scaling; pgvector extension if needed. github​                        |
| Frontend (React?)                        | Cloud Run or App Engine                            | Static hosting via Cloud Storage + CDN, or full app on Run. github​                            |
| pdf2image, fpdf (PDF handling)           | Document AI + Cloud Storage                        | Native OCR/parsing, chunking, clean PDF gen. github​                                           |


----
----



```


## GCP Architecture
┌─────────────────────────────────────────────────────────────────────────────┐
│                          CLIENT (Cloud Storage/CDN)                         │
│   • Upload PDF  • Chat w/ docs  • Download clean PDF                        │
└─────────────────────────────────────────────────────────────────────────────┘
         │ HTTP/HTTPS
         ▼
┌───────────────────────────────┐
│     Cloud Run (FastAPI)       │  ← Deploy: gcloud run deploy discovery-api
└───────────────┬───────────────┘
                │
┌───────────────┼───────────────┐
│               │               │
▼               ▼               ▼
Cloud Storage   │   Vertex AI   │  /files/{id}/pdf
(PDF landing)   │   Endpoints   │  (Document AI)
    │           │ (Gemma VL/    │
    ▼           │  Text+Embed)  │
Eventarc ───────┼───────────────┘
    │           │
    ▼           ▼
Document AI     │
(OCR/Parse) ────┼──> Vertex AI Embeddings
    │           │
    ▼           ▼
BigQuery/       │
Cloud SQL       │
(metadata) ────┼──> Vertex AI Vector Search (hybrid BM25+semantic)
                │
                ▼
             RAG Query (/chat) → RRF fusion → Grounded response + citations
```


---
---

| Subsystem     | Products                                                          | Purpose                                                        |
| ------------- | ----------------------------------------------------------------- | -------------------------------------------------------------- |
| Ingestion     | Cloud Storage, Pub/Sub, Cloud Run functions, Vertex AI Embeddings | Land, trigger, parse/chunk/embed/index documents. reddit+1     |
| Retrieval     | Vertex AI Vector Search or Vertex AI Search                       | Semantic search with ACL filtering, autoscaling. reddit+1      |
| Generation    | Vertex AI (Gemini models), Prompt Optimizer                       | Grounded generation, safety filters, prompt tuning. reddit+1   |
| App Layer     | Cloud Run (frontend/backend), IAM/VPC Service Controls            | Secure serving, SSO integration, regional deployment. reddit+1 |
| Observability | Cloud Logging, Cloud Monitoring, BigQuery                         | Query logs, eval metrics, A/B testing. reddit​                 |

## Setup on GCP

### Prerequisites
- GCP project with Vertex AI API, Cloud Run, Storage enabled.
- Service account with roles: Vertex AI User, Storage Admin, Cloud Run Developer.
- Install: `gcloud`, Terraform, Google Cloud SDK.  [github](https://github.com/dwani-ai/discovery)

### 1. Deploy Infrastructure (Terraform)
```bash
terraform init
terraform apply
# Creates: GCS bucket, Pub/Sub, Vector Search index, IAM
```

### 2. Deploy Models (Vertex AI)
```bash
# Embeddings & LLM (Gemma 3 or equiv)
gcloud ai models upload \
  --region=us-central1 \
  --display-name=discovery-embeddings \
  --container-ports=8080 \
  --machine-type=n1-standard-4 \
  --accelerator=type=t4, count=1 \
  gs://your-models/text-embedding-004
```
Deploy Vision/OCR model similarly.  [cloud.google](https://cloud.google.com/vertex-ai/generative-ai/docs/open-models/vllm/use-vllm)

### 3. Deploy API/UI (Cloud Run)
```bash
# Build & deploy FastAPI (update code for Vertex SDK)
gcloud builds submit --tag gcr.io/PROJECT/discovery-api
gcloud run deploy discovery-api \
  --image gcr.io/PROJECT/discovery-api \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars "VECTOR_SEARCH_ENDPOINT=projects/.../locations/.../indexes/..." \
                 "VERTEX_PROJECT=your-project" \
                 "VERTEX_REGION=us-central1"
```
Frontend: Static to Cloud Storage or same Run service.  [colab.research.google](https://colab.research.google.com/github/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/official/custom/SDK_Custom_Container_Prediction.ipynb)

### 4. Migrate Local Config
```
export VERTEX_PROJECT=your-project
export VERTEX_REGION=us-central1
export MAX_CONTEXT_TOKENS=32768
export MAX_HISTORY_TOKENS=4096
export VECTOR_SEARCH_PROJECT_ID=your-project  # For hybrid search
```

### 5. Test
```
curl -X POST https://discovery-api-XXXX.run.app/files \
  -F "file=@sample.pdf"
curl -X POST https://discovery-api-XXXX.run.app/chat-with-document \
  -d '{"file_id": "abc", "query": "Summarize page 3"}'
```

## Features (GCP‑Enhanced)
- Source audit & delete: Via Storage lifecycle + IAM.
- PDF regenerate: Document AI output to Storage.
- Multi‑doc chat: Vector Search across files.
- Page citations: Metadata filtering.
- Contradictions: Vertex grounding scores.  [github](https://github.com/dwani-ai/discovery)

## Code Changes Needed
- Replace Chroma → `google.cloud.aiplatform.MatchingEngineIndexEndpoint`
- VLLM → `vertexai.generative_models.GenerativeModel`
- SQLite → `google.cloud.sql` or BigQuery SDK.
- BackgroundTasks → Cloud Run jobs/Eventarc.

## Costs & Scaling
- Vertex: $0.0001/1k chars (LLM), $0.00002/query (Vector).
- Cloud Run: Pay‑per‑request, autoscales to 1k+.

Fork repo, apply Terraform, deploy in <30min. Full migration aligns with GCP RAG blueprints.  [github](https://github.com/dwani-ai/discovery)

Discovery now runs natively on Google Cloud Platform (GCP) with Vertex AI for RAG/LLM, Vector Search for hybrid retrieval, and Cloud Run for serverless APIs—ideal for enterprise scale.  [github](https://github.com/dwani-ai/discovery)
