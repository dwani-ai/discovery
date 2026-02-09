"""
dwani.ai API - GCP + Vertex AI Edition
Privacy-focused multimodal document extraction, regeneration, and multi-document aware hybrid RAG API
Rebuilt with Google Cloud Platform + Vertex AI components
"""

import argparse
import base64
import enum
import hashlib
import json
import logging
import logging.config
import os
import tempfile
import uuid
from datetime import datetime
from io import BytesIO
from typing import List, Optional, Dict, Tuple

from fastapi import FastAPI, UploadFile, HTTPException, BackgroundTasks, Depends, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from rank_bm25 import BM25Okapi
from sqlalchemy import Column, String, Text, DateTime, create_engine, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
import unicodedata
import uvicorn

# Google Cloud imports
from google.cloud import aiplatform, storage
from google.cloud.sql.connector import Connector
import vertexai
from vertexai.generative_models import GenerativeModel, Part, GenerationConfig
from vertexai.language_models import TextEmbeddingModel
from google.cloud import tasks_v2
import asyncio

# ========================= CONFIG & LOGGING =========================

logging.config.dictConfig({
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "simple": {"format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"}
    },
    "handlers": {
        "console": {"class": "logging.StreamHandler", "formatter": "simple"},
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "formatter": "simple",
            "filename": "dwani_api.log",
            "maxBytes": 10_000_000,
            "backupCount": 5,
        },
    },
    "root": {"level": "INFO", "handlers": ["console", "file"]},
})

logger = logging.getLogger("dwani_gcp_server")

# GCP Configuration
GCP_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT")
GCP_LOCATION = os.getenv("GCP_LOCATION", "europe-west4")  # Frankfurt region
BUCKET_PDFS = os.getenv("GCS_BUCKET_PDFS", "dwani-pdfs")
BUCKET_CLEAN_PDFS = os.getenv("GCS_BUCKET_CLEAN", "dwani-clean-pdfs")
BUCKET_AUDIO = os.getenv("GCS_BUCKET_AUDIO", "dwani-audio")

# Cloud SQL / AlloyDB Configuration
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_NAME = os.getenv("DB_NAME", "files_db")
INSTANCE_CONNECTION_NAME = os.getenv("INSTANCE_CONNECTION_NAME")  # project:region:instance

# Cloud Tasks Configuration
CLOUD_TASKS_QUEUE = os.getenv("CLOUD_TASKS_QUEUE")
CLOUD_RUN_SERVICE_URL = os.getenv("CLOUD_RUN_SERVICE_URL")

# Token Limits
MAX_CONTEXT_TOKENS = int(os.getenv("MAX_CONTEXT_TOKENS", "12000"))
MAX_HISTORY_TOKENS = int(os.getenv("MAX_HISTORY_TOKENS", "3000"))

if not GCP_PROJECT:
    raise RuntimeError("GOOGLE_CLOUD_PROJECT environment variable is required")

# Initialize Vertex AI
vertexai.init(project=GCP_PROJECT, location=GCP_LOCATION)

logger.info(f"Initialized Vertex AI: project={GCP_PROJECT}, location={GCP_LOCATION}")
logger.info(f"Context limits: MAX_CONTEXT_TOKENS={MAX_CONTEXT_TOKENS}, MAX_HISTORY_TOKENS={MAX_HISTORY_TOKENS}")

# ========================= FASTAPI APP =========================

app = FastAPI(
    title="dwani.ai API (GCP Edition)",
    description="Privacy-focused multimodal document extraction and RAG with GCP + Vertex AI",
    version="2.0.0-gcp",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://app.dwani.ai", "http://localhost:5173", "http://127.0.0.1:5173", "https://*.dwani.ai"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========================= DATABASE (Cloud SQL / AlloyDB) =========================

def get_db_engine():
    """Create SQLAlchemy engine with Cloud SQL Connector"""
    if not INSTANCE_CONNECTION_NAME:
        # Fallback to SQLite for local dev
        logger.warning("INSTANCE_CONNECTION_NAME not set, using local SQLite")
        return create_engine("sqlite:///./files.db", connect_args={"check_same_thread": False})

    connector = Connector()

    def getconn():
        conn = connector.connect(
            INSTANCE_CONNECTION_NAME,
            "pg8000",
            user=DB_USER,
            password=DB_PASSWORD,
            db=DB_NAME,
        )
        return conn

    return create_engine(
        "postgresql+pg8000://",
        creator=getconn,
    )

engine = get_db_engine()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class FileStatus(str, enum.Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class PodcastStatus(str, enum.Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class FileRecord(Base):
    __tablename__ = "files"
    id = Column(String, primary_key=True)
    filename = Column(String, index=True)
    content_type = Column(String)
    gcs_uri = Column(String)  # GCS path to original PDF
    status = Column(String, default=FileStatus.PENDING)
    extracted_text = Column(Text, nullable=True)
    page_count = Column(String, nullable=True)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class PodcastRecord(Base):
    __tablename__ = "podcasts"
    id = Column(String, primary_key=True)
    title = Column(String, index=True)
    file_ids = Column(Text)  # JSON encoded
    status = Column(String, default=PodcastStatus.PENDING)
    script = Column(Text, nullable=True)
    audio_gcs_uri = Column(String, nullable=True)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class NotebookRecord(Base):
    __tablename__ = "notebooks"
    id = Column(String, primary_key=True)
    name = Column(String, index=True)
    file_ids = Column(Text)  # JSON encoded
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class ChunkRecord(Base):
    """Store document chunks with embeddings metadata for hybrid search"""
    __tablename__ = "chunks"
    id = Column(String, primary_key=True)
    file_id = Column(String, index=True)
    filename = Column(String)
    chunk_index = Column(String)
    text = Column(Text)
    page_start = Column(String)
    page_end = Column(String)
    embedding_id = Column(String)  # Reference to Vertex AI Vector Search
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (Index('idx_file_chunk', 'file_id', 'chunk_index'),)


Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ========================= GCS CLIENT =========================

storage_client = storage.Client(project=GCP_PROJECT)


def upload_to_gcs(bucket_name: str, blob_name: str, data: bytes, content_type: str = "application/octet-stream") -> str:
    """Upload bytes to GCS and return public URI"""
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_string(data, content_type=content_type)
    return f"gs://{bucket_name}/{blob_name}"


def get_signed_url(gcs_uri: str, expiration_minutes: int = 60) -> str:
    """Generate signed URL for private GCS object"""
    bucket_name = gcs_uri.split("/")[2]
    blob_name = "/".join(gcs_uri.split("/")[3:])
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    return blob.generate_signed_url(expiration=datetime.timedelta(minutes=expiration_minutes))


# ========================= VERTEX AI MODELS =========================

# Embedding model
embedding_model = TextEmbeddingModel.from_pretrained("text-embedding-004")

# Generative models
gemini_flash = GenerativeModel("gemini-1.5-flash")
gemini_pro = GenerativeModel("gemini-1.5-pro")


def get_embeddings(texts: List[str]) -> List[List[float]]:
    """Get embeddings from Vertex AI"""
    embeddings = embedding_model.get_embeddings(texts)
    return [emb.values for emb in embeddings]


# ========================= SCHEMAS =========================

class ExtractionResponse(BaseModel):
    extracted_text: str
    page_count: int
    status: str = "success"


class FileUploadResponse(BaseModel):
    file_id: str
    filename: str
    status: str = "pending"
    message: str


class FileRetrieveResponse(BaseModel):
    file_id: str
    filename: str
    status: str
    extracted_text: Optional[str] = None
    error_message: Optional[str] = None
    created_at: datetime
    updated_at: datetime


class PodcastCreateRequest(BaseModel):
    file_ids: List[str]
    title: Optional[str] = None
    style: Optional[str] = "explainer"
    duration_minutes: Optional[int] = 20
    language: Optional[str] = "en"


class PodcastCreateResponse(BaseModel):
    podcast_id: str
    status: str
    message: str


class PodcastRetrieveResponse(BaseModel):
    podcast_id: str
    title: str
    file_ids: List[str]
    status: str
    audio_url: Optional[str] = None
    error_message: Optional[str] = None
    created_at: datetime
    updated_at: datetime


class NotebookCreateRequest(BaseModel):
    name: str
    file_ids: List[str]


class NotebookResponse(BaseModel):
    notebook_id: str
    name: str
    file_ids: List[str]
    created_at: datetime
    updated_at: datetime


class ChatMessage(BaseModel):
    role: str
    content: str


class MultiChatRequest(BaseModel):
    file_ids: List[str]
    messages: List[ChatMessage]


class MergePdfRequest(BaseModel):
    file_ids: List[str]


# ========================= UTILS =========================

def clean_text(text: str) -> str:
    return "".join(ch for ch in text if unicodedata.category(ch)[0] != "C" or ch in "\n\r\t")


def chunk_text_with_pages(page_texts: List[str], chunk_size: int = 800, overlap: int = 100) -> List[Dict]:
    """Chunk text while preserving page boundaries"""
    chunks = []
    words_buffer = []

    for page_num, page_text in enumerate(page_texts, start=1):
        page_words = page_text.split()
        words_buffer.extend([(word, page_num) for word in page_words])

        while len(words_buffer) > chunk_size:
            chunk_words = words_buffer[:chunk_size]
            chunk_text = " ".join(word for word, _ in chunk_words)
            pages_in_chunk = {page for _, page in chunk_words}

            chunks.append({
                'text': chunk_text,
                'page_start': min(pages_in_chunk),
                'page_end': max(pages_in_chunk)
            })

            words_buffer = words_buffer[chunk_size - overlap:]

    if words_buffer:
        chunk_text = " ".join(word for word, _ in words_buffer)
        pages_in_chunk = {page for _, page in words_buffer}
        chunks.append({
            'text': chunk_text,
            'page_start': min(pages_in_chunk),
            'page_end': max(pages_in_chunk)
        })

    return chunks


# ========================= VERTEX AI SERVICES =========================

async def extract_text_from_pdf_gemini(gcs_uri: str) -> Tuple[str, int]:
    """
    Extract text from PDF using Gemini 1.5 Flash multimodal
    Returns (full_text, estimated_page_count)
    """
    try:
        pdf_part = Part.from_uri(gcs_uri, mime_type="application/pdf")

        prompt = """Extract all text from this PDF document. 
        For each page, prefix with "--- PAGE {number} ---" on its own line.
        Preserve the structure and formatting of the text.
        Be thorough and accurate."""

        response = await asyncio.to_thread(
            gemini_flash.generate_content,
            [prompt, pdf_part],
            generation_config=GenerationConfig(
                temperature=0.1,
                max_output_tokens=8000,
            )
        )

        full_text = response.text

        # Estimate pages from markers
        page_count = full_text.count("--- PAGE")
        if page_count == 0:
            page_count = max(1, len(full_text) // 3000)  # Rough estimate

        return full_text, page_count

    except Exception as e:
        logger.error(f"Gemini PDF extraction failed for {gcs_uri}: {e}")
        raise


async def parse_pages_from_extraction(full_text: str) -> List[str]:
    """Parse page-delimited text into list of page texts"""
    if "--- PAGE" in full_text:
        pages = []
        current_page = []

        for line in full_text.split("\n"):
            if line.strip().startswith("--- PAGE"):
                if current_page:
                    pages.append("\n".join(current_page))
                current_page = []
            else:
                current_page.append(line)

        if current_page:
            pages.append("\n".join(current_page))

        return pages
    else:
        # Fallback: split by approximate page length
        words = full_text.split()
        page_size = 500
        return [" ".join(words[i:i+page_size]) for i in range(0, len(words), page_size)]


async def store_chunks_with_embeddings(file_id: str, filename: str, page_texts: List[str], db: Session):
    """
    Chunk text, generate embeddings, and store in DB
    For production: also index in Vertex AI Vector Search
    """
    chunks_meta = chunk_text_with_pages(page_texts)

    if not chunks_meta:
        return

    texts = [c['text'] for c in chunks_meta]

    # Generate embeddings in batches
    embeddings = get_embeddings(texts)

    # Store in database
    for i, (chunk, emb) in enumerate(zip(chunks_meta, embeddings)):
        chunk_id = hashlib.md5(f"{file_id}_{i}".encode()).hexdigest()

        # For MVP: store embedding as JSON; for production: use Vector Search index
        chunk_record = ChunkRecord(
            id=chunk_id,
            file_id=file_id,
            filename=filename,
            chunk_index=str(i),
            text=chunk['text'],
            page_start=str(chunk['page_start']),
            page_end=str(chunk['page_end']),
            embedding_id=chunk_id,  # Placeholder; map to Vector Search ID
        )

        db.merge(chunk_record)

    db.commit()
    logger.info(f"Stored {len(chunks_meta)} chunks with embeddings for {filename}")


# ========================= HYBRID SEARCH =========================

def reciprocal_rank_fusion(results: List[Tuple[int, float]], k: int = 60) -> List[int]:
    """Fuse ranked results using RRF"""
    score_dict = {}
    for rank_offset, (doc_idx, _) in enumerate(results):
        rank = rank_offset + 1
        score_dict[doc_idx] = score_dict.get(doc_idx, 0) + 1 / (k + rank)
    return sorted(score_dict.keys(), key=lambda i: score_dict[i], reverse=True)


async def hybrid_search(question: str, file_ids: List[str], db: Session, top_k: int = 20) -> List[Dict]:
    """
    Hybrid search: Vertex AI semantic + BM25 lexical, fused with RRF
    Returns list of {text, filename, page_start, page_end, score}
    """
    # Get all chunks for these files
    chunks = db.query(ChunkRecord).filter(ChunkRecord.file_id.in_(file_ids)).all()

    if not chunks:
        return []

    # Semantic search: embedding similarity
    query_embedding = get_embeddings([question])[0]
    chunk_embeddings = get_embeddings([c.text for c in chunks])

    # Cosine similarity
    from numpy import dot
    from numpy.linalg import norm

    semantic_scores = []
    for i, chunk_emb in enumerate(chunk_embeddings):
        similarity = dot(query_embedding, chunk_emb) / (norm(query_embedding) * norm(chunk_emb))
        semantic_scores.append((i, 1 - similarity))  # Distance for sorting

    semantic_scores.sort(key=lambda x: x[1])
    semantic_ranked = semantic_scores[:top_k]

    # BM25 lexical search
    corpus = [c.text.lower().split() for c in chunks]
    bm25 = BM25Okapi(corpus)
    tokenized_query = question.lower().split()
    bm25_scores = bm25.get_scores(tokenized_query)

    bm25_ranked = [(i, -score) for i, score in enumerate(bm25_scores) if score > 0]
    bm25_ranked.sort(key=lambda x: x[1])
    bm25_ranked = bm25_ranked[:top_k]

    # Fuse rankings
    all_ranked = semantic_ranked + bm25_ranked
    fused_indices = reciprocal_rank_fusion(all_ranked, k=60)[:top_k]

    # Build results
    results = []
    for idx in fused_indices:
        if idx >= len(chunks):
            continue
        chunk = chunks[idx]
        results.append({
            "text": chunk.text,
            "filename": chunk.filename,
            "page_start": chunk.page_start,
            "page_end": chunk.page_end,
            "score": 1 - semantic_scores[idx][1] if idx < len(semantic_scores) else 0.5
        })

    return results


# ========================= CONTRADICTION DETECTION =========================

async def detect_contradictions(question: str, sources: List[Dict]) -> Optional[str]:
    """Use Gemini to detect contradictions across sources"""
    if len(sources) < 2:
        return None

    doc_excerpts = {}
    for s in sources:
        filename = s["filename"]
        if filename not in doc_excerpts:
            doc_excerpts[filename] = []
        doc_excerpts[filename].append(s["text"][:500])

    prompt = f"""You are an expert analyst. Review the following excerpts from different documents and the user's question.

Question: {question}

"""

    for filename, excerpts in doc_excerpts.items():
        prompt += f"\n--- From {filename} ---\n" + "\n\n".join(excerpts[:3]) + "\n"

    prompt += """
Are there any contradictions or conflicting information between the documents regarding the question?
If yes, clearly state what contradicts what.
If no clear contradiction, respond with "No contradictions detected."
Respond in one short paragraph."""

    try:
        response = await asyncio.to_thread(
            gemini_flash.generate_content,
            prompt,
            generation_config=GenerationConfig(temperature=0.3, max_output_tokens=512)
        )

        result = response.text.strip()
        return result if "no contradictions" not in result.lower() else None

    except Exception as e:
        logger.warning(f"Contradiction detection failed: {e}")
        return None


# ========================= PODCAST GENERATION =========================

async def generate_podcast_script(
    file_ids: List[str],
    style: str,
    duration_minutes: int,
    language: str,
    db: Session,
) -> str:
    """Generate NotebookLM-style podcast script using Gemini"""

    # Validate files
    records = db.query(FileRecord).filter(FileRecord.id.in_(file_ids)).all()
    if len(records) != len(file_ids):
        raise HTTPException(status_code=404, detail="Some files not found")

    for r in records:
        if r.status != FileStatus.COMPLETED:
            raise HTTPException(status_code=400, detail=f"File {r.filename} not processed")

    # Get broad context via hybrid search
    pseudo_question = "Summarize the main points from these documents for a podcast conversation"
    search_results = await hybrid_search(pseudo_question, file_ids, db, top_k=30)

    # Build context
    context_parts = []
    current_tokens = 0

    for result in search_results:
        text = result["text"]
        estimated_tokens = int(len(text.split()) * 1.3)

        if current_tokens + estimated_tokens > MAX_CONTEXT_TOKENS:
            break

        filename = result["filename"]
        page_info = f"(Page {result['page_start']})" if result['page_start'] == result['page_end'] else f"(Pages {result['page_start']}–{result['page_end']})"

        context_parts.append(f"[{filename} {page_info}]\n{text}")
        current_tokens += estimated_tokens

    context = "\n\n".join(context_parts) if context_parts else "No content retrieved."

    approx_tokens = min(duration_minutes * 150, 4000)

    prompt = f"""You are creating a podcast episode transcript in {language}.
Create a natural, engaging dialogue between two hosts, "Host A" and "Host B",
that explains and discusses the key ideas from the following documents.

- Style: {style}
- Target length: about {duration_minutes} minutes of spoken audio
- Refer to document names and pages when relevant (e.g., "According to Contract.pdf, page 3...")
- Do not read documents verbatim; summarize, explain, and discuss

Documents context:
{context}

Generate the podcast transcript now:"""

    response = await asyncio.to_thread(
        gemini_pro.generate_content,
        prompt,
        generation_config=GenerationConfig(
            temperature=0.8,
            max_output_tokens=approx_tokens,
        )
    )

    return response.text.strip()


async def generate_podcast_audio(script: str, podcast_id: str, language: str = "en-US") -> str:
    """
    Generate audio from script using Vertex AI Text-to-Speech
    Returns GCS URI
    """
    from google.cloud import texttospeech_v1 as texttospeech

    client = texttospeech.TextToSpeechClient()

    synthesis_input = texttospeech.SynthesisInput(text=script)

    # Configure voice
    voice = texttospeech.VoiceSelectionParams(
        language_code=language if "-" in language else "en-US",
        ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL,
    )

    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3,
        speaking_rate=1.0,
    )

    response = await asyncio.to_thread(
        client.synthesize_speech,
        input=synthesis_input,
        voice=voice,
        audio_config=audio_config
    )

    # Upload to GCS
    blob_name = f"{podcast_id}.mp3"
    gcs_uri = upload_to_gcs(BUCKET_AUDIO, blob_name, response.audio_content, "audio/mpeg")

    logger.info(f"Generated podcast audio: {gcs_uri}")
    return gcs_uri


# ========================= BACKGROUND TASKS =========================

async def background_extraction_task(file_id: str, gcs_uri: str, filename: str, db: Session):
    """Background task to extract text from PDF using Gemini"""
    file_record = db.query(FileRecord).filter(FileRecord.id == file_id).first()
    if not file_record:
        return

    file_record.status = FileStatus.PROCESSING
    db.commit()

    try:
        # Extract with Gemini
        full_text, page_count = await extract_text_from_pdf_gemini(gcs_uri)
        page_texts = await parse_pages_from_extraction(full_text)

        file_record.extracted_text = full_text
        file_record.page_count = str(page_count)
        file_record.status = FileStatus.COMPLETED
        db.commit()

        # Store chunks + embeddings
        await store_chunks_with_embeddings(file_id, filename, page_texts, db)

        logger.info(f"Extraction completed for {filename} ({page_count} pages)")

    except Exception as e:
        file_record.status = FileStatus.FAILED
        file_record.error_message = str(e)
        logger.error(f"Extraction failed for {file_id}: {e}")
    finally:
        file_record.updated_at = datetime.utcnow()
        db.commit()


async def background_podcast_task(podcast_id: str, request: PodcastCreateRequest, db: Session):
    """Background task to generate podcast script and audio"""
    podcast = db.query(PodcastRecord).filter(PodcastRecord.id == podcast_id).first()
    if not podcast:
        return

    podcast.status = PodcastStatus.PROCESSING
    db.commit()

    try:
        # Generate script
        script = await generate_podcast_script(
            file_ids=request.file_ids,
            style=request.style or "explainer",
            duration_minutes=request.duration_minutes or 20,
            language=request.language or "en",
            db=db,
        )

        podcast.script = script
        db.commit()

        # Generate audio
        audio_gcs_uri = await generate_podcast_audio(
            script=script,
            podcast_id=podcast_id,
            language=request.language or "en-US",
        )

        podcast.audio_gcs_uri = audio_gcs_uri
        podcast.status = PodcastStatus.COMPLETED

    except Exception as e:
        podcast.status = PodcastStatus.FAILED
        podcast.error_message = str(e)
        logger.error(f"Podcast generation failed for {podcast_id}: {e}")
    finally:
        podcast.updated_at = datetime.utcnow()
        db.commit()


# ========================= ROUTES =========================

@app.post("/files/upload", response_model=FileUploadResponse, tags=["Files"])
async def upload_file(
    file: UploadFile,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Upload PDF file to GCS and trigger extraction"""
    if not file.filename.lower().endswith(".pdf") or file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files supported")

    content = await file.read()
    file_id = str(uuid.uuid4())

    # Upload to GCS
    blob_name = f"{file_id}/{file.filename}"
    gcs_uri = upload_to_gcs(BUCKET_PDFS, blob_name, content, "application/pdf")

    # Create record
    record = FileRecord(
        id=file_id,
        filename=file.filename,
        content_type=file.content_type,
        gcs_uri=gcs_uri
    )
    db.add(record)
    db.commit()

    # Trigger background extraction
    background_tasks.add_task(background_extraction_task, file_id, gcs_uri, file.filename, db)

    return FileUploadResponse(
        file_id=file_id,
        filename=file.filename,
        message="Upload successful. Processing in background."
    )


@app.get("/files/{file_id}", response_model=FileRetrieveResponse, tags=["Files"])
def get_file(file_id: str, db: Session = Depends(get_db)):
    """Get file metadata and extraction status"""
    record = db.query(FileRecord).filter(FileRecord.id == file_id).first()
    if not record:
        raise HTTPException(status_code=404, detail="File not found")
    return record


@app.get("/files/", tags=["Files"])
def list_files(limit: int = 20, db: Session = Depends(get_db)):
    """List recent files"""
    files = db.query(FileRecord).order_by(FileRecord.created_at.desc()).limit(limit).all()
    return [
        {
            "file_id": f.id,
            "filename": f.filename,
            "status": f.status,
            "created_at": f.created_at.isoformat(),
        }
        for f in files
    ]


@app.delete("/files/{file_id}", tags=["Files"])
def delete_file(file_id: str, db: Session = Depends(get_db)):
    """Delete file and associated chunks"""
    record = db.query(FileRecord).filter(FileRecord.id == file_id).first()
    if not record:
        raise HTTPException(status_code=404, detail="File not found")

    # Delete chunks
    db.query(ChunkRecord).filter(ChunkRecord.file_id == file_id).delete()

    db.delete(record)
    db.commit()

    return {"message": "File deleted successfully"}


# ========================= CHAT ROUTE =========================

@app.post("/chat-with-document", tags=["Files"])
async def chat_with_documents(request: MultiChatRequest, db: Session = Depends(get_db)):
    """Multi-document chat with hybrid RAG"""
    if not request.file_ids:
        raise HTTPException(status_code=400, detail="file_ids required")

    # Validate files
    records = db.query(FileRecord).filter(FileRecord.id.in_(request.file_ids)).all()
    if len(records) != len(request.file_ids):
        raise HTTPException(status_code=404, detail="Some files not found")

    for r in records:
        if r.status != FileStatus.COMPLETED:
            raise HTTPException(status_code=400, detail=f"File {r.filename} not processed")

    # Extract question
    user_msg = [m for m in request.messages if m.role == "user"]
    if not user_msg:
        raise HTTPException(status_code=400, detail="No user question")

    question = user_msg[-1].content

    # Hybrid search
    search_results = await hybrid_search(question, request.file_ids, db, top_k=15)

    # Build context
    context_parts = []
    sources = []
    current_tokens = 0

    for result in search_results:
        text = result["text"]
        estimated_tokens = len(text.split()) * 1.3

        if current_tokens + estimated_tokens > MAX_CONTEXT_TOKENS:
            break

        context_parts.append(text)
        current_tokens += estimated_tokens

        page_citation = f"Page {result['page_start']}" if result['page_start'] == result['page_end'] else f"Pages {result['page_start']}–{result['page_end']}"

        sources.append({
            "filename": result["filename"],
            "page": page_citation,
            "excerpt": text.strip()[:500],
            "relevance_score": round(result["score"], 3),
            "text": text  # For contradiction detection
        })

    context = "\n\n".join(context_parts) if context_parts else "No relevant content found."

    # Build prompt
    doc_list = ", ".join(set(s["filename"] for s in sources))

    system_prompt = f"""You are an expert assistant analyzing multiple documents: {doc_list}.

When answering:
- Always explicitly mention which document each fact comes from (e.g., "According to Contract.pdf...", "The NDA states...")
- If information differs between documents, clearly note it
- Be precise and cite page numbers when available
If the answer cannot be found, say "I don't know"

Context:
{context}

Answer clearly and professionally."""

    # Build conversation history (trimmed)
    recent_messages = []
    history_tokens = 0

    for msg in reversed(request.messages[-10:]):
        msg_tokens = len(msg.content.split()) * 1.3
        if history_tokens + msg_tokens > MAX_HISTORY_TOKENS:
            break
        recent_messages.insert(0, msg.content if msg.role == "user" else f"Assistant: {msg.content}")
        history_tokens += msg_tokens

    full_prompt = system_prompt + "\n\nConversation:\n" + "\n".join(recent_messages)

    try:
        # Generate response with Gemini
        response = await asyncio.to_thread(
            gemini_pro.generate_content,
            full_prompt,
            generation_config=GenerationConfig(
                temperature=0.5,
                max_output_tokens=1024,
            )
        )

        answer = response.text.strip()

        # Detect contradictions
        contradiction_warning = await detect_contradictions(question, sources[:10])

        sources.sort(key=lambda x: x["relevance_score"], reverse=True)

        final_answer = answer
        if contradiction_warning:
            final_answer = f"⚠️ **Potential Contradiction Detected**\n\n{contradiction_warning}\n\n**Answer:**\n{answer}"

        return {
            "answer": final_answer,
            "sources": [
                {
                    "filename": s["filename"],
                    "page": s["page"],
                    "excerpt": s["excerpt"],
                    "relevance_score": s["relevance_score"]
                }
                for s in sources[:5]
            ]
        }

    except Exception as e:
        logger.error(f"Chat generation failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate response")


# ========================= PODCAST ROUTES =========================

@app.post("/podcasts", response_model=PodcastCreateResponse, tags=["Podcasts"])
async def create_podcast(
    request: PodcastCreateRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    """Create podcast from documents"""
    if not request.file_ids:
        raise HTTPException(status_code=400, detail="file_ids required")

    # Validate files
    records = db.query(FileRecord).filter(FileRecord.id.in_(request.file_ids)).all()
    if len(records) != len(request.file_ids):
        raise HTTPException(status_code=404, detail="Some files not found")

    for r in records:
        if r.status != FileStatus.COMPLETED:
            raise HTTPException(status_code=400, detail=f"File {r.filename} not processed")

    # Check for existing podcast
    requested_ids_set = set(request.file_ids)
    existing_podcasts = db.query(PodcastRecord).filter(
        PodcastRecord.status != PodcastStatus.FAILED
    ).all()

    for p in existing_podcasts:
        try:
            stored_ids = json.loads(p.file_ids) if p.file_ids else []
        except Exception:
            continue

        if set(stored_ids) == requested_ids_set:
            message = (
                "Podcast already generated."
                if p.status == PodcastStatus.COMPLETED
                else "Podcast generation already in progress."
            )
            return PodcastCreateResponse(
                podcast_id=p.id,
                status=p.status,
                message=message,
            )

    # Create new podcast
    podcast_id = str(uuid.uuid4())
    podcast = PodcastRecord(
        id=podcast_id,
        title=request.title or "AI Podcast",
        file_ids=json.dumps(sorted(request.file_ids)),
        status=PodcastStatus.PENDING,
    )

    db.add(podcast)
    db.commit()

    background_tasks.add_task(background_podcast_task, podcast_id, request, db)

    return PodcastCreateResponse(
        podcast_id=podcast_id,
        status=podcast.status,
        message="Podcast generation started.",
    )


@app.get("/podcasts/{podcast_id}", response_model=PodcastRetrieveResponse, tags=["Podcasts"])
def get_podcast(podcast_id: str, db: Session = Depends(get_db)):
    """Get podcast status and download URL"""
    podcast = db.query(PodcastRecord).filter(PodcastRecord.id == podcast_id).first()
    if not podcast:
        raise HTTPException(status_code=404, detail="Podcast not found")

    audio_url = None
    if podcast.audio_gcs_uri and podcast.status == PodcastStatus.COMPLETED:
        # Generate signed URL
        audio_url = get_signed_url(podcast.audio_gcs_uri, expiration_minutes=120)

    return PodcastRetrieveResponse(
        podcast_id=podcast.id,
        title=podcast.title,
        file_ids=json.loads(podcast.file_ids) if podcast.file_ids else [],
        status=podcast.status,
        audio_url=audio_url,
        error_message=podcast.error_message,
        created_at=podcast.created_at,
        updated_at=podcast.updated_at,
    )


# ========================= NOTEBOOK ROUTES =========================

@app.get("/notebooks", response_model=List[NotebookResponse], tags=["Notebooks"])
def list_notebooks(db: Session = Depends(get_db)):
    """List all notebooks"""
    notebooks = db.query(NotebookRecord).order_by(NotebookRecord.created_at.desc()).all()
    return [
        NotebookResponse(
            notebook_id=n.id,
            name=n.name,
            file_ids=json.loads(n.file_ids) if n.file_ids else [],
            created_at=n.created_at,
            updated_at=n.updated_at,
        )
        for n in notebooks
    ]


@app.post("/notebooks", response_model=NotebookResponse, tags=["Notebooks"])
def create_notebook(request: NotebookCreateRequest, db: Session = Depends(get_db)):
    """Create a notebook"""
    if not request.file_ids:
        raise HTTPException(status_code=400, detail="file_ids required")

    records = db.query(FileRecord).filter(FileRecord.id.in_(request.file_ids)).all()
    if len(records) != len(request.file_ids):
        raise HTTPException(status_code=404, detail="Some files not found")

    notebook_id = str(uuid.uuid4())
    notebook = NotebookRecord(
        id=notebook_id,
        name=request.name.strip() or "Untitled Notebook",
        file_ids=json.dumps(sorted(request.file_ids)),
    )

    db.add(notebook)
    db.commit()
    db.refresh(notebook)

    return NotebookResponse(
        notebook_id=notebook.id,
        name=notebook.name,
        file_ids=json.loads(notebook.file_ids) if notebook.file_ids else [],
        created_at=notebook.created_at,
        updated_at=notebook.updated_at,
    )


@app.delete("/notebooks/{notebook_id}", tags=["Notebooks"])
def delete_notebook(notebook_id: str, db: Session = Depends(get_db)):
    """Delete a notebook"""
    notebook = db.query(NotebookRecord).filter(NotebookRecord.id == notebook_id).first()
    if not notebook:
        raise HTTPException(status_code=404, detail="Notebook not found")

    db.delete(notebook)
    db.commit()

    return {"message": "Notebook deleted successfully"}


# ========================= HEALTH CHECK =========================

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": "2.0.0-gcp",
        "project": GCP_PROJECT,
        "location": GCP_LOCATION
    }


# ========================= ENTRY POINT =========================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    uvicorn.run("main:app", host=args.host, port=args.port, reload=True)
