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

import chromadb
import dwani
from chromadb.utils import embedding_functions
from fastapi import FastAPI, UploadFile, HTTPException, BackgroundTasks, Depends, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fpdf import FPDF
from openai import AsyncOpenAI
from pdf2image import convert_from_bytes
from PIL import Image
from pydantic import BaseModel
from rank_bm25 import BM25Okapi
from sqlalchemy import Column, String, Text, DateTime, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
import unicodedata

import uvicorn

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

logger = logging.getLogger("dwani_server")

DWANI_API_BASE_URL = os.getenv("DWANI_API_BASE_URL")
if not DWANI_API_BASE_URL:
    raise RuntimeError("DWANI_API_BASE_URL environment variable is required.")

DWANI_API_KEY = os.getenv("DWANI_API_KEY")
if not DWANI_API_KEY:
    logger.warning("DWANI_API_KEY environment variable not set – audio podcast features will be disabled.")
else:
    dwani.api_key = DWANI_API_KEY
    dwani.api_base = DWANI_API_BASE_URL

FONT_PATH = os.path.join(os.path.dirname(__file__), "fonts", "DejaVuSans.ttf")
if not os.path.exists(FONT_PATH):
    logger.warning("DejaVuSans.ttf not found – PDF regeneration will fail until added.")

# ========================= TOKEN LIMITS (ENV CONFIGURABLE) =========================

MAX_CONTEXT_TOKENS = int(os.getenv("MAX_CONTEXT_TOKENS", "12000"))
MAX_HISTORY_TOKENS = int(os.getenv("MAX_HISTORY_TOKENS", "3000"))

logger.info(f"Context limits: MAX_CONTEXT_TOKENS={MAX_CONTEXT_TOKENS}, MAX_HISTORY_TOKENS={MAX_HISTORY_TOKENS}")

# ========================= FASTAPI APP =========================

app = FastAPI(
    title="dwani.ai API",
    description="Privacy-focused multimodal document extraction, regeneration, and multi-document aware hybrid RAG API",
    version="1.3.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://app.dwani.ai", "http://localhost:5173", "http://127.0.0.1:5173", "https://*.dwani.ai"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========================= DATABASE =========================

DATABASE_URL = "sqlite:///./files.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
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
    status = Column(String, default=FileStatus.PENDING)
    extracted_text = Column(Text, nullable=True)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class PodcastRecord(Base):
    __tablename__ = "podcasts"
    id = Column(String, primary_key=True)
    title = Column(String, index=True)
    file_ids = Column(Text)  # JSON encoded list of FileRecord IDs
    status = Column(String, default=PodcastStatus.PENDING)
    script = Column(Text, nullable=True)
    audio_path = Column(String, nullable=True)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ========================= VECTOR STORE =========================

chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="documents")

embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="BAAI/bge-small-en-v1.5"
)


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


def encode_image(image: Image.Image) -> str:
    buf = BytesIO()
    image.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


async def generate_podcast_script(
    file_ids: List[str],
    style: str,
    duration_minutes: int,
    language: str,
    db: Session,
) -> str:
    """
    Generate a NotebookLM-style podcast script summarizing the selected documents.
    """
    # Ensure all files exist and are completed
    records = db.query(FileRecord).filter(FileRecord.id.in_(file_ids)).all()
    if len(records) != len(file_ids):
        raise HTTPException(status_code=404, detail="Some files not found")

    for r in records:
        if r.status != FileStatus.COMPLETED:
            raise HTTPException(status_code=400, detail=f"File {r.filename} not processed")

    # Use a broad pseudo-question to pull diverse context
    pseudo_question = "Summarize the main points from these documents and structure them into an engaging conversation."

    vector_results = collection.query(
        query_embeddings=embedding_function([pseudo_question]),
        n_results=40,
        where={"file_id": {"$in": file_ids}},
        include=["documents", "metadatas", "distances"],
    )

    docs = vector_results.get("documents", [[]])[0] if vector_results.get("documents") else []
    metas = vector_results.get("metadatas", [[]])[0] if vector_results.get("metadatas") else []

    context_parts: List[str] = []
    current_tokens = 0

    for doc, meta in zip(docs, metas):
        estimated_tokens = int(len(doc.split()) * 1.3)
        if current_tokens + estimated_tokens > MAX_CONTEXT_TOKENS:
            break
        filename = meta.get("filename", "Unknown Document")
        page_start = meta.get("page_start")
        page_end = meta.get("page_end")
        page_info = ""
        if page_start is not None and page_end is not None:
            if page_start == page_end:
                page_info = f"(Page {page_start})"
            else:
                page_info = f"(Pages {page_start}–{page_end})"

        context_parts.append(f"[{filename} {page_info}]\n{doc}")
        current_tokens += estimated_tokens

    context = "\n\n".join(context_parts) if context_parts else "No relevant content could be retrieved from the documents."

    approx_tokens = min(duration_minutes * 150, 4000)

    system_prompt = f"""
You are creating a podcast episode transcript in {language}.

Create a natural, engaging dialogue between two hosts, "Host A" and "Host B",
that explains and discusses the key ideas from the following documents.

- Style: {style}
- Target length: about {duration_minutes} minutes of spoken audio.
- Refer to document names and pages when relevant (e.g., "According to Contract.pdf, page 3...").
- Do not read the documents verbatim; instead summarize, explain, and discuss.

Documents context:
{context}
"""

    client = get_openai_client()
    response = await client.chat.completions.create(
        model="gemma3",
        messages=[{"role": "system", "content": system_prompt}],
        temperature=0.8,
        max_tokens=approx_tokens,
    )
    script = response.choices[0].message.content.strip()
    return script


def generate_podcast_audio(script: str, podcast_id: str, language: str = "english") -> str:
    """
    Use dwani Audio.speech to generate an mp3 file for the podcast script.
    Returns the absolute path to the saved audio file.
    """
    if not DWANI_API_KEY:
        raise RuntimeError("DWANI_API_KEY is not set – audio generation is disabled.")

    base_dir = os.path.join(os.getcwd(), "podcasts")
    os.makedirs(base_dir, exist_ok=True)

    output_path = os.path.join(base_dir, f"{podcast_id}.mp3")

    # dwani.Audio.speech returns raw audio bytes
    audio_bytes = dwani.Audio.speech(
        input=script,
        response_format="mp3",
        language=language,
    )

    # Write to a temp file first, then move into place to avoid partial writes
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    os.replace(tmp_path, output_path)
    return output_path


async def pdf_to_images(pdf_bytes: bytes) -> List[Image.Image]:
    try:
        return convert_from_bytes(pdf_bytes, fmt="png")
    except Exception as e:
        logger.error(f"PDF to image conversion failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to process PDF")


def get_openai_client(model: str = "gemma3") -> AsyncOpenAI:
    valid_models = {"gemma3", "gpt-oss"}
    if model not in valid_models:
        raise ValueError(f"Invalid model: {model}")
    
    return AsyncOpenAI(api_key="http", base_url="https://qwen.dwani.ai/v1")


# ========================= SERVICES =========================

async def extract_text_from_images_per_page(images: List[Image.Image]) -> List[str]:
    client = get_openai_client()
    page_texts = []

    for i, img in enumerate(images):
        base64_img = encode_image(img)
        messages = [
            {"role": "system", "content": "You are an expert OCR assistant. Extract accurate plain text only."},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}},
                    {"type": "text", "text": f"Extract clean, accurate plain text from this page. Preserve structure."},
                ],
            },
        ]

        response = await client.chat.completions.create(
            model="gemma3",
            messages=messages,
            temperature=0.2,
            max_tokens=2048,
        )
        page_text = response.choices[0].message.content.strip()
        page_texts.append(page_text)

    return page_texts


async def store_embeddings_with_pages(file_id: str, filename: str, page_texts: List[str]):
    chunks_with_meta = chunk_text_with_pages(page_texts)

    if not chunks_with_meta:
        return

    documents = [c['text'] for c in chunks_with_meta]
    metadatas = [
        {
            "file_id": file_id,
            "filename": filename,
            "page_start": c['page_start'],
            "page_end": c['page_end'],
            "chunk_index": i
        }
        for i, c in enumerate(chunks_with_meta)
    ]
    chunk_ids = [hashlib.md5(f"{file_id}_{i}".encode()).hexdigest() for i in range(len(documents))]

    collection.delete(where={"file_id": file_id})

    collection.add(
        embeddings=embedding_function(documents),
        documents=documents,
        metadatas=metadatas,
        ids=chunk_ids,
    )
    logger.info(f"Stored {len(documents)} page-aware chunks for {filename}")


def generate_pdf_from_text(text: str) -> BytesIO:
    pdf = FPDF(format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_margins(15, 15, 15)

    if not os.path.exists(FONT_PATH):
        raise HTTPException(status_code=500, detail="Font file missing on server")

    pdf.add_font(fname=FONT_PATH, uni=True)
    pdf.set_font("DejaVuSans", size=11)

    cleaned = clean_text(text)
    pdf.add_page()
    pdf.multi_cell(0, 7, cleaned)

    output = BytesIO()
    output.write(pdf.output())
    output.seek(0)
    return output


# ========================= BACKGROUND TASK =========================

async def background_extraction_task(file_id: str, pdf_bytes: bytes, filename: str, db: Session):
    file_record = db.query(FileRecord).filter(FileRecord.id == file_id).first()
    if not file_record:
        return

    file_record.status = FileStatus.PROCESSING
    db.commit()

    try:
        images = await pdf_to_images(pdf_bytes)
        page_texts = await extract_text_from_images_per_page(images)

        full_text = "\n\n".join(page_texts)
        file_record.extracted_text = full_text
        file_record.status = FileStatus.COMPLETED
        db.commit()

        await store_embeddings_with_pages(file_id, filename, page_texts)
        logger.info(f"Extraction completed for {filename} ({len(page_texts)} pages)")

    except Exception as e:
        file_record.status = FileStatus.FAILED
        file_record.error_message = str(e)
        logger.error(f"Extraction failed for {file_id}: {e}")
    finally:
        file_record.updated_at = datetime.utcnow()
        db.commit()


async def background_podcast_task(podcast_id: str, request: PodcastCreateRequest, db: Session):
    """
    Background task to generate a podcast script and corresponding audio file.
    """
    podcast = db.query(PodcastRecord).filter(PodcastRecord.id == podcast_id).first()
    if not podcast:
        return

    podcast.status = PodcastStatus.PROCESSING
    db.commit()

    try:
        script = await generate_podcast_script(
            file_ids=request.file_ids,
            style=request.style or "explainer",
            duration_minutes=request.duration_minutes or 20,
            language=request.language or "en",
            db=db,
        )
        podcast.script = script
        db.commit()

        audio_path = generate_podcast_audio(
            script=script,
            podcast_id=podcast_id,
            language=request.language or "en",
        )
        podcast.audio_path = audio_path
        podcast.status = PodcastStatus.COMPLETED
    except Exception as e:
        podcast.status = PodcastStatus.FAILED
        podcast.error_message = str(e)
        logger.error(f"Podcast generation failed for {podcast_id}: {e}")
    finally:
        podcast.updated_at = datetime.utcnow()
        db.commit()


# ========================= HYBRID SEARCH HELPERS =========================

def reciprocal_rank_fusion(results: List[Tuple[int, float]], k: int = 60) -> List[int]:
    score_dict = {}
    for rank_offset, (doc_idx, _) in enumerate(results):
        rank = rank_offset + 1
        score_dict[doc_idx] = score_dict.get(doc_idx, 0) + 1 / (k + rank)

    return sorted(score_dict.keys(), key=lambda i: score_dict[i], reverse=True)


# ========================= MULTI-DOCUMENT CONTRADICTION DETECTION =========================

async def detect_contradictions(question: str, sources: List[Dict]) -> Optional[str]:
    if len(sources) < 2:
        return None

    doc_excerpts = {}
    for s in sources:
        filename = s["filename"]
        if filename not in doc_excerpts:
            doc_excerpts[filename] = []
        doc_excerpts[filename].append(s["excerpt"])

    contradiction_prompt = f"""You are an expert analyst. Review the following excerpts from different documents and the user's question.

Question: {question}

"""
    for filename, excerpts in doc_excerpts.items():
        contradiction_prompt += f"\n--- From {filename} ---\n" + "\n\n".join(excerpts[:3]) + "\n"

    contradiction_prompt += """
Are there any contradictions or conflicting information between the documents regarding the question?
If yes, clearly state what contradicts what.
If no clear contradiction, respond with "No contradictions detected."

Respond in one short paragraph."""

    try:
        client = get_openai_client()
        response = await client.chat.completions.create(
            model="gemma3",
            messages=[{"role": "user", "content": contradiction_prompt}],
            temperature=0.3,
            max_tokens=512,
        )
        result = response.choices[0].message.content.strip()
        return result if "no contradictions" not in result.lower() else None
    except Exception as e:
        logger.warning(f"Contradiction detection failed: {e}")
        return None


# ========================= ROUTES =========================

@app.post("/app-extract-text", response_model=ExtractionResponse, tags=["Legacy"])
async def legacy_extract_text(file: UploadFile):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files allowed")

    pdf_bytes = await file.read()
    images = await pdf_to_images(pdf_bytes)
    page_texts = await extract_text_from_images_per_page(images)
    full_text = "\n\n".join(page_texts)

    return ExtractionResponse(extracted_text=full_text, page_count=len(images))


@app.post("/files/upload", response_model=FileUploadResponse, tags=["Files"])
async def upload_file(
    file: UploadFile,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    if not file.filename.lower().endswith(".pdf") or file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files supported")

    content = await file.read()
    file_id = str(uuid.uuid4())

    record = FileRecord(id=file_id, filename=file.filename, content_type=file.content_type)
    db.add(record)
    db.commit()

    background_tasks.add_task(background_extraction_task, file_id, content, file.filename, db)

    return FileUploadResponse(
        file_id=file_id,
        filename=file.filename,
        message="Upload successful. Processing in background."
    )


@app.get("/files/{file_id}", response_model=FileRetrieveResponse, tags=["Files"])
def get_file(file_id: str, db: Session = Depends(get_db)):
    record = db.query(FileRecord).filter(FileRecord.id == file_id).first()
    if not record:
        raise HTTPException(status_code=404, detail="File not found")
    return record


@app.get("/files/{file_id}/pdf", tags=["Files"])
def download_clean_pdf(file_id: str, db: Session = Depends(get_db)):
    record = db.query(FileRecord).filter(FileRecord.id == file_id).first()
    if not record:
        raise HTTPException(status_code=404, detail="File not found")
    if record.status != FileStatus.COMPLETED or not record.extracted_text:
        raise HTTPException(status_code=400, detail="Document not processed yet")

    pdf_io = generate_pdf_from_text(record.extracted_text)
    clean_name = f"clean_{record.filename}"

    return StreamingResponse(
        pdf_io,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{clean_name}"'},
    )


@app.post("/files/merge-pdf", tags=["Files"])
async def merge_pdfs(request: MergePdfRequest = Body(...), db: Session = Depends(get_db)):
    if not request.file_ids:
        raise HTTPException(status_code=400, detail="At least one file_id required")

    records = db.query(FileRecord).filter(FileRecord.id.in_(request.file_ids)).all()
    if len(records) != len(request.file_ids):
        raise HTTPException(status_code=404, detail="One or more files not found")

    for r in records:
        if r.status != FileStatus.COMPLETED or not r.extracted_text:
            raise HTTPException(status_code=400, detail=f"File {r.filename} not ready")

    pdf = FPDF(format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_margins(15, 15, 15)
    pdf.add_font(fname=FONT_PATH, uni=True)
    pdf.set_font("DejaVuSans", size=11)

    for record in records:
        pdf.add_page()
        pdf.multi_cell(0, 7, clean_text(record.extracted_text))

    output = BytesIO(pdf.output())
    output.seek(0)

    filename = (
        f"clean_{records[0].filename}"
        if len(records) == 1
        else f"merged_clean_{len(records)}_docs.pdf"
    )

    return StreamingResponse(
        output,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/files/", tags=["Files"])
def list_files(limit: int = 20, db: Session = Depends(get_db)):
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


# ========================= PODCAST ROUTES =========================


@app.post("/podcasts", response_model=PodcastCreateResponse, tags=["Podcasts"])
async def create_podcast(
    request: PodcastCreateRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    if not request.file_ids:
        raise HTTPException(status_code=400, detail="file_ids required")

    # Validate files exist and are processed
    records = db.query(FileRecord).filter(FileRecord.id.in_(request.file_ids)).all()
    if len(records) != len(request.file_ids):
        raise HTTPException(status_code=404, detail="Some files not found")
    for r in records:
        if r.status != FileStatus.COMPLETED:
            raise HTTPException(status_code=400, detail=f"File {r.filename} not processed")

    # If a podcast for the same set of documents already exists (or is in progress),
    # return that instead of creating a new one.
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
            # Reuse existing podcast
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
    podcast = db.query(PodcastRecord).filter(PodcastRecord.id == podcast_id).first()
    if not podcast:
        raise HTTPException(status_code=404, detail="Podcast not found")

    audio_url = None
    if podcast.audio_path and podcast.status == PodcastStatus.COMPLETED:
        audio_url = f"/podcasts/{podcast_id}/audio"

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


@app.get("/podcasts/{podcast_id}/audio", tags=["Podcasts"])
def stream_podcast_audio(podcast_id: str, db: Session = Depends(get_db)):
    podcast = db.query(PodcastRecord).filter(PodcastRecord.id == podcast_id).first()
    if not podcast:
        raise HTTPException(status_code=404, detail="Podcast not found")
    if podcast.status != PodcastStatus.COMPLETED or not podcast.audio_path:
        raise HTTPException(status_code=400, detail="Podcast not ready yet")
    if not os.path.exists(podcast.audio_path):
        raise HTTPException(status_code=404, detail="Audio file missing on server")

    return FileResponse(
        podcast.audio_path,
        media_type="audio/mpeg",
        filename=os.path.basename(podcast.audio_path),
    )


@app.post("/chat-with-document", tags=["Files"])
async def chat_with_documents(request: MultiChatRequest, db: Session = Depends(get_db)):
    if not request.file_ids:
        raise HTTPException(status_code=400, detail="file_ids required")

    records = db.query(FileRecord).filter(FileRecord.id.in_(request.file_ids)).all()
    if len(records) != len(request.file_ids):
        raise HTTPException(status_code=404, detail="Some files not found")

    for r in records:
        if r.status != FileStatus.COMPLETED:
            raise HTTPException(status_code=400, detail=f"File {r.filename} not processed")

    user_msg = [m for m in request.messages if m.role == "user"]
    if not user_msg:
        raise HTTPException(status_code=400, detail="No user question")
    question = user_msg[-1].content.lower()

    # === Hybrid Search ===
    vector_results = collection.query(
        query_embeddings=embedding_function([question]),
        n_results=20,
        where={"file_id": {"$in": request.file_ids}},
        include=["documents", "metadatas", "distances"],
    )

    vector_docs = vector_results["documents"][0] if vector_results["documents"] else []
    vector_metas = vector_results["metadatas"][0] if vector_results["metadatas"] else []
    vector_distances = vector_results["distances"][0] if vector_results["distances"] else []

    vector_ranked = [(i, dist) for i, dist in enumerate(vector_distances)]

    all_chunks = [d for d in vector_docs]
    bm25 = BM25Okapi([doc.lower().split() for doc in all_chunks])
    tokenized_query = question.split()
    bm25_scores = bm25.get_scores(tokenized_query)
    bm25_ranked = [(i, -score) for i, score in enumerate(bm25_scores) if score > 0]
    bm25_ranked.sort(key=lambda x: x[1], reverse=True)

    all_ranked_pairs = vector_ranked + bm25_ranked
    fused_indices = reciprocal_rank_fusion(all_ranked_pairs)[:20]

    # === Build context with token limit ===
    context_parts = []
    sources = []
    current_token_count = 0

    for idx in fused_indices:
        if idx >= len(vector_docs):
            continue

        doc = vector_docs[idx]
        estimated_tokens = len(doc.split()) * 1.3

        if current_token_count + estimated_tokens > MAX_CONTEXT_TOKENS:
            break

        context_parts.append(doc)
        current_token_count += estimated_tokens

        meta = vector_metas[idx]
        filename = meta.get("filename", "Unknown Document")
        page_start = meta.get("page_start")
        page_end = meta.get("page_end")
        page_citation = f"Page {page_start}"
        if page_end != page_start:
            page_citation = f"Pages {page_start}–{page_end}"

        sources.append({
            "filename": filename,
            "page": page_citation,
            "excerpt": doc.strip(),
            "relevance_score": round(1 - vector_distances[idx] if idx < len(vector_distances) else 0.5, 3)
        })

    context = "\n\n".join(context_parts) if context_parts else "No relevant content was found in the document."

    # === Multi-document aware system prompt ===
    doc_list = ", ".join(set(s["filename"] for s in sources))
    system_prompt = f"""You are an expert assistant analyzing multiple documents: {doc_list}.

When answering:
- Always explicitly mention which document each fact comes from (e.g., "According to Contract.pdf...", "The NDA states...").
- If information differs between documents, clearly note it.
- Be precise and cite page numbers when available.

If the answer cannot be found, say "I don't know".

Context:
{context}

Answer clearly and professionally."""

    # === Trim conversation history ===
    recent_messages = []
    history_token_estimate = 0

    for msg in reversed(request.messages[-12:]):
        msg_tokens = len(msg.content.split()) * 1.3
        if history_token_estimate + msg_tokens > MAX_HISTORY_TOKENS:
            break
        recent_messages.insert(0, msg)
        history_token_estimate += msg_tokens

    full_messages = [
        {"role": "system", "content": system_prompt},
        *[{"role": m.role, "content": m.content} for m in recent_messages]
    ]

    try:
        client = get_openai_client()
        response = await client.chat.completions.create(
            model="gemma3",
            messages=full_messages,
            temperature=0.5,
            max_tokens=1024,
        )

        answer = response.choices[0].message.content.strip()

        # === Detect contradictions ===
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
                    "excerpt": s["excerpt"][:500],
                    "relevance_score": s["relevance_score"]
                }
                for s in sources[:5]
            ]
        }
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate response")


@app.delete("/files/{file_id}", tags=["Files"])
def delete_file(file_id: str, db: Session = Depends(get_db)):
    record = db.query(FileRecord).filter(FileRecord.id == file_id).first()
    if not record:
        raise HTTPException(status_code=404, detail="File not found")

    collection.delete(where={"file_id": file_id})
    db.delete(record)
    db.commit()
    return {"message": "File deleted successfully"}


# ========================= ENTRY POINT =========================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    uvicorn.run("main:app", host=args.host, port=args.port, reload=True)