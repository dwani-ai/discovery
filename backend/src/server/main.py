import argparse
import base64
import enum
import hashlib
import logging
import logging.config
import os
import uuid
from datetime import datetime
from io import BytesIO
from typing import List, Optional

import chromadb
from chromadb.utils import embedding_functions
from fastapi import FastAPI, UploadFile, HTTPException, BackgroundTasks, Depends, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fpdf import FPDF
from openai import AsyncOpenAI
from pdf2image import convert_from_bytes
from PIL import Image
from pydantic import BaseModel
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

FONT_PATH = os.path.join(os.path.dirname(__file__), "fonts", "DejaVuSans.ttf")
if not os.path.exists(FONT_PATH):
    logger.warning("DejaVuSans.ttf not found – PDF regeneration will fail until added.")

# ========================= FASTAPI APP =========================

app = FastAPI(
    title="dwani.ai API",
    description="Privacy-focused multimodal document extraction, regeneration, and RAG API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://app.dwani.ai", "http://localhost:5173", "http://127.0.0.1:5173"],
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
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")


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


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> List[str]:
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
        if i >= len(words) and len(chunks) > 1:
            break
    return chunks


def encode_image(image: Image.Image) -> str:
    buf = BytesIO()
    image.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


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
    return AsyncOpenAI(api_key="http", base_url=DWANI_API_BASE_URL)


# ========================= SERVICES =========================

async def extract_text_from_images(images: List[Image.Image]) -> str:
    client = get_openai_client()
    result = ""

    for i, img in enumerate(images):
        base64_img = encode_image(img)
        messages = [
            {"role": "system", "content": "You are an expert OCR assistant. Extract accurate plain text only."},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}},
                    {"type": "text", "text": f"Extract clean plain text from page {i+1}."},
                ],
            },
        ]

        response = await client.chat.completions.create(
            model="gemma3",
            messages=messages,
            temperature=0.2,
            max_tokens=2048,
        )
        result += response.choices[0].message.content.strip() + "\n\n"

    return result.strip()


async def store_embeddings(file_id: str, text: str):
    chunks = chunk_text(text)
    if not chunks:
        return

    chunk_ids = [hashlib.md5(f"{file_id}_{j}".encode()).hexdigest() for j in range(len(chunks))]
    collection.delete(where={"file_id": file_id})

    collection.add(
        embeddings=embedding_fn(chunks),
        documents=chunks,
        metadatas=[{"file_id": file_id, "chunk_index": j} for j in range(len(chunks))],
        ids=chunk_ids,
    )
    logger.info(f"Stored {len(chunks)} embeddings for file {file_id}")


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
        extracted = await extract_text_from_images(images)

        file_record.extracted_text = extracted
        file_record.status = FileStatus.COMPLETED
        db.commit()

        await store_embeddings(file_id, extracted)
        logger.info(f"Extraction completed for {filename} ({file_id})")

    except Exception as e:
        file_record.status = FileStatus.FAILED
        file_record.error_message = str(e)
        logger.error(f"Extraction failed for {file_id}: {e}")
    finally:
        file_record.updated_at = datetime.utcnow()
        db.commit()


# ========================= ROUTES =========================

@app.post("/app-extract-text", response_model=ExtractionResponse, tags=["Legacy"])
async def legacy_extract_text(file: UploadFile):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files allowed")

    pdf_bytes = await file.read()
    images = await pdf_to_images(pdf_bytes)
    text = await extract_text_from_images(images)

    return ExtractionResponse(extracted_text=text, page_count=len(images))


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
    question = user_msg[-1].content

    results = collection.query(
        query_embeddings=embedding_fn([question]),
        n_results=12,
        where={"file_id": {"$in": request.file_ids}},
        include=["documents", "metadatas", "distances"],
    )

    docs = results["documents"][0] if results["documents"] else []
    metas = results["metadatas"][0] if results["metadatas"] else []
    distances = results["distances"][0] if results["distances"] else []

    context_parts = []
    sources = []

    for doc, meta, dist in zip(docs, metas, distances):
        relevance = 1 - dist
        if relevance < 0.0:
            continue
        filename = next((r.filename for r in records if r.id == meta["file_id"]), "Unknown")
        context_parts.append(doc)
        sources.append({"filename": filename, "excerpt": doc[:300], "relevance_score": round(relevance, 3)})

    context = "\n\n".join(context_parts) or "No relevant content found."

    system_prompt = f"""You are an expert assistant. Answer using only the provided context.
If the answer is not in the context, say "I don't know".

Context:
{context}"""

    messages = [{"role": "system", "content": system_prompt}] + [
        {"role": m.role, "content": m.content} for m in request.messages
    ]

    client = get_openai_client()
    response = await client.chat.completions.create(
        model="gemma3",
        messages=messages,
        temperature=0.5,
        max_tokens=1024,
    )

    sources.sort(key=lambda x: x["relevance_score"], reverse=True)

    return {
        "answer": response.choices[0].message.content.strip(),
        "sources": sources[:5]
    }


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