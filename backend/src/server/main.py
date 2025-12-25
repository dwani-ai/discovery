import argparse
import asyncio
import base64
import enum
import logging
import logging.config
import os
import uuid
import unicodedata
import re
import hashlib
from datetime import datetime
from io import BytesIO
from typing import List, Optional

import chromadb
from chromadb.utils import embedding_functions

from fastapi import (
    FastAPI,
    File,
    UploadFile,
    HTTPException,
    BackgroundTasks,
    Depends,
    Body,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from pydantic import BaseModel

import uvicorn
from openai import AsyncOpenAI
from pdf2image import convert_from_bytes
from PIL import Image

# For PDF regeneration
from fpdf import FPDF

# -------------------------- Logging Setup --------------------------
logging_config = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "simple": {"format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"},
    },
    "handlers": {
        "stdout": {
            "class": "logging.StreamHandler",
            "formatter": "simple",
            "stream": "ext://sys.stdout",
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "formatter": "simple",
            "filename": "dwani_api.log",
            "maxBytes": 10 * 1024 * 1024,  # 10MB
            "backupCount": 5,
        },
    },
    "loggers": {
        "root": {
            "level": "INFO",
            "handlers": ["stdout", "file"],
        },
    },
}

logging.config.dictConfig(logging_config)
logger = logging.getLogger("dwani_server")

# -------------------------- FastAPI App Setup --------------------------
app = FastAPI(
    title="dwani.ai API",
    description="A multimodal Inference API designed for Privacy – Text Extraction & Document Regeneration",
    version="1.0.0",
    openapi_tags=[
        {"name": "Legacy", "description": "Original direct extraction endpoint"},
        {"name": "Files", "description": "Persistent file upload, extraction, and regeneration"},
        {"name": "Utility", "description": "General utility endpoints"},
    ],
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://app.dwani.ai",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS", "DELETE"],
    allow_headers=["X-API-KEY", "Content-Type", "Accept"],
    max_age=600,
)

# -------------------------- Chroma Setup --------------------------
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="documents")

embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

# -------------------------- Database Models --------------------------
from sqlalchemy import create_engine, Column, String, Text, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

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

    id = Column(String, primary_key=True, index=True)
    filename = Column(String, index=True)
    content_type = Column(String)
    status = Column(String, default=FileStatus.PENDING)
    extracted_text = Column(Text, nullable=True)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


Base.metadata.create_all(bind=engine)


def get_db() -> Session:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


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


# -------------------------- Pydantic Models --------------------------
class ExtractionResponse(BaseModel):
    extracted_text: str
    page_count: int
    status: str = "success"


class FileUploadResponse(BaseModel):
    file_id: str
    filename: str
    status: str = "pending"
    message: str = "File uploaded successfully. Extraction in progress."


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


# -------------------------- Helper Functions --------------------------
def encode_image(image_io: BytesIO) -> str:
    return base64.b64encode(image_io.getvalue()).decode("utf-8")


def get_async_openai_client(model: str) -> AsyncOpenAI:
    valid_models = ["gemma3", "gpt-oss"]
    if model not in valid_models:
        raise ValueError(f"Invalid model: {model}")

    base_url = os.getenv("DWANI_API_BASE_URL")
    if not base_url:
        raise RuntimeError("DWANI_API_BASE_URL environment variable is not set")

    return AsyncOpenAI(api_key="http", base_url=base_url)


async def render_pdf_to_images(pdf_bytes: bytes) -> List[Image.Image]:
    try:
        images = convert_from_bytes(pdf_bytes, fmt="png")
        return images
    except Exception as e:
        logger.error(f"PDF to image conversion failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to process PDF pages")


def clean_text(text: str) -> str:
    """Remove control characters that can cause rendering issues."""
    return ''.join(ch for ch in text if unicodedata.category(ch)[0] != 'C' or ch in '\n\r\t')


# -------------------------- Background Task: Extraction + Storage --------------------------
async def extract_and_store(file_id: str, pdf_bytes: bytes, filename: str, db: Session):
    db_file = db.query(FileRecord).filter(FileRecord.id == file_id).first()
    if not db_file:
        return

    db_file.status = FileStatus.PROCESSING
    db.commit()

    try:
        images = await render_pdf_to_images(pdf_bytes)
        if not images:
            raise ValueError("No pages extracted from PDF")

        result = ""
        model = "gemma3"
        client = get_async_openai_client(model)

        for i, image in enumerate(images):
            img_io = BytesIO()
            image.save(img_io, format="JPEG", quality=85)
            img_io.seek(0)
            base64_img = encode_image(img_io)

            message_content = [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}},
                {
                    "type": "text",
                    "text": f"Extract plain text from page {i+1}. Preserve reading order, headings, lists, and paragraphs. Output clean text only.",
                },
            ]

            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert OCR assistant. Extract accurate plain text from document images without adding markdown unless necessary.",
                    },
                    {"role": "user", "content": message_content},
                ],
                temperature=0.2,
                max_tokens=2048,
            )

            page_text = response.choices[0].message.content.strip()
            result += page_text + "\n\n"

        cleaned_text = result.strip()
        db_file.extracted_text = cleaned_text
        db_file.status = FileStatus.COMPLETED

        # Create and store embeddings
        chunks = chunk_text(cleaned_text)
        chunk_ids = [hashlib.md5(f"{file_id}_{j}".encode()).hexdigest() for j in range(len(chunks))]

        collection.delete(where={"file_id": file_id})

        collection.add(
            embeddings=embedding_function(chunks),
            documents=chunks,
            metadatas=[{"file_id": file_id, "chunk_index": j} for j in range(len(chunks))],
            ids=chunk_ids
        )

        logger.info(f"Embedded {len(chunks)} chunks for file {file_id}")

    except Exception as e:
        db_file.status = FileStatus.FAILED
        db_file.error_message = str(e)
        logger.error(f"Extraction failed for {file_id}: {e}")

    finally:
        db_file.updated_at = datetime.utcnow()
        db.commit()


# -------------------------- Endpoints --------------------------

@app.post("/app-extract-text", response_model=ExtractionResponse, tags=["Legacy"])
async def extract_text_endpoint(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf") or file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    pdf_bytes = await file.read()
    images = await render_pdf_to_images(pdf_bytes)

    result = ""
    model = "gemma3"
    client = get_async_openai_client(model)

    for i, image in enumerate(images):
        img_io = BytesIO()
        image.save(img_io, format="JPEG", quality=85)
        img_io.seek(0)
        base64_img = encode_image(img_io)

        message_content = [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}},
            {"type": "text", "text": f"Extract clean plain text from page {i+1}."},
        ]

        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Extract accurate plain text only."},
                {"role": "user", "content": message_content},
            ],
            temperature=0.2,
            max_tokens=2048,
        )
        result += response.choices[0].message.content.strip() + "\n\n"

    return ExtractionResponse(
        extracted_text=result.strip(),
        page_count=len(images),
    )


@app.post("/files/upload", response_model=FileUploadResponse, tags=["Files"])
async def upload_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files supported")

    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Invalid file type")

    content = await file.read()
    file_id = str(uuid.uuid4())

    db_file = FileRecord(
        id=file_id,
        filename=file.filename,
        content_type=file.content_type,
        status=FileStatus.PENDING,
    )
    db.add(db_file)
    db.commit()
    db.refresh(db_file)

    background_tasks.add_task(extract_and_store, file_id, content, file.filename, db)

    return FileUploadResponse(
        file_id=file_id,
        filename=file.filename,
        message="Upload successful. Extraction in progress.",
    )


@app.get("/files/{file_id}", response_model=FileRetrieveResponse, tags=["Files"])
def get_file_status(file_id: str, db: Session = Depends(get_db)):
    record = db.query(FileRecord).filter(FileRecord.id == file_id).first()
    if not record:
        raise HTTPException(status_code=404, detail="File not found")

    return FileRetrieveResponse(
        file_id=record.id,
        filename=record.filename,
        status=record.status,
        extracted_text=record.extracted_text,
        error_message=record.error_message,
        created_at=record.created_at,
        updated_at=record.updated_at,
    )


@app.get("/files/{file_id}/pdf", tags=["Files"])
def download_regenerated_pdf(file_id: str, db: Session = Depends(get_db)):
    record = db.query(FileRecord).filter(FileRecord.id == file_id).first()
    if not record:
        raise HTTPException(status_code=404, detail="File not found")

    if record.status != FileStatus.COMPLETED or not record.extracted_text:
        raise HTTPException(status_code=400, detail="Text extraction not complete or failed")

    pdf = FPDF(format='A4')
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.set_margins(left=20, top=20, right=20)

    font_path = os.path.join(os.path.dirname(__file__), "fonts", "DejaVuSans.ttf")
    if not os.path.exists(font_path):
        raise HTTPException(status_code=500, detail="Font file DejaVuSans.ttf not found")

    pdf.add_font(fname=font_path, uni=True)
    pdf.set_font("DejaVuSans", size=11)

    text = clean_text(record.extracted_text)
    pdf.write(h=7, txt=text)

    pdf_bytes = pdf.output()

    return StreamingResponse(
        BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="clean_{record.filename}"'
        }
    )


@app.get("/files/", tags=["Files"])
def list_files(db: Session = Depends(get_db), limit: int = 20):
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
async def chat_with_document(request: MultiChatRequest, db: Session = Depends(get_db)):
    if not request.file_ids:
        raise HTTPException(status_code=400, detail="At least one file_id required")

    records = db.query(FileRecord).filter(FileRecord.id.in_(request.file_ids)).all()
    if len(records) != len(request.file_ids):
        raise HTTPException(status_code=404, detail="One or more files not found")

    for record in records:
        if record.status != FileStatus.COMPLETED:
            raise HTTPException(status_code=400, detail=f"Document {record.filename} is not ready")

    user_messages = [m for m in request.messages if m.role == "user"]
    if not user_messages:
        raise HTTPException(status_code=400, detail="No user question")
    question = user_messages[-1].content

    query_embedding = embedding_function([question])
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=6,
        where={"file_id": {"$in": request.file_ids}},
        include=["documents", "metadatas", "distances"]
    )

    relevant_chunks = results['documents'][0] if results['documents'] else []
    metadatas = results['metadatas'][0] if results['metadatas'] else []
    distances = results['distances'][0] if results['distances'] else []

    context_parts = []
    sources = []
    for i, (chunk, meta) in enumerate(zip(relevant_chunks, metadatas)):
        if distances[i] > 0.6:
            continue
        context_parts.append(chunk)
        filename = next(r.filename for r in records if r.id == meta['file_id'])
        sources.append({
            "filename": filename,
            "excerpt": chunk.strip(),
            "relevance_score": round(1 - distances[i], 3)
        })

    context = "\n\n".join(context_parts) if context_parts else "No highly relevant content found."

    system_prompt = f"""You are an expert assistant. Answer based ONLY on the provided document excerpts.
If the question cannot be answered from the context, say "I don't know".

Context:
{context}

Answer clearly and cite sources naturally where relevant."""

    full_messages = [
        {"role": "system", "content": system_prompt},
        *[{"role": m.role, "content": m.content} for m in request.messages]
    ]

    model = "gemma3"
    client = get_async_openai_client(model)

    try:
        response = await client.chat.completions.create(
            model=model,
            messages=full_messages,
            temperature=0.5,
            max_tokens=1024,
        )
        answer = response.choices[0].message.content.strip()

        sources.sort(key=lambda x: x['relevance_score'], reverse=True)

        return {
            "answer": answer,
            "sources": sources
        }
    except Exception as e:
        logger.error(f"Chat failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate response")


from fastapi.responses import StreamingResponse
from io import BytesIO
import os
import unicodedata

@app.post("/files/merge-pdf", tags=["Files"])
async def merge_pdf(request: MergePdfRequest = Body(...), db: Session = Depends(get_db)):
    if len(request.file_ids) == 0:
        raise HTTPException(status_code=400, detail="At least one file_id required")

    records = db.query(FileRecord).filter(FileRecord.id.in_(request.file_ids)).all()
    if len(records) != len(request.file_ids):
        raise HTTPException(status_code=404, detail="One or more files not found")

    for record in records:
        if record.status != FileStatus.COMPLETED or not record.extracted_text:
            raise HTTPException(
                status_code=400,
                detail=f"Document '{record.filename}' is not ready (status: {record.status})"
            )

    # Generate PDF
    pdf = FPDF(format='A4')
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_margins(15, 15, 15)

    font_path = os.path.join(os.path.dirname(__file__), "fonts", "DejaVuSans.ttf")
    if not os.path.exists(font_path):
        raise HTTPException(status_code=500, detail="Font file DejaVuSans.ttf not found on server")

    pdf.add_font(fname=font_path, uni=True)
    pdf.set_font("DejaVuSans", size=11)

    for record in records:
        pdf.add_page()
        cleaned_text = ''.join(
            ch for ch in record.extracted_text 
            if unicodedata.category(ch)[0] != 'C' or ch in '\n\r\t'
        )
        pdf.multi_cell(0, 7, cleaned_text)

    pdf_bytes = pdf.output()

    # Smart filename
    if len(records) == 1:
        base_name = records[0].filename.rsplit('.', 1)[0]
        filename = f"clean_{base_name}.pdf"
    else:
        filename = f"merged_clean_{len(records)}_documents.pdf"

    return StreamingResponse(
        BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"'
        }
    )


@app.delete("/files/{file_id}", tags=["Files"])
def delete_file(file_id: str, db: Session = Depends(get_db)):
    record = db.query(FileRecord).filter(FileRecord.id == file_id).first()
    if not record:
        raise HTTPException(status_code=404, detail="File not found")

    collection.delete(where={"file_id": file_id})

    db.delete(record)
    db.commit()

    return {"message": "File deleted successfully"}


# -------------------------- Run Server --------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the dwani.ai FastAPI server")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    uvicorn.run("main:app", host=args.host, port=args.port, reload=True)