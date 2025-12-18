import argparse
import asyncio
import base64
import enum
import logging
import logging.config
import os
import uuid
from datetime import datetime
from io import BytesIO
from typing import List, Optional

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
from fastapi.responses import FileResponse, JSONResponse
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
        "https://*.hf.space",
        "https://dwani.ai",
        "https://*.dwani.ai",
        "https://dwani-*.hf.space",
        "http://localhost:11080",
        "http://localhost:3000",  # React dev
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------- Database Models --------------------------
from sqlalchemy import create_engine, Column, String, Text, DateTime
from sqlalchemy.dialects.sqlite import DATETIME
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
    status = Column(String, default=FileStatus.PENDING)  # Using String for simplicity
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

        db_file.extracted_text = result.strip()
        db_file.status = FileStatus.COMPLETED

    except Exception as e:
        db_file.status = FileStatus.FAILED
        db_file.error_message = str(e)
        logger.error(f"Extraction failed for {file_id}: {e}")

    finally:
        db_file.updated_at = datetime.utcnow()
        db.commit()


# -------------------------- Endpoints --------------------------

# Legacy endpoint (kept for compatibility)
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


# New Files API

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

import unicodedata
import re
from fastapi.responses import StreamingResponse


def clean_text(text: str) -> str:
    """Remove control characters that can cause rendering issues."""
    return ''.join(ch for ch in text if unicodedata.category(ch)[0] != 'C' or ch in '\n\r\t')


def insert_soft_hyphens(text: str, max_chars: int = 30) -> str:
    """Prevent long unbreakable strings from crashing PDF generation."""
    def replace(match):
        seq = match.group(0)
        return '\u00ad'.join([seq[i:i + max_chars] for i in range(0, len(seq), max_chars)])
    return re.sub(r'\S{' + str(max_chars + 1) + r'}', replace, text)


@app.get("/files/{file_id}/pdf", tags=["Files"])
def download_regenerated_pdf(file_id: str, db: Session = Depends(get_db)):
    record = db.query(FileRecord).filter(FileRecord.id == file_id).first()
    if not record:
        raise HTTPException(status_code=404, detail="File not found")

    if record.status != FileStatus.COMPLETED or not record.extracted_text:
        raise HTTPException(status_code=400, detail="Text extraction not complete or failed")

    # Generate PDF in memory
    pdf = FPDF(format='A4')
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.set_margins(left=20, top=20, right=20)

    font_path = os.path.join(os.path.dirname(__file__), "fonts", "DejaVuSans.ttf")
    if not os.path.exists(font_path):
        raise HTTPException(status_code=500, detail="Font file DejaVuSans.ttf not found")

    pdf.add_font(fname=font_path, uni=True)
    pdf.set_font("DejaVuSans", size=11)

    # Prepare text
    text = clean_text(record.extracted_text)
    text = insert_soft_hyphens(text, max_chars=30)

    # Write flowing text (safest method)
    pdf.write(h=7, txt=text)

    # Get bytes
    pdf_bytes = pdf.output()  # Returns bytes in fpdf2

    # === CRITICAL FIX: Use StreamingResponse, NOT FileResponse ===
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


# -------------------------- Run Server --------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the dwani.ai FastAPI server")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    uvicorn.run("fast_main:app", host=args.host, port=args.port, reload=True)