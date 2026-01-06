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
from typing import List, Optional, Dict, Tuple

import chromadb
from chromadb.utils import embedding_functions
from fastapi import FastAPI, UploadFile, HTTPException, BackgroundTasks, Depends, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
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


# utils/token_counter.py
import tiktoken
from typing import Optional

# We'll use cl100k_base — most common for recent OpenAI models & many open models
# If you later switch to a model that uses a different tokenizer, you can make this configurable
DEFAULT_ENCODING = "cl100k_base"

try:
    _tokenizer = tiktoken.get_encoding(DEFAULT_ENCODING)
except Exception as e:
    raise RuntimeError(f"Failed to load tiktoken encoding '{DEFAULT_ENCODING}': {e}")

def count_tokens(text: str, encoding_name: str = DEFAULT_ENCODING) -> int:
    """
    Count the number of tokens in a string using tiktoken.
    Uses cl100k_base by default (good approximation for most recent models).
    """
    if not text:
        return 0
    
    try:
        return len(_tokenizer.encode(text, disallowed_special=()))
    except Exception:
        # Fallback in case of rare encoding errors
        return len(text) // 4 + len(text.split()) // 2


def count_messages_tokens(messages: list[dict], encoding_name: str = DEFAULT_ENCODING) -> int:
    """
    Estimate total tokens for a list of chat messages (including role + ~4–8 overhead per message)
    """
    total = 0
    for msg in messages:
        content = msg.get("content", "")
        role = msg.get("role", "user")
        
        # Rough OpenAI-style overhead
        overhead = 4 if role == "system" else 3  # system gets slightly more
        total += count_tokens(content, encoding_name) + overhead
    
    # Final reply overhead (~3 tokens)
    total += 3
    return total

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


from typing import List, Dict, Optional
from fastapi import HTTPException, Depends
from sqlalchemy.orm import Session


@app.post("/chat-with-document", tags=["Files"])
async def chat_with_documents(
    request: MultiChatRequest,
    db: Session = Depends(get_db)
):
    """
    Multi-document RAG chat endpoint with hybrid search, page-aware citations,
    contradiction detection and realistic token-aware context building.
    """
    if not request.file_ids:
        raise HTTPException(status_code=400, detail="At least one file_id is required")

    # Load file records
    records = db.query(FileRecord).filter(FileRecord.id.in_(request.file_ids)).all()
    if len(records) != len(request.file_ids):
        raise HTTPException(status_code=404, detail="One or more files not found")

    for record in records:
        if record.status != FileStatus.COMPLETED or not record.extracted_text:
            raise HTTPException(
                status_code=400,
                detail=f"File '{record.filename}' is not yet fully processed"
            )

    # Get the latest user question
    user_messages = [m for m in request.messages if m.role == "user"]
    if not user_messages:
        raise HTTPException(status_code=400, detail="No user question found")
    question = user_messages[-1].content.strip()

    # ─────────────────────────────────────────────────────────────
    # 1. Hybrid Retrieval (Vector + BM25 + RRF)
    # ─────────────────────────────────────────────────────────────
    vector_results = collection.query(
        query_embeddings=embedding_function([question]),
        n_results=25,
        where={"file_id": {"$in": request.file_ids}},
        include=["documents", "metadatas", "distances"]
    )

    vector_docs = vector_results["documents"][0] or []
    vector_metas = vector_results["metadatas"][0] or []
    vector_distances = vector_results["distances"][0] or []

    # BM25 ranking on the retrieved candidates
    if vector_docs:
        tokenized_docs = [doc.lower().split() for doc in vector_docs]
        bm25 = BM25Okapi(tokenized_docs)
        tokenized_query = question.lower().split()
        bm25_scores = bm25.get_scores(tokenized_query)
        bm25_ranked = sorted(
            [(i, score) for i, score in enumerate(bm25_scores) if score > 0],
            key=lambda x: x[1],
            reverse=True
        )
    else:
        bm25_ranked = []

    # Simple vector ranking (lower distance = better)
    vector_ranked = [(i, dist) for i, dist in enumerate(vector_distances)]

    # Reciprocal Rank Fusion
    all_ranked_pairs = vector_ranked + [(i, -score) for i, score in bm25_ranked]  # negate BM25 so higher = better
    fused_indices = reciprocal_rank_fusion(all_ranked_pairs, k=60)[:20]

    # ─────────────────────────────────────────────────────────────
    # 2. Build context — token-aware
    # ─────────────────────────────────────────────────────────────
    context_parts: List[str] = []
    sources: List[Dict] = []
    context_tokens_used = 0

    # Reserve tokens for system prompt, question, formatting, etc.
    RESERVED_TOKENS = 1200
    max_context_for_chunks = MAX_CONTEXT_TOKENS - RESERVED_TOKENS

    for rank_idx, chunk_idx in enumerate(fused_indices):
        if chunk_idx >= len(vector_docs):
            continue

        chunk_text = vector_docs[chunk_idx]
        chunk_tokens = count_tokens(chunk_text)

        if context_tokens_used + chunk_tokens > max_context_for_chunks:
            break

        context_parts.append(chunk_text)
        context_tokens_used += chunk_tokens

        meta = vector_metas[chunk_idx]
        filename = meta.get("filename", "Unknown")
        page_start = meta.get("page_start")
        page_end = meta.get("page_end")

        page_str = f"Page {page_start}" if page_start == page_end else f"Pages {page_start}–{page_end}"

        sources.append({
            "filename": filename,
            "page": page_str,
            "excerpt": chunk_text.strip()[:600] + ("..." if len(chunk_text) > 600 else ""),
            "relevance_score": round(1 - vector_distances[chunk_idx], 3) if chunk_idx < len(vector_distances) else 0.4
        })

    context_block = "\n\n".join(context_parts) if context_parts else "[No relevant content found in the provided documents]"

    # ─────────────────────────────────────────────────────────────
    # 3. System prompt with multi-document awareness
    # ─────────────────────────────────────────────────────────────
    unique_filenames = ", ".join(sorted(set(s["filename"] for s in sources))) or "the document"
    
    system_prompt = f"""You are an expert analyst working with content from the following document(s): {unique_filenames}.

Guidelines:
• For every factual statement, clearly state which document it comes from (example: "According to {list(sources)[0]['filename']} on page X, ...")
• If information appears to conflict between documents, explicitly note the differences
• Use page references when available
• Be concise, accurate and professional
• If the answer is not supported by the provided context, respond only with: "I don't have sufficient information in the provided documents to answer this question."

Context:
{context_block}
"""

    # ─────────────────────────────────────────────────────────────
    # 4. Prepare conversation history (token-limited)
    # ─────────────────────────────────────────────────────────────
    recent_messages = []
    history_tokens = 0

    for msg in reversed(request.messages):
        # Rough per-message token estimate (content + role overhead)
        msg_content = f"{msg.role}: {msg.content}"
        msg_tokens = count_tokens(msg_content) + 8  # rough role + formatting overhead

        if history_tokens + msg_tokens > MAX_HISTORY_TOKENS:
            break

        recent_messages.insert(0, msg)
        history_tokens += msg_tokens

    # Final message list
    full_messages = [
        {"role": "system", "content": system_prompt},
        *[ {"role": m.role, "content": m.content} for m in recent_messages ]
    ]

    # Optional: final size check (for logging / debugging)
    total_input_tokens = count_messages_tokens(full_messages)

    # ─────────────────────────────────────────────────────────────
    # 5. Call LLM
    # ─────────────────────────────────────────────────────────────
    try:
        client = get_openai_client()
        response = await client.chat.completions.create(
            model="gemma3",
            messages=full_messages,
            temperature=0.55,
            max_tokens=1400,
            top_p=0.92,
        )

        answer = response.choices[0].message.content.strip()

        # Optional contradiction check on top sources
        contradiction_note = await detect_contradictions(question, sources[:8])

        # Sort sources by relevance for response
        sources.sort(key=lambda x: x["relevance_score"], reverse=True)

        final_answer = answer
        if contradiction_note:
            final_answer = f"⚠️ **Possible Contradiction Detected**\n\n{contradiction_note}\n\n**Answer:**\n{answer}"

        return {
            "answer": final_answer,
            "sources": [
                {
                    "filename": s["filename"],
                    "page": s["page"],
                    "excerpt": s["excerpt"],
                    "relevance_score": s["relevance_score"]
                }
                for s in sources[:6]   # top 6 usually sufficient
            ],
            "usage": {
                "context_tokens_used": context_tokens_used,
                "total_input_tokens_estimated": total_input_tokens,
                "max_context_limit": MAX_CONTEXT_TOKENS
            }
        }

    except Exception as e:
        logger.error(f"LLM generation failed", extra={
            "file_ids": request.file_ids,
            "question": question[:120],
            "input_tokens_approx": total_input_tokens,
            "error": str(e)
        })
        raise HTTPException(status_code=500, detail="Failed to generate answer")


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