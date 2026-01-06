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

from sqlalchemy import Column, String, Text, DateTime, Integer, Float

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

    # ── New metadata columns ─────────────────────────────────────────────
    normalized_title     = Column(String, index=True, nullable=True)
    document_type        = Column(String, index=True, nullable=True)          # invoice, contract, report, ...
    detected_language    = Column(String, default="en", nullable=True)
    document_date        = Column(DateTime, nullable=True)
    year                 = Column(Integer, nullable=True)
    counterpart          = Column(String, index=True, nullable=True)
    total_amount         = Column(Float, nullable=True)
    tags                 = Column(String, nullable=True)                      # comma separated or JSON
    short_summary        = Column(Text, nullable=True)                        # 200–500 tokens


Base.metadata.create_all(bind=engine)

from sqlalchemy import text

def create_document_fts_table():
    """Create FTS5 virtual table for fast full-text document lookup"""
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE VIRTUAL TABLE IF NOT EXISTS document_fts USING fts5(
                file_id UNINDEXED,
                content,
                tokenize='unicode61'
            )
        """))
        conn.commit()

# Call it once at startup
create_document_fts_table()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def index_document_in_fts(
    file_id: str,
    filename: str,
    short_summary: str = "",
    document_type: str = "",
    counterpart: str = "",
    tags: str = "",
    db: Session
):
    """Index document-level searchable content into FTS5"""
    # Build rich searchable text
    parts = [
        filename,
        unicodedata.normalize("NFKD", filename.lower()).encode("ascii", "ignore").decode(),
        short_summary.strip(),
        document_type,
        counterpart,
        tags.replace(",", " "),
    ]
    searchable_content = " ".join(filter(None, parts))

    db.execute(
        text("""
            INSERT OR REPLACE INTO document_fts (file_id, content)
            VALUES (:file_id, :content)
        """),
        {"file_id": file_id, "content": searchable_content}
    )
    db.commit()

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

#====================== RAG EVAL ==============

import json
from typing import List, Dict, Tuple, Optional
from openai import AsyncOpenAI
import logging

logger = logging.getLogger(__name__)

async def llm_judge(
    prompt: str,
    client: AsyncOpenAI,
    model: str = "gemma3",
    temperature: float = 0.1,
    max_tokens: int = 350
) -> Optional[Dict]:
    """Helper: call LLM and try to parse JSON output"""
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        content = response.choices[0].message.content.strip()
        
        # Try to clean common markdown / code fence issues
        if content.startswith("```json"):
            content = content.split("```json", 1)[1].rsplit("```", 1)[0].strip()
        elif content.startswith("```"):
            content = content.split("```", 2)[1].strip()
            
        return json.loads(content)
    except Exception as e:
        logger.warning(f"LLM judge parsing failed: {e}", exc_info=True)
        return None


async def evaluate_rag_triad(
    question: str,
    contexts: List[str],
    answer: str,
    client: AsyncOpenAI,
    model: str = "gemma3"
) -> Dict[str, Dict[str, float | str]]:
    """
    Evaluate RAG response using three reference-free metrics.
    Returns dict with metric name → {score: float 0–1, explanation: str}
    """
    results = {}

    # ── 1. Faithfulness / Groundedness ────────────────────────────────────
    faithfulness_prompt = f"""You are an impartial evaluator checking if an answer is fully grounded in the given context.

CONTEXT (multiple chunks):
{chr(10).join(f"[{i+1}] {chunk.strip()}" for i, chunk in enumerate(contexts))}

QUESTION:
{question}

ANSWER:
{answer}

Instructions:
1. Break the ANSWER into individual factual statements/claims.
2. For each claim, verify if it is directly supported, strongly implied, or entailed by the CONTEXT.
   - Supported = clearly present (possibly paraphrased)
   - Partially supported = weak implication
   - Unsupported / contradicted / invented = hallucination
3. Score = (number of fully/partially supported claims) / (total claims)
   - Range: 0.0 (completely hallucinated) to 1.0 (perfectly grounded)

Return **only** valid JSON:
{{
  "score": <float between 0.0 and 1.0>,
  "explanation": "one or two sentence summary of findings",
  "claim_count": <int>,
  "supported_count": <int>
}}
"""

    faith_result = await llm_judge(faithfulness_prompt, client, model)
    if faith_result and "score" in faith_result:
        results["faithfulness"] = {
            "score": float(faith_result["score"]),
            "explanation": faith_result.get("explanation", "No explanation provided")
        }
    else:
        results["faithfulness"] = {"score": 0.0, "explanation": "Evaluation failed"}

    # ── 2. Answer Relevancy ───────────────────────────────────────────────
    relevancy_prompt = f"""Evaluate how relevant and focused the ANSWER is to the QUESTION.

QUESTION:
{question}

ANSWER:
{answer}

Scoring guidelines (0.0–1.0):
• 1.0 = Directly, concisely, and completely answers the question
• 0.7–0.9 = Mostly relevant, minor digressions
• 0.4–0.6 = Partially relevant or contains noticeable irrelevant content
• 0.0–0.3 = Mostly off-topic, evasive, or rambling

Return **only** JSON:
{{
  "score": <float 0.0–1.0>,
  "explanation": "one sentence explanation"
}}
"""

    rel_result = await llm_judge(relevancy_prompt, client, model)
    if rel_result and "score" in rel_result:
        results["answer_relevancy"] = {
            "score": float(rel_result["score"]),
            "explanation": rel_result.get("explanation", "")
        }
    else:
        results["answer_relevancy"] = {"score": 0.0, "explanation": "Evaluation failed"}

    # ── 3. Context Relevancy (average chunk relevance) ─────────────────────
    if not contexts:
        results["context_relevancy"] = {"score": 0.0, "explanation": "No context retrieved"}
    else:
        chunk_scores = []
        explanations = []

        for i, chunk in enumerate(contexts, 1):
            chunk_prompt = f"""Rate how relevant this CONTEXT chunk is to the QUESTION.

QUESTION:
{question}

CONTEXT CHUNK [{i}/{len(contexts)}]:
{chunk.strip()}

Score 0.0–1.0:
• 1.0 = highly relevant, necessary information
• 0.6–0.9 = useful / related
• 0.3–0.5 = marginally related
• 0.0–0.2 = irrelevant / noise

Return **only** JSON:
{{
  "score": <float>,
  "explanation": "short reason"
}}
"""

            chunk_eval = await llm_judge(chunk_prompt, client, model, max_tokens=120)
            if chunk_eval and "score" in chunk_eval:
                score = float(chunk_eval["score"])
                chunk_scores.append(score)
                explanations.append(f"Chunk {i}: {chunk_eval.get('explanation', '—')} ({score:.2f})")
            else:
                chunk_scores.append(0.0)

        if chunk_scores:
            avg_score = sum(chunk_scores) / len(chunk_scores)
            results["context_relevancy"] = {
                "score": round(avg_score, 3),
                "explanation": f"Average over {len(contexts)} chunks • " + " | ".join(explanations[:3])
            }
        else:
            results["context_relevancy"] = {"score": 0.0, "explanation": "Evaluation failed"}

    return results


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

import json
import logging
from datetime import datetime
from typing import Optional

from sqlalchemy.orm import Session
from openai import AsyncOpenAI

from .models import FileRecord
from .utils import get_openai_client, clean_text
from .services import index_document_in_fts   # assuming this function exists

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# LLM METADATA EXTRACTION PROMPT (production-ready, structured output)
# ──────────────────────────────────────────────────────────────────────────────

METADATA_EXTRACTION_PROMPT = """You are an accurate document metadata extractor. 
Analyze the provided document text excerpt and return ONLY a valid JSON object.

Rules:
- Be conservative: only extract information that is clearly and explicitly present.
- If a field is uncertain or not present → use null (not empty string or "unknown")
- Do NOT hallucinate dates, amounts, names, or types
- Infer document_type only if very confident
- Use ISO 8601 for dates (YYYY-MM-DD) when possible
- Normalize amounts: remove currency symbols → keep only the numeric value
- Currency: use ISO 4217 code (USD, EUR, GBP, ...) or null
- Language: use ISO 639-1 code (en, fr, de, es, it, ...)
- Summary: 1–3 concise sentences, max ~250 tokens

Return exactly this structure:

{
  "document_type":      string | null,   // invoice, receipt, contract, bank_statement, report, letter, email, terms_of_service, insurance_policy, tax_form, payslip, ...
  "title":              string | null,
  "document_date":      string | null,   // issuance / due / statement / signing date
  "year":               integer | null,
  "counterparty":       string | null,   // supplier, client, bank, employer, insurer, ...
  "total_amount":       number | null,
  "currency":           string | null,
  "language":           string,
  "short_summary":      string
}

Examples:

Input:
"INVOICE # INV-2025-0789
Tesla Energy Inc.
Date: December 15, 2025
Due: January 14, 2026
Powerwall 3 installation - 1 unit
Total: 12,450.00 EUR"

Output:
{
  "document_type": "invoice",
  "title": "Powerwall 3 Installation Invoice",
  "document_date": "2025-12-15",
  "year": 2025,
  "counterparty": "Tesla Energy Inc.",
  "total_amount": 12450.00,
  "currency": "EUR",
  "language": "en",
  "short_summary": "Invoice issued by Tesla Energy Inc. on December 15, 2025 for Powerwall 3 installation services totaling 12,450 EUR."
}

Now extract metadata from this document excerpt:

{excerpt}

Respond ONLY with valid JSON — no explanation, no markdown, no code fences.
"""


async def extract_metadata_with_llm(
    page_texts: list[str],
    model: str = "gemma3",           # or "qwen2.5-7b-instruct", "phi-4-mini", etc.
    max_pages_for_context: int = 4,
    max_chars: int = 12000
) -> dict:
    """
    Call LLM to extract structured metadata from document text.
    Returns dict with extracted fields (or empty on failure).
    """
    if not page_texts:
        return {}

    # Take first few pages — usually contains title, date, parties, type
    excerpt_pages = page_texts[:max_pages_for_context]
    excerpt = "\n\n".join(excerpt_pages)
    excerpt = clean_text(excerpt)[:max_chars]

    if len(excerpt.strip()) < 200:
        logger.warning("Excerpt too short for metadata extraction")
        return {}

    prompt = METADATA_EXTRACTION_PROMPT.format(excerpt=excerpt)

    client: AsyncOpenAI = get_openai_client(model=model)

    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=768,
            top_p=0.95,
        )

        content = response.choices[0].message.content.strip()

        # Clean possible markdown / fences
        if content.startswith("```json"):
            content = content.split("```json", 1)[1].rsplit("```", 1)[0].strip()
        elif content.startswith("```"):
            content = content.split("```", 2)[1].strip()

        meta = json.loads(content)

        # Basic validation / normalization
        if not isinstance(meta, dict):
            return {}

        # Ensure expected types
        meta["year"] = int(meta["year"]) if meta.get("year") else None
        meta["total_amount"] = float(meta["total_amount"]) if meta.get("total_amount") else None

        return meta

    except json.JSONDecodeError as e:
        logger.warning(f"Metadata JSON parse failed: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"Metadata extraction LLM call failed: {e}", exc_info=True)

    return {}


# ──────────────────────────────────────────────────────────────────────────────
# INTEGRATION INTO BACKGROUND PROCESSING TASK
# ──────────────────────────────────────────────────────────────────────────────

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

        # ── Basic fallback values ────────────────────────────────────────
        normalized = unicodedata.normalize("NFKD", filename.lower()).encode("ascii", "ignore").decode()
        file_record.normalized_title = normalized.replace(".pdf", "").strip()

        # Default short summary from first page(s)
        fallback_summary = "\n".join(page_texts[:2])[:800] + "..." if page_texts else ""
        file_record.short_summary = fallback_summary

        # ── LLM METADATA EXTRACTION ──────────────────────────────────────
        meta = await extract_metadata_with_llm(page_texts)

        if meta:
            logger.info(f"LLM metadata extracted for {filename}: {meta.get('document_type')} - {meta.get('title')}")

            file_record.document_type    = meta.get("document_type") or file_record.document_type
            file_record.normalized_title = meta.get("title") or file_record.normalized_title
            file_record.short_summary    = meta.get("short_summary") or file_record.short_summary
            file_record.counterpart      = meta.get("counterparty")
            file_record.total_amount     = meta.get("total_amount")
            file_record.currency         = meta.get("currency")
            file_record.detected_language = meta.get("language", "en")

            # Handle date
            doc_date_str = meta.get("document_date")
            if doc_date_str and isinstance(doc_date_str, str) and len(doc_date_str) >= 4:
                try:
                    dt = datetime.fromisoformat(doc_date_str)
                    file_record.document_date = dt
                    file_record.year = dt.year
                except ValueError:
                    pass

            # Use LLM year if date parsing failed but year is present
            if file_record.year is None and meta.get("year"):
                try:
                    file_record.year = int(meta["year"])
                except:
                    pass

        # Save updated record
        file_record.status = FileStatus.COMPLETED
        db.commit()

        # ── Store vector embeddings ──────────────────────────────────────
        await store_embeddings_with_pages(file_id, filename, page_texts)

        # ── Index enriched metadata into FTS5 for fast search ────────────
        index_document_in_fts(
            file_id=file_id,
            filename=filename,
            short_summary=file_record.short_summary or "",
            document_type=file_record.document_type or "",
            counterpart=file_record.counterpart or "",
            tags="",  # extend later if you add tags field
            db=db
        )

        logger.info(f"Processing completed for {filename} ({len(page_texts)} pages)")

    except Exception as e:
        file_record.status = FileStatus.FAILED
        file_record.error_message = str(e)
        logger.error(f"Extraction failed for {file_id}: {e}", exc_info=True)
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



from typing import List, Optional

async def select_relevant_file_ids(
    question: str,
    user_selected_file_ids: Optional[List[str]] = None,
    db: Session,
    max_files: int = 7,
) -> List[str]:
    """
    Fast first-pass: find most promising documents using FTS5
    """
    if user_selected_file_ids and len(user_selected_file_ids) <= max_files:
        return user_selected_file_ids

    # Prepare FTS5 query
    words = [w for w in question.lower().split() if len(w) >= 3]
    if not words:
        fts_query = question.lower()[:80]
    else:
        fts_query = " OR ".join(f'"{w}"' for w in words)

    sql = text("""
        SELECT file_id
        FROM document_fts
        WHERE content MATCH :query
        ORDER BY rank
        LIMIT :limit
    """)

    result = db.execute(sql, {"query": fts_query, "limit": max_files * 2})
    candidates = [row[0] for row in result.fetchall()]

    # Intersect with user-selected files if provided
    if user_selected_file_ids:
        candidates = [fid for fid in candidates if fid in set(user_selected_file_ids)]

    # Fallback: if nothing matched, return all user-selected (or empty)
    if not candidates and user_selected_file_ids:
        return user_selected_file_ids[:max_files]

    return candidates[:max_files]
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
import logging
from fastapi import HTTPException, Depends, Body
from sqlalchemy.orm import Session
from openai import AsyncOpenAI
from rank_bm25 import BM25Okapi
from sqlalchemy import text

from .models import FileRecord, FileStatus, MultiChatRequest, ChatMessage
from .database import get_db, engine
from .vector_store import collection, embedding_function
from .utils import count_tokens, count_messages_tokens, reciprocal_rank_fusion
from .services import detect_contradictions, evaluate_rag_triad, get_openai_client

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# PASS 1: Fast candidate document selection using FTS5
# ──────────────────────────────────────────────────────────────────────────────

async def select_relevant_file_ids(
    question: str,
    user_selected_file_ids: Optional[List[str]] = None,
    db: Session,
    max_files: int = 7,
) -> List[str]:
    """
    Fast first-pass: find most promising documents using FTS5 full-text search
    Returns ranked list of file_ids most likely relevant to the question
    """
    if user_selected_file_ids and len(user_selected_file_ids) <= max_files:
        return user_selected_file_ids

    # Prepare FTS5 query
    words = [w.strip() for w in question.lower().split() if len(w.strip()) >= 3]
    if not words:
        fts_query = question.lower()[:80].strip()
    else:
        fts_query = " OR ".join(f'"{w}"' for w in words if w)

    if not fts_query:
        fts_query = question.lower()[:60]

    sql = text("""
        SELECT file_id
        FROM document_fts
        WHERE content MATCH :query
        ORDER BY rank
        LIMIT :limit
    """)

    try:
        result = db.execute(sql, {"query": fts_query, "limit": max_files * 2})
        candidates = [row[0] for row in result.fetchall()]
    except Exception as e:
        logger.warning(f"FTS5 query failed: {e}", exc_info=True)
        candidates = []

    # Intersect with user-provided file_ids if any
    if user_selected_file_ids:
        candidates = [fid for fid in candidates if fid in set(user_selected_file_ids)]

    # Fallback: if no matches from FTS5, fall back to all provided files
    if not candidates and user_selected_file_ids:
        logger.info("No FTS5 matches → falling back to user-selected files")
        return user_selected_file_ids[:max_files]

    selected = candidates[:max_files]
    logger.debug(f"FTS5 selected {len(selected)} candidates: {selected}")
    return selected


# ──────────────────────────────────────────────────────────────────────────────
# MAIN CHAT ENDPOINT — Two-pass hybrid RAG
# ──────────────────────────────────────────────────────────────────────────────

@app.post("/chat-with-document", tags=["Files"])
async def chat_with_documents(
    request: MultiChatRequest,
    db: Session = Depends(get_db)
):
    """
    Multi-document RAG chat endpoint with:
    • Two-pass retrieval (FTS5 candidate selection → filtered hybrid RAG)
    • Page-aware citations
    • Basic contradiction detection
    • Lightweight RAG triad evaluation
    """
    if not request.file_ids:
        raise HTTPException(status_code=400, detail="At least one file_id is required")

    # Load and validate file records
    records = db.query(FileRecord).filter(FileRecord.id.in_(request.file_ids)).all()
    if len(records) != len(request.file_ids):
        raise HTTPException(status_code=404, detail="One or more files not found")

    for record in records:
        if record.status != FileStatus.COMPLETED or not record.extracted_text:
            raise HTTPException(
                status_code=400,
                detail=f"File '{record.filename}' is not yet fully processed"
            )

    # Get latest user question
    user_messages = [m for m in request.messages if m.role.lower() == "user"]
    if not user_messages:
        raise HTTPException(status_code=400, detail="No user question found")
    question = user_messages[-1].content.strip()

    # ── PASS 1: Candidate document selection ────────────────────────────────
    candidate_file_ids = await select_relevant_file_ids(
        question=question,
        user_selected_file_ids=request.file_ids,
        db=db,
        max_files=6           # ← tune this (4–10 is typical sweet spot)
    )

    if not candidate_file_ids:
        return {
            "answer": "No relevant documents matched this question in the provided files.",
            "sources": [],
            "usage": {
                "context_tokens_used": 0,
                "total_input_tokens_estimated": 0,
                "max_context_allowed": MAX_CONTEXT_TOKENS
            },
            "rag_evaluation": {}
        }

    logger.info(f"Pass 1 selected {len(candidate_file_ids)} candidate files")

    # ── PASS 2: Hybrid retrieval only on selected candidates ────────────────
    vector_results = collection.query(
        query_embeddings=embedding_function([question]),
        n_results=40,                      # more generous now that files are filtered
        where={"file_id": {"$in": candidate_file_ids}},
        include=["documents", "metadatas", "distances"]
    )

    vector_docs = vector_results["documents"][0] or []
    vector_metas = vector_results["metadatas"][0] or []
    vector_distances = vector_results["distances"][0] or []

    # BM25 re-ranking on retrieved chunks
    bm25_ranked = []
    if vector_docs:
        tokenized_docs = [doc.lower().split() for doc in vector_docs]
        bm25 = BM25Okapi(tokenized_docs)
        tokenized_query = question.lower().split()
        bm25_scores = bm25.get_scores(tokenized_query)
        bm25_ranked = [
            (i, score) for i, score in enumerate(bm25_scores) if score > 0
        ]
        bm25_ranked.sort(key=lambda x: x[1], reverse=True)

    # Reciprocal Rank Fusion
    vector_ranked = [(i, dist) for i, dist in enumerate(vector_distances)]
    combined_pairs = vector_ranked + [(i, -score) for i, score in bm25_ranked]
    fused_indices = reciprocal_rank_fusion(combined_pairs, k=60)[:20]

    # ── Build context (token-aware) ─────────────────────────────────────────
    context_parts: List[str] = []
    sources: List[Dict] = []
    context_tokens_used = 0
    RESERVED_TOKENS = 1200
    max_context_tokens = MAX_CONTEXT_TOKENS - RESERVED_TOKENS

    for idx in fused_indices:
        if idx >= len(vector_docs):
            continue

        chunk = vector_docs[idx]
        chunk_tokens = count_tokens(chunk)

        if context_tokens_used + chunk_tokens > max_context_tokens:
            break

        context_parts.append(chunk)
        context_tokens_used += chunk_tokens

        meta = vector_metas[idx]
        filename = meta.get("filename", "Unknown")
        page_start = meta.get("page_start")
        page_end = meta.get("page_end")
        page_str = f"Page {page_start}" if page_start == page_end else f"Pages {page_start}–{page_end}"

        sources.append({
            "filename": filename,
            "page": page_str,
            "excerpt": chunk.strip()[:600] + ("..." if len(chunk) > 600 else ""),
            "relevance_score": round(1 - vector_distances[idx], 3)
                if idx < len(vector_distances) else 0.45
        })

    context_block = "\n\n".join(context_parts) if context_parts else "[No relevant content found]"

    # ── System prompt ───────────────────────────────────────────────────────
    unique_docs = ", ".join(sorted(set(s["filename"] for s in sources))) or "the documents"
    system_prompt = f"""You are an expert analyst working with content extracted from: {unique_docs}.

Rules:
• Clearly state which document(s) and page(s) each fact comes from
• Note any apparent contradictions between documents
• Use page numbers when available
• Be concise, accurate and professional
• If the question cannot be answered from the provided context, reply only: "I don't have sufficient information in the provided documents to answer this question."

Context:
{context_block}
"""

    # ── Prepare conversation history (token limited) ────────────────────────
    recent_messages = []
    history_tokens = 0

    for msg in reversed(request.messages):
        content_for_count = f"{msg.role}: {msg.content}"
        msg_tokens = count_tokens(content_for_count) + 10
        if history_tokens + msg_tokens > MAX_HISTORY_TOKENS:
            break
        recent_messages.insert(0, msg)
        history_tokens += msg_tokens

    full_messages = [
        {"role": "system", "content": system_prompt},
        *[{"role": m.role, "content": m.content} for m in recent_messages]
    ]

    total_input_tokens = count_messages_tokens(full_messages)

    # ── Generate answer ─────────────────────────────────────────────────────
    client = get_openai_client()
    try:
        response = await client.chat.completions.create(
            model="gemma3",
            messages=full_messages,
            temperature=0.55,
            max_tokens=1400,
            top_p=0.92,
        )
        answer = response.choices[0].message.content.strip()
    except Exception as e:
        logger.error("LLM generation failed", extra={"error": str(e), "question": question[:180]})
        raise HTTPException(status_code=500, detail="Failed to generate answer")

    # ── Optional contradiction detection ────────────────────────────────────
    contradiction_note = None
    if len(sources) >= 2:
        contradiction_note = await detect_contradictions(question, sources[:10])

    final_answer = answer
    if contradiction_note and "no contradictions" not in contradiction_note.lower():
        final_answer = f"⚠️ **Possible Contradiction Detected**\n\n{contradiction_note}\n\n**Answer:**\n{answer}"

    # ── RAG evaluation (optional, lightweight) ──────────────────────────────
    eval_metrics = {
        "faithfulness": {"score": None, "explanation": "not evaluated"},
        "answer_relevancy": {"score": None, "explanation": "not evaluated"},
        "context_relevancy": {"score": None, "explanation": "not evaluated"}
    }

    try:
        eval_metrics = await evaluate_rag_triad(
            question=question,
            contexts=context_parts,
            answer=answer,
            client=client,
            model="gemma3"
        )
    except Exception as eval_err:
        logger.warning("RAG evaluation failed", exc_info=True)

    # ── Final response ──────────────────────────────────────────────────────
    sources_sorted = sorted(sources, key=lambda x: x["relevance_score"], reverse=True)

    return {
        "answer": final_answer,
        "sources": [
            {
                "filename": s["filename"],
                "page": s["page"],
                "excerpt": s["excerpt"],
                "relevance_score": s["relevance_score"]
            }
            for s in sources_sorted[:6]   # top 6 most relevant sources
        ],
        "usage": {
            "context_tokens_used": context_tokens_used,
            "total_input_tokens_estimated": total_input_tokens,
            "max_context_allowed": MAX_CONTEXT_TOKENS,
            "selected_documents": len(candidate_file_ids)
        },
        "rag_evaluation": eval_metrics,
        "debug_info": {  # optional — remove in production if desired
            "candidate_file_ids": candidate_file_ids,
            "retrieved_chunks": len(vector_docs),
            "fused_chunks_used": len(context_parts)
        }
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