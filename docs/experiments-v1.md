Automatic metadata extraction from PDFs and Word files is usually done with a mix of two layers:  

1) file/container properties (author, title, creation date, etc.), and  
2) content-derived metadata (parties, dates, amounts, etc.) via OCR/NLP.[2][4][7]

## What “metadata” you can extract

- **Technical/file metadata**: title, author, subject, keywords, creator/producer, creation/modification date, page count, file size, file type.[2]
- **Business/content metadata** (via AI/OCR): parties to a contract, effective dates, amounts, project IDs, invoice numbers, etc., even from scanned PDFs and mixed layouts.[4][7]

For PDFs and Word you typically combine both types to populate your metadata index.

## PDFs: automatic extraction approaches

- **Direct XMP/info dictionary**: libraries can read PDF document info to get title, author, subject, keywords, creator, producer, and timestamps programmatically.[2]
- **Scripted extraction (e.g., Python)**:
  - Open the PDF in code, read its metadata dictionary, and map fields to your internal schema (e.g., `Title`, `Author`, `CreationDate`).[2]
  - This can be run in batch over folders or buckets to auto-populate a database.[2]
- **OCR/AI on content**:
  - For scanned or messy PDFs, send pages to an OCR+NLP service that extracts structured fields like contract values, renewal terms, or invoice totals.[7][4]

In practice, a pipeline will first read native metadata and then fall back to content analysis if fields are missing or low quality.[4][2]

## Word (DOC/DOCX): automatic extraction approaches

- **Office document properties**:
  - Word files store properties such as title, subject, author, manager, company, creation date, last modified date and total editing time, which can be read automatically.[6]
- **Automation/connector tools**:
  - Workflow tools and document services expose “Get metadata” or similar actions that return all properties for a Word document for use in automation flows.[6]
- **AI/contract analysis on DOC/DOCX**:
  - The same OCR/NLP-style analysis used for PDFs can be applied to Word content to pull out contract parties, key dates, and clause-level information.[7][4]

A common pattern is: read built‑in properties, infer tags (e.g., department, document type) from content, then store all fields as unified metadata per file.[4][6]

## AI‑based metadata extraction (PDF + Word)

- **OCR + NLP engines**:
  - Modern services accept PDFs and Word documents (including scans), run OCR if needed, and return structured JSON with fields like dates, names, totals, and IDs.[7][4]
- **Template or rule-based extraction**:
  - For recurring formats (invoices, standard contracts), template-based extractors can reliably map specific positions or patterns into structured fields.[7]

You can integrate these services in an ingestion pipeline: upload file → extract base metadata → run AI extraction → merge results → save to your vector/relational store.[4][7]

## Practical pipeline pattern

- Step 1: For each PDF/DOCX, read built‑in properties (title, author, dates, etc.) using a language/library suitable for your stack.[6][2]
- Step 2: If needed, run OCR/NLP over content to derive higher-level metadata (e.g., document type, parties, key dates).[4][7]
- Step 3: Normalize field names (e.g., `created_at`, `author`, `doc_type`, `parties`) and store them in your metadata index or database to support search and filtering.[7][2][4]

If you share your tech stack (Python/Node/Java, cloud provider, etc.), a more concrete library list and code sketch for PDFs and Word can be outlined.

[1](https://www.youtube.com/watch?v=BaOTtt5L2C4)
[2](https://pdf.minitool.com/news/extract-metadata-from-pdf.html)
[3](https://www.docupile.com/upload-auto-metadata-extraction/)
[4](https://fynk.com/en/blog/contract-metadata-extraction/)
[5](https://www.reddit.com/r/DataHoarder/comments/5zfuhv/pdf_library_extracting_metadata_ocr/)
[6](https://docs.pdf4me.com/integrations/power-automate/word/extract-metadata/)
[7](https://parseur.com/blog/ai-data-extraction)
[8](https://www.youtube.com/watch?v=FpBCJjxB-pg)
[9](https://community.spiceworks.com/t/extracting-data-created-from-metadata-of-word-excel-and-pdf-files/633784)

--


A good approach is to build one metadata index over all files in Chroma, use that for the first pass to pick the right file(s), and then do a second semantic search only inside the chosen file to extract the needed data.[1][3]

## Overall architecture

- **Ingestion step** (offline or background):
  - Walk a document store (e.g., S3/local folder/DB), compute embeddings for each file (or representative chunks), and store:
    - One Chroma collection for **file-level metadata** (first-pass search).
    - One Chroma collection for **content-level chunks** (second-pass search).
- **Serving step** (FastAPI):
  - Endpoint 1: search on metadata collection → returns candidate files.
  - Endpoint 2: given file ID and query → search inside that file’s chunks.

## Metadata design (first pass)

Store one record per file in a “files” collection in Chroma.[7][1]

Suggested fields:

- **id**: stable file ID (e.g., path, UUID).
- **document**: short description / title / first N lines of the file.
- **metadata**:
  - `filename`
  - `filepath` or `uri`
  - `filesize`
  - `filetype` (pdf, docx, csv, txt, etc.)
  - `created_at`, `modified_at`
  - `owner`, `department`, `project`
  - `tags` (user-assigned labels)
  - `domain` (hr, finance, tech, etc.)
  - `language`
- **Embedding**: embedding of concatenated text like:
  - title + tags + first N lines + important headers.

This lets you:

- Do semantic search on “what the file is about”.
- Filter by structured metadata: e.g. `where={"department": "finance", "filetype": "pdf"}`.[3][7]

## Content/chunk design (second pass)

Store chunks in a separate “chunks” collection in Chroma.[2][3]

For each file:

- Split its content into chunks (e.g., 512–1024 tokens, with overlap).
- For each chunk:
  - `id`: unique chunk ID.
  - `document`: chunk text.
  - `metadata`:
    - `file_id` (foreign key to files collection).
    - `filename`
    - `page` / `section` / `heading` if available.
    - Optional: `chunk_index`, `start_offset`, `end_offset`.

Second-pass queries use:

- `query_texts=[user_query]`
- `where={"file_id": "<chosen-file-id>"}`.[3]

This gives the best chunk(s) from within that file only.

## Ingestion workflow (pseudo-code)

### 1. Setup Chroma client and collections

```python
import chromadb
from chromadb.config import Settings

client = chromadb.Client(Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory="./chroma_data"
))

files_col = client.get_or_create_collection(name="files")
chunks_col = client.get_or_create_collection(name="file_chunks")
```

### 2. Index files (run as batch job)

```python
from pathlib import Path
from uuid import uuid4

def embed(text: str) -> list[float]:
    # Call your embedding model here (OpenAI, HF, etc.)
    ...

def iter_documents(root: str):
    for path in Path(root).rglob("*"):
        if not path.is_file():
            continue
        yield path

def extract_text_and_metadata(path: Path):
    # Implement per file type (pdf/docx/txt/...)
    text = path.read_text(errors="ignore")
    metadata = {
        "filename": path.name,
        "filepath": str(path),
        "filetype": path.suffix.lower(),
        "filesize": path.stat().st_size,
        # Add timestamps, owner, etc.
    }
    return text, metadata

def chunk_text(text: str, max_chars: int = 2000, overlap: int = 200):
    start = 0
    while start < len(text):
        end = start + max_chars
        yield text[start:end]
        start = end - overlap

def index_corpus(root: str):
    for path in iter_documents(root):
        file_id = str(uuid4())
        full_text, meta = extract_text_and_metadata(path)

        # File-level vector (metadata collection)
        summary_text = f"{meta['filename']} {full_text[:2000]}"
        files_col.add(
            ids=[file_id],
            documents=[summary_text],
            metadatas=[meta],
            embeddings=[embed(summary_text)]
        )

        # Chunk-level vectors (content collection)
        chunk_ids = []
        chunk_docs = []
        chunk_metas = []
        chunk_embs = []

        for idx, chunk in enumerate(chunk_text(full_text)):
            chunk_id = f"{file_id}::chunk-{idx}"
            chunk_ids.append(chunk_id)
            chunk_docs.append(chunk)
            chunk_metas.append({
                "file_id": file_id,
                "filename": meta["filename"],
                "chunk_index": idx
            })
            chunk_embs.append(embed(chunk))

        if chunk_ids:
            chunks_col.add(
                ids=chunk_ids,
                documents=chunk_docs,
                metadatas=chunk_metas,
                embeddings=chunk_embs
            )

    client.persist()
```

## FastAPI endpoints

### 1. First-pass: find candidate files

```python
from fastapi import FastAPI, Query
from pydantic import BaseModel

app = FastAPI()

class FileSearchResult(BaseModel):
    file_id: str
    score: float
    metadata: dict

@app.get("/files/search", response_model=list[FileSearchResult])
def search_files(
    q: str = Query(...),
    limit: int = 10,
    filetype: str | None = None,
    owner: str | None = None
):
    where = {}
    if filetype:
        where["filetype"] = filetype
    if owner:
        where["owner"] = owner

    query_embedding = embed(q)

    res = files_col.query(
        query_embeddings=[query_embedding],
        n_results=limit,
        where=where or None,
        include=["metadatas", "distances", "ids"]
    )

    ids = res["ids"][0]
    dists = res["distances"][0]
    metas = res["metadatas"][0]

    results = []
    for fid, dist, meta in zip(ids, dists, metas):
        results.append(FileSearchResult(
            file_id=fid,
            score=float(dist),
            metadata=meta
        ))
    return results
```

Chroma returns IDs, distances, metadatas and documents, which you can map directly to a response DTO.[1][3]

### 2. Second-pass: search inside one file

```python
class ChunkSearchResult(BaseModel):
    chunk_id: str
    score: float
    text: str
    metadata: dict

@app.get("/files/{file_id}/search", response_model=list[ChunkSearchResult])
def search_in_file(
    file_id: str,
    q: str = Query(...),
    limit: int = 5
):
    query_embedding = embed(q)

    res = chunks_col.query(
        query_embeddings=[query_embedding],
        n_results=limit,
        where={"file_id": file_id},
        include=["documents", "metadatas", "ids", "distances"]
    )

    ids = res["ids"][0]
    docs = res["documents"][0]
    metas = res["metadatas"][0]
    dists = res["distances"][0]

    results = []
    for cid, doc, meta, dist in zip(ids, docs, metas, dists):
        results.append(ChunkSearchResult(
            chunk_id=cid,
            score=float(dist),
            text=doc,
            metadata=meta
        ))
    return results
```

This uses a metadata filter `where={"file_id": file_id}` to restrict the search to a single file’s chunks.[7][3]

## Notes and extensions

- You can store **only metadata** in the first collection and keep raw content in your own storage; Chroma will still index and query on an embedding of a short summary or title.[1]
- For “lookup” style queries (e.g., exact file name), first check a structured store (SQL or the metadata fields in Chroma with `where`), and fall back to semantic search when needed.[3][7]
- To support “multi-file second pass,” change `where` to a list of `file_id`s or apply a broader metadata filter (e.g., same project).[3]

If you share your preferred embedding provider and document store (S3, local, database), a more concrete wiring with configuration and dependency injection for FastAPI can be sketched.

[1](https://realpython.com/chromadb-vector-database/)
[2](https://www.youtube.com/watch?v=mhitt9QV3fE)
[3](https://anderfernandez.com/en/blog/chroma-vector-database-tutorial/)
[4](https://www.youtube.com/watch?v=JjSCezpZbI0)
[5](https://stackoverflow.com/questions/76184540/get-all-documents-from-chromadb-using-python-and-langchain)
[6](https://github.com/langchain-ai/langchain/discussions/17405)
[7](https://www.geeksforgeeks.org/nlp/introduction-to-chromadb/)
[8](https://www.youtube.com/watch?v=YoIOct5zIA4)
[9](https://www.youtube.com/watch?v=QojA5urIDzA)
[10](https://www.dataquest.io/blog/metadata-filtering-and-hybrid-search-for-vector-databases/)

--

Upload code, 
Ask it to create a system diagram usable with draw.io 

Summarise the project 

Explain important items


Build a metadata for file search/ lookup from all documents 


First pass - look into metadata to find the right file.

Second pass - extract data from single file.


