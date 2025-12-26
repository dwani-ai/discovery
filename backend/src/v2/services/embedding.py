import hashlib
from utils.text import clean_text

def chunk_text_with_pages(page_texts: list[str], chunk_size: int = 800, overlap: int = 100) -> list[dict]:
    # (same implementation as before)
    ...

async def store_embeddings(file_id: str, filename: str, page_texts: list[str]):
    chunks = chunk_text_with_pages(page_texts)
    if not chunks:
        return

    documents = [c['text'] for c in chunks]
    metadatas = [{
        "file_id": file_id,
        "filename": filename,
        "page_start": c['page_start'],
        "page_end": c['page_end'],
        "chunk_index": i
    } for i, c in enumerate(chunks)]
    ids = [hashlib.md5(f"{file_id}_{i}".encode()).hexdigest() for i in range(len(documents))]

    from vectorstore.chroma import add_chunks
    add_chunks(file_id, filename, documents, metadatas, ids)