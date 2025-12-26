import pytest
from services.embedding import chunk_text_with_pages
from vectorstore.chroma import collection, add_chunks, delete_by_file_id

def test_chunk_text_with_pages():
    page_texts = ["Short page.", "This is a longer page with many words." * 10]
    chunks = chunk_text_with_pages(page_texts, chunk_size=5, overlap=1)

    assert len(chunks) >= 2
    assert chunks[0]["page_start"] == 1
    assert "Short page" in chunks[0]["text"]

def test_store_and_delete_embeddings():
    documents = ["chunk one", "chunk two"]
    metadatas = [{"file_id": "test123", "filename": "test.pdf", "page_start": 1, "page_end": 1, "chunk_index": i} for i in range(2)]
    ids = ["id1", "id2"]

    add_chunks("test123", "test.pdf", documents, metadatas, ids)

    results = collection.get(ids=ids)
    assert len(results["documents"]) == 2
    assert results["documents"][0] == "chunk one"

    delete_by_file_id("test123")
    results = collection.get(ids=ids)
    assert len(results["documents"]) == 0