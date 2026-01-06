from database.models import FileRecord, FileStatus
import uuid

def test_chat_requires_processed_files(client, db_session, monkeypatch):
    file_id = str(uuid.uuid4())
    record = FileRecord(id=file_id, filename="doc.pdf", status=FileStatus.PENDING)
    db_session.add(record)
    db_session.commit()

    payload = {
        "file_ids": [file_id],
        "messages": [{"role": "user", "content": "What is this document about?"}]
    }

    response = client.post("/chat-with-document", json=payload)
    assert response.status_code == 400
    assert "not processed" in response.json()["detail"]

# Add more tests for successful chat once embeddings are mocked

# tests/test_routes/test_chat.py

import pytest
import uuid
from database.models import FileRecord, FileStatus
from vectorstore.chroma import add_chunks
from unittest.mock import patch

@pytest.fixture
def processed_file_with_embeddings(db_session):
    file_id = str(uuid.uuid4())
    record = FileRecord(
        id=file_id,
        filename="test_doc.pdf",
        status=FileStatus.COMPLETED,
        extracted_text="This is a test document about apples and bananas."
    )
    db_session.add(record)
    db_session.commit()

    # Add fake embeddings
    docs = ["This is a test document about apples.", "Bananas are mentioned on page 2."]
    metas = [
        {"file_id": file_id, "filename": "test_doc.pdf", "page_start": 1, "page_end": 1, "chunk_index": 0},
        {"file_id": file_id, "filename": "test_doc.pdf", "page_start": 2, "page_end": 2, "chunk_index": 1},
    ]
    ids = [f"{file_id}_0", f"{file_id}_1"]
    add_chunks(file_id, "test_doc.pdf", docs, metas, ids)

    return file_id, record

@pytest.mark.asyncio
async def test_chat_with_document_success(
    client,
    processed_file_with_embeddings,
    mock_chat_response,           # Mocks final answer generation
    mock_contradiction_none       # Mocks contradiction check
):
    file_id, _ = processed_file_with_embeddings

    payload = {
        "file_ids": [file_id],
        "messages": [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "What fruits are mentioned?"}
        ]
    }

    response = client.post("/chat-with-document", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert "answer" in data
    assert "sources" in data
    assert len(data["sources"]) > 0
    assert data["sources"][0]["filename"] == "test_doc.pdf"
    assert "apples" in data["sources"][0]["excerpt"] or "bananas" in data["sources"][0]["excerpt"]

@pytest.mark.asyncio
async def test_chat_with_document_contradiction_warning(
    client,
    processed_file_with_embeddings,
    mock_chat_response,
    mock_contradiction_found
):
    file_id, _ = processed_file_with_embeddings

    payload = {
        "file_ids": [file_id],
        "messages": [{"role": "user", "content": "What is the salary?"}]
    }

    response = client.post("/chat-with-document", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert "⚠️ **Potential Contradiction Detected**" in data["answer"]
    assert "100k" in data["answer"] and "120k" in data["answer"]