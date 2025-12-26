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