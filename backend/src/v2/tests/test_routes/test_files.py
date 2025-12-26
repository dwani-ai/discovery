import uuid
from database.models import FileStatus

def test_upload_file(client, sample_pdf_bytes, monkeypatch):
    # Mock background task
    def mock_add_task(*args):
        pass
    monkeypatch.setattr("fastapi.BackgroundTasks.add_task", mock_add_task)

    response = client.post(
        "/files/upload",
        files={"file": ("test.pdf", sample_pdf_bytes, "application/pdf")}
    )
    assert response.status_code == 200
    data = response.json()
    assert "file_id" in data
    assert data["filename"] == "test.pdf"
    assert data["status"] == "pending"

def test_get_file_not_found(client):
    response = client.get("/files/nonexistent")
    assert response.status_code == 404

def test_list_files(client, db_session):
    file_id = str(uuid.uuid4())
    record = FileRecord(id=file_id, filename="listed.pdf", status=FileStatus.COMPLETED)
    db_session.add(record)
    db_session.commit()

    response = client.get("/files/")
    assert response.status_code == 200
    files = response.json()
    assert len(files) == 1
    assert files[0]["filename"] == "listed.pdf"