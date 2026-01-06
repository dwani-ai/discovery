# tests/test_background/test_tasks.py

import pytest
from background.tasks import background_extraction_task
from database.models import FileStatus
from unittest.mock import AsyncMock

@pytest.mark.asyncio
async def test_background_extraction_task_success(db_session, mock_ocr_response):
    file_id = "mock123"
    pdf_bytes = b"%PDF-1.4 fake pdf content"
    filename = "test.pdf"

    # Mock pdf_to_images to return dummy images
    dummy_img = Image.new("RGB", (10, 10))
    with patch("background.tasks.pdf_to_images", return_value=[dummy_img, dummy_img]):
        with patch("background.tasks.store_embeddings", new=AsyncMock()):
            await background_extraction_task(file_id, pdf_bytes, filename, db_session)

    record = db_session.query(FileRecord).filter(FileRecord.id == file_id).first()
    assert record is not None
    assert record.status == FileStatus.COMPLETED
    assert "extracted text from page 1" in record.extracted_text.lower()