# tests/integration/test_full_flow.py

import pytest
import time
from io import BytesIO
from fpdf import FPDF

from database.models import FileRecord, FileStatus

def create_sample_pdf() -> bytes:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Integration Test Document", ln=1)
    pdf.cell(200, 10, txt="This PDF has two pages.", ln=1)
    pdf.add_page()
    pdf.cell(200, 10, txt="Page 2: Contains important information.", ln=1)
    output = BytesIO()
    pdf.output(output)
    output.seek(0)
    return output.read()

@pytest.mark.asyncio
async def test_full_document_lifecycle(async_client, mock_llm):
    sample_pdf = create_sample_pdf()

    # 1. Upload PDF
    upload_response = await async_client.post(
        "/files/upload",
        files={"file": ("test_integration.pdf", sample_pdf, "application/pdf")}
    )
    assert upload_response.status_code == 200
    data = upload_response.json()
    file_id = data["file_id"]
    assert data["status"] == "pending"

    # 2. Wait for background processing (poll status)
    max_attempts = 20
    for _ in range(max_attempts):
        status_resp = await async_client.get(f"/files/{file_id}")
        if status_resp.status_code != 200:
            continue
        status_data = status_resp.json()
        if status_data["status"] == FileStatus.COMPLETED:
            break
        await asyncio.sleep(0.5)
    else:
        pytest.fail("Document did not finish processing in time")

    assert status_data["status"] == FileStatus.COMPLETED
    assert "Integration Test Document" in status_data["extracted_text"]
    assert "Page 2" in status_data["extracted_text"]

    # 3. Verify embeddings were stored
    chat_payload = {
        "file_ids": [file_id],
        "messages": [{"role": "user", "content": "What is this document about?"}]
    }
    chat_response = await async_client.post("/chat-with-document", json=chat_payload)
    assert chat_response.status_code == 200
    chat_data = chat_response.json()
    assert "answer" in chat_data
    assert len(chat_data["sources"]) > 0
    assert any("integration" in s["excerpt"].lower() for s in chat_data["sources"])

    # 4. Download clean PDF
    pdf_response = await async_client.get(f"/files/{file_id}/pdf")
    assert pdf_response.status_code == 200
    assert pdf_response.headers["content-type"] == "application/pdf"
    assert "clean_test_integration.pdf" in pdf_response.headers["content-disposition"]
    pdf_content = pdf_response.content
    assert b"Integration Test Document" in pdf_content

    # 5. Delete file and verify cleanup
    delete_response = await async_client.delete(f"/files/{file_id}")
    assert delete_response.status_code == 200

    not_found = await async_client.get(f"/files/{file_id}")
    assert not_found.status_code == 404