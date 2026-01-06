# tests/integration/test_multi_document_chat.py

import pytest
from database.models import FileStatus

@pytest.mark.asyncio
async def test_multi_document_chat_and_contradiction(async_client, mock_llm):
    # Upload two different PDFs
    pdf1 = create_sample_pdf()  # Same function as above
    pdf2_bytes = create_sample_pdf()  # Slightly modify if needed

    ids = []
    for name in ["doc1.pdf", "doc2.pdf"]:
        resp = await async_client.post(
            "/files/upload",
            files={"file": (name, pdf1 if name == "doc1.pdf" else pdf2_bytes, "application/pdf")}
        )
        assert resp.status_code == 200
        ids.append(resp.json()["file_id"])

    # Wait for both to complete
    for file_id in ids:
        for _ in range(20):
            r = await async_client.get(f"/files/{file_id}")
            if r.json()["status"] == FileStatus.COMPLETED:
                break
            await asyncio.sleep(0.5)

    # Mock contradiction detection to return a warning
    with respx.mock:
        respx.post(f"{settings.DWANI_API_BASE_URL}/chat/completions").mock(side_effect=[
            httpx.Response(200, json={"choices": [{"message": {"content": "Answer based on both docs"}}]}),
            httpx.Response(200, json={"choices": [{"message": {"content": "Doc1 says X, Doc2 says Y – contradiction!"}}]})
        ])

        chat_resp = await async_client.post("/chat-with-document", json={
            "file_ids": ids,
            "messages": [{"role": "user", "content": "Compare the information"}]
        })

    assert chat_resp.status_code == 200
    answer = chat_resp.json()["answer"]
    assert "Potential Contradiction Detected" in answer
    assert "Doc1" in answer and "Doc2" in answer