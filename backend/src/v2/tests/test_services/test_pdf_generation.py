import pytest
from services.pdf_generation import generate_pdf_from_text, generate_merged_pdf
from database.models import FileRecord
from io import BytesIO

def test_generate_pdf_from_text():
    text = "Hello World\nThis is a test document."
    pdf_io = generate_pdf_from_text(text)
    assert isinstance(pdf_io, BytesIO)
    pdf_bytes = pdf_io.getvalue()
    assert pdf_bytes.startswith(b"%PDF-1.")
    assert b"Hello World" in pdf_bytes

@pytest.fixture
def mock_records(db_session):
    rec1 = FileRecord(id="1", filename="doc1.pdf", extracted_text="First doc", status="completed")
    rec2 = FileRecord(id="2", filename="doc2.pdf", extracted_text="Second doc", status="completed")
    db_session.add_all([rec1, rec2])
    db_session.commit()
    return [rec1, rec2]

def test_generate_merged_pdf(mock_records):
    pdf_io, filename = generate_merged_pdf(mock_records)
    assert isinstance(pdf_io, BytesIO)
    assert filename == "merged_clean_2_docs.pdf"
    content = pdf_io.getvalue()
    assert b"First doc" in content
    assert b"Second doc" in content