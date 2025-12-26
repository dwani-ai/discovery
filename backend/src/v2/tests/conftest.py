import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from database.models import Base, FileRecord, create_tables
from database.session import get_db
from main import app
from config.settings import settings
import tempfile
import shutil
from pathlib import Path

@pytest.fixture(scope="session")
def test_db_path():
    # Use in-memory SQLite for speed, or temp file for persistence
    return "sqlite:///:memory:"

@pytest.fixture(scope="function")
def engine(test_db_path):
    eng = create_engine(test_db_path)
    Base.metadata.create_all(eng)
    yield eng
    Base.metadata.drop_all(eng)

@pytest.fixture(scope="function")
def db_session(engine):
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()

@pytest.fixture(scope="function")
def client(db_session, monkeypatch):
    # Override get_db dependency
    def override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = override_get_db

    # Use temporary directories for chroma and fonts if needed
    with tempfile.TemporaryDirectory() as tmpdir:
        monkeypatch.setattr("vectorstore.chroma.client", None)  # Reset client
        monkeypatch.setattr("chromadb.PersistentClient.path", str(Path(tmpdir) / "chroma"))

        yield TestClient(app)

    app.dependency_overrides.clear()

@pytest.fixture
def sample_pdf_bytes():
    # Create a minimal valid PDF (1x1 white pixel)
    from fpdf import FPDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Test PDF for OCR", ln=1)
    output = pdf.output()
    return output

# tests/conftest.py (add these fixtures)

import pytest
import respx
from httpx import Response
from config.settings import settings

@pytest.fixture
def mock_llm_router():
    """Mock all calls to the local LLM endpoint"""
    with respx.mock(base_url=settings.DWANI_API_BASE_URL, assert_all_mocked=True) as router:
        yield router

@pytest.fixture
def mock_ocr_response(mock_llm_router):
    """Mock successful OCR response for one page"""
    mock_llm_router.post("/chat/completions").respond(
        status_code=200,
        json={
            "choices": [
                {
                    "message": {
                        "content": "This is extracted text from page 1.\n\nKey points: Test document."
                    }
                }
            ]
        }
    )

@pytest.fixture
def mock_chat_response(mock_llm_router, request):
    """Parametrized mock for general chat completions (e.g. final answer)"""
    content = getattr(request, "param", "This is a helpful answer based on the documents.")
    mock_llm_router.post("/chat/completions").respond(
        status_code=200,
        json={
            "choices": [{"message": {"content": content}}]
        }
    )

@pytest.fixture
def mock_contradiction_none(mock_llm_router):
    mock_llm_router.post("/chat/completions").respond(
        status_code=200,
        json={
            "choices": [{"message": {"content": "No contradictions detected."}}]
        }
    )

@pytest.fixture
def mock_contradiction_found(mock_llm_router):
    mock_llm_router.post("/chat/completions").respond(
        status_code=200,
        json={
            "choices": [{"message": {"content": "The salary in Doc1 is $100k, but Doc2 says $120k."}}]
        }
    )