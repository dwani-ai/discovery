# tests/test_services/test_contradiction.py

import pytest
from services.contradiction import detect_contradictions

@pytest.mark.asyncio
async def test_detect_contradictions_none(mock_contradiction_none):
    sources = [
        {"filename": "doc1.pdf", "excerpt": "Salary is $100k"},
        {"filename": "doc2.pdf", "excerpt": "Salary is $100k"},
    ]
    result = await detect_contradictions("What is the salary?", sources)
    assert result is None

@pytest.mark.asyncio
async def test_detect_contradictions_found(mock_contradiction_found):
    sources = [
        {"filename": "doc1.pdf", "excerpt": "Salary is $100k"},
        {"filename": "doc2.pdf", "excerpt": "Salary is $120k"},
    ]
    result = await detect_contradictions("What is the salary?", sources)
    assert result is not None
    assert "100k" in result and "120k" in result