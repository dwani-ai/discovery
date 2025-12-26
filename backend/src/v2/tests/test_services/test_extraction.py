# tests/test_services/test_extraction.py

import pytest
from PIL import Image
from io import BytesIO
from services.extraction import extract_text_from_images_per_page

@pytest.mark.asyncio
async def test_extract_text_from_images_per_page(mock_ocr_response):
    # Create a tiny dummy image
    img = Image.new("RGB", (100, 100), color="white")
    images = [img, img]  # Two pages

    page_texts = await extract_text_from_images_per_page(images)

    assert len(page_texts) == 2
    assert "This is extracted text from page 1." in page_texts[0]
    assert "Key points: Test document." in page_texts[0]
    assert page_texts[0] == page_texts[1]  # Same mock response