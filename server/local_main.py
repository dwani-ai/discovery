import os
import logging
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from io import BytesIO
from pdf2image import convert_from_path
import pytesseract
import json
from openai import AsyncOpenAI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="PDF Text Extract + GPT-OSS Chat API")

vlm_base_url = os.getenv('VLLM_IP', "0.0.0.0")

def get_openai_client():
    base_url = f"http://{vlm_base_url}:9500/v1"  # port for gpt-oss
    return AsyncOpenAI(api_key="http", base_url=base_url)

async def extract_text_from_pdf(uploaded_file: UploadFile) -> dict:
    """
    Extract text from all pages in the uploaded PDF using OCR (pytesseract).
    Returns a dict mapping page numbers to extracted text.
    """
    temp_path = "temp_extract.pdf"
    try:
        with open(temp_path, "wb") as f:
            f.write(await uploaded_file.read())

        images = convert_from_path(temp_path)
        page_texts = {}
        for i, image in enumerate(images):
            text = pytesseract.image_to_string(image)
            page_texts[i] = text.strip()
        return page_texts

    except Exception as e:
        logger.error(f"Failed to extract text: {e}")
        raise HTTPException(status_code=500, detail=f"Text extraction failed: {e}")

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.post("/process_pdf")
async def process_pdf(file: UploadFile = File(...), prompt: str = Form(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="File must be a PDF")
    if not prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt must be non-empty")

    # Extract text from PDF
    extracted_text = await extract_text_from_pdf(file)

    client = get_openai_client()

    # Combine extracted text with prompt
    try:
        extracted_text_str = json.dumps(extracted_text)
    except Exception as e:
        logger.error(f"Failed to serialize extracted text: {e}")
        raise HTTPException(status_code=500, detail="Failed to serialize extracted text")

    combined_prompt = f"{prompt}\n\nExtracted text from PDF:\n{extracted_text_str}"

    try:
        response = await client.chat.completions.create(
            model="gpt-oss",
            messages=[{"role": "user", "content": combined_prompt}],
            temperature=0.3,
            max_tokens=2048,
        )
        answer = response.choices[0].message.content
    except Exception as e:
        logger.error(f"GPT-OSS API call failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to process prompt with GPT-OSS")

    return JSONResponse(content={
        "response": answer,
        "extracted_text": extracted_text
    })

@app.post("/process_message")
async def process_message(prompt: str = Form(...), extracted_text: str = Form(...)):
    if not prompt.strip():
        raise HTTPException(status_code=400, detail="Please provide a non-empty prompt")
    if not extracted_text.strip():
        raise HTTPException(status_code=400, detail="Please provide non-empty extracted text")

    try:
        all_results = json.loads(extracted_text)
        if not isinstance(all_results, dict):
            raise ValueError("Extracted text must be a valid JSON object")
    except json.JSONDecodeError as e:
        logger.error(f"Invalid extracted text format: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Invalid extracted text format: {str(e)}")

    client = get_openai_client()

    extracted_text_str = json.dumps(all_results)

    combined_prompt = f"{prompt}\n\nExtracted text from PDF:\n{extracted_text_str}"

    try:
        response = await client.chat.completions.create(
            model="gpt-oss",
            messages=[{"role": "user", "content": combined_prompt}],
            temperature=0.3,
            max_tokens=2048,
        )
        answer = response.choices[0].message.content
    except Exception as e:
        logger.error(f"GPT-OSS API call failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to process prompt with GPT-OSS")

    return JSONResponse(content={
        "response": answer,
        "extracted_text": all_results
    })