import logging
import json
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import JSONResponse
from openai import AsyncOpenAI
import base64
from io import BytesIO
from pdf2image import convert_from_path
import os
import asyncio
import re
from typing import Optional
from starlette.middleware.base import BaseHTTPMiddleware
import time

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Dwani PDF Processing API")

# Middleware to measure request processing time
class TimingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        response = await call_next(request)
        processing_time = time.time() - start_time
        logger.info(f"Request: {request.method} {request.url.path} took {processing_time:.3f} seconds")
        return response

app.add_middleware(TimingMiddleware)

vlm_base_url = os.getenv('VLLM_IP', "0.0.0.0")

def encode_image(image: BytesIO) -> str:
    """Encode image bytes to base64 string."""
    return base64.b64encode(image.read()).decode("utf-8")

def get_openai_client(model: str) -> AsyncOpenAI:
    """Initialize AsyncOpenAI client with model-specific base URL."""
    valid_models = ["gemma3", "gpt-oss"]
    if model not in valid_models:
        raise ValueError(f"Invalid model: {model}. Choose from: {', '.join(valid_models)}")
    
    model_ports = {"gemma3": "9000", "gpt-oss": "9500"}
    base_url = f"http://{vlm_base_url}:{model_ports[model]}/v1"
    return AsyncOpenAI(api_key="http", base_url=base_url)

def clean_response(raw_response: str) -> Optional[str]:
    """Clean markdown code blocks or other non-JSON content from the response."""
    if not raw_response:
        return None
    cleaned = re.sub(r'```(?:json)?\s*([\s\S]*?)\s*```', r'\1', raw_response)
    return cleaned.strip()

async def process_single_page(client, model, image, page_idx):
    """Process a single PDF page asynchronously."""
    try:
        image_bytes_io = BytesIO()
        image.save(image_bytes_io, format='JPEG', quality=85)
        image_bytes_io.seek(0)
        image_base64 = encode_image(image_bytes_io)
        
        message = [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
            {
                "type": "text",
                "text": (
                    f"Extract plain text from this PDF page (page {page_idx}). "
                    f"Return the result as a JSON object with the key as page number ({page_idx}) "
                    "and the value as the extracted text. Ensure JSON format."
                )
            }
        ]
        
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": message}],
            temperature=0.2,
            max_tokens=29695
        )
        raw_response = response.choices[0].message.content
        logger.debug(f"Raw response for page {page_idx}: {raw_response}")

        cleaned_response = clean_response(raw_response)
        if not cleaned_response:
            logger.warning(f"Empty response for page {page_idx}")
            return None, page_idx

        try:
            page_result = json.loads(cleaned_response)
            if not isinstance(page_result, dict) or str(page_idx) not in page_result:
                logger.warning(f"Invalid JSON for page {page_idx}")
                return None, page_idx
            return page_result, None
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing failed for page {page_idx}: {str(e)}")
            return None, page_idx
    except Exception as e:
        logger.error(f"Failed to process page {page_idx}: {str(e)}")
        return None, page_idx

async def render_pdf_to_png(pdf_file):
    """Convert PDF to images."""
    try:
        with open("temp.pdf", "wb") as f:
            f.write(await pdf_file.read())
        images = convert_from_path("temp.pdf")
    except Exception as e:
        logger.error(f"PDF conversion failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to convert PDF to images: {str(e)}")
    finally:
        if os.path.exists("temp.pdf"):
            os.remove("temp.pdf")
    return images

@app.post("/process_pdf")
async def process_pdf(file: UploadFile = File(...), prompt: str = Form(...)):
    """Endpoint to process PDF and extract text based on prompt."""
    if not file:
        raise HTTPException(status_code=400, detail="Please upload a PDF file")
    if not prompt.strip():
        raise HTTPException(status_code=400, detail="Please provide a non-empty prompt")

    images = await render_pdf_to_png(file)
    all_results = {}
    skipped_pages = []
    model = "gemma3"
    client = get_openai_client(model)

    # Process each page sequentially
    for page_idx, image in enumerate(images):
        page_result, skipped_idx = await process_single_page(client, model, image, page_idx)
        if page_result:
            all_results.update(page_result)
        if skipped_idx is not None:
            skipped_pages.append(skipped_idx)

    if not all_results and skipped_pages:
        return JSONResponse(
            content={"error": "No valid text extracted from any pages", "skipped_pages": skipped_pages},
            status_code=400
        )

    # Process with the provided prompt
    dwani_prompt = "You are dwani, a helpful assistant. Provide a concise response in one sentence maximum."
    try:
        results_str = json.dumps(all_results)
    except Exception as e:
        logger.error(f"Failed to serialize all_results: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to serialize extracted text: {str(e)}")

    combined_prompt = f"{dwani_prompt}\nUser prompt: {prompt}\nExtracted text: {results_str}"

    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": [{"type": "text", "text": combined_prompt}]}],
            temperature=0.3,
            max_tokens=29695
        )
        generated_response = response.choices[0].message.content
        return {
            "response": generated_response,
            "extracted_text": all_results,
            "skipped_pages": skipped_pages
        }
    except Exception as e:
        logger.error(f"Final API request failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Final API request failed: {str(e)}")

@app.post("/process_message")
async def process_message(prompt: str = Form(...), extracted_text: str = Form(...)):
    """Endpoint to process a query using extracted text."""
    if not prompt.strip():
        raise HTTPException(status_code=400, detail="Please provide a non-empty prompt")
    if not extracted_text.strip():
        raise HTTPException(status_code=400, detail="Please provide non-empty extracted text")

    model = "gemma3"
    client = get_openai_client(model)

    try:
        all_results = json.loads(extracted_text)
        if not isinstance(all_results, dict):
            raise ValueError("Extracted text must be a valid JSON object")
    except json.JSONDecodeError as e:
        logger.error(f"Invalid extracted text format: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Invalid extracted text format: {str(e)}")

    dwani_prompt = "You are dwani, a helpful assistant. Provide a concise response in one sentence maximum."
    combined_prompt = f"{dwani_prompt}\nUser prompt: {prompt}\nExtracted text: {json.dumps(all_results)}"

    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": [{"type": "text", "text": combined_prompt}]}],
            temperature=0.3,
            max_tokens=29695
        )
        generated_response = response.choices[0].message.content
        return {
            "response": generated_response,
            "extracted_text": all_results,
            "skipped_pages": []
        }
    except Exception as e:
        logger.error(f"Final API request failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Final API request failed: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint to verify API is operational."""
    return {"status": "healthy", "message": "API is operational"}