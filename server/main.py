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
from typing import List, Dict, Optional
import time
from starlette.middleware.base import BaseHTTPMiddleware

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Dwani PDF Processing API")

# Middleware to measure request processing time
class TimingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()  # Record start time
        response = await call_next(request)  # Process the request
        end_time = time.time()  # Record end time
        processing_time = end_time - start_time  # Calculate processing time
        logger.info(f"Request: {request.method} {request.url.path} took {processing_time:.3f} seconds")
        return response

# Add the middleware to the FastAPI app
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
    
    model_ports = {
        "gemma3": "9000",
        "gpt-oss": "9500",
    }
    base_url = f"http://{vlm_base_url}:{model_ports[model]}/v1"
    return AsyncOpenAI(api_key="http", base_url=base_url)

def clean_response(raw_response: str) -> Optional[str]:
    """Clean markdown code blocks or other non-JSON content from the response."""
    if not raw_response:
        return None
    cleaned = re.sub(r'```(?:json)?\s*([\s\S]*?)\s*```', r'\1', raw_response)
    return cleaned.strip()

async def process_single_batch(client, model, batch_messages, batch_start, batch_end):
    """Process a single batch of pages asynchronously."""
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": batch_messages}],
            temperature=0.2,
            max_tokens=29695
        )
        raw_response = response.choices[0].message.content
        logger.debug(f"Raw response for batch {batch_start}-{batch_end-1}: {raw_response}")

        cleaned_response = clean_response(raw_response)
        if not cleaned_response:
            logger.warning(f"Empty response for batch {batch_start}-{batch_end-1}")
            return None, list(range(batch_start, batch_end))

        try:
            batch_results = json.loads(cleaned_response)
            if not isinstance(batch_results, dict):
                logger.warning(f"Response is not a JSON object for batch {batch_start}-{batch_end-1}")
                return None, list(range(batch_start, batch_end))
            return batch_results, []
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing failed for batch {batch_start}-{batch_end-1}: {str(e)}")
            return None, list(range(batch_start, batch_end))
    except Exception as e:
        logger.error(f"API request failed for batch {batch_start}-{batch_end-1}: {str(e)}")
        return None, list(range(batch_start, batch_end))

async def process_single_page(client, model, image, page_idx):
    """Process a single skipped page asynchronously."""
    try:
        image_bytes_io = BytesIO()
        image.save(image_bytes_io, format='JPEG', quality=85)
        image_bytes_io.seek(0)
        image_base64 = encode_image(image_bytes_io)
        
        single_message = [
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
            },
            {
                "type": "text",
                "text": (
                    f"Extract plain text from this single PDF page (page number {page_idx}). "
                    "Return the result as a valid JSON object where the key is the page number "
                    f"({page_idx}) and the value is the extracted text. "
                    "Ensure the response is strictly JSON-formatted and does not include markdown code blocks."
                )
            }
        ]
        
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": single_message}],
            temperature=0.2,
            max_tokens=29695
        )
        raw_response = response.choices[0].message.content
        logger.debug(f"Raw response for skipped page {page_idx}: {raw_response}")

        cleaned_response = clean_response(raw_response)
        if not cleaned_response:
            logger.warning(f"Empty response for skipped page {page_idx}")
            return None, page_idx

        try:
            page_result = json.loads(cleaned_response)
            if not isinstance(page_result, dict) or str(page_idx) not in page_result:
                logger.warning(f"Invalid JSON for skipped page {page_idx}")
                return None, page_idx
            return page_result, None
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing failed for skipped page {page_idx}: {str(e)}")
            return None, page_idx
    except Exception as e:
        logger.error(f"Failed to process skipped page {page_idx}: {str(e)}")
        return None, page_idx

@app.post("/process_pdf")
async def process_pdf_endpoint(file: UploadFile = File(...), prompt: str = Form(...)):
    """Endpoint to process PDF and extract text based on prompt."""
    if not file:
        raise HTTPException(status_code=400, detail="Please upload a PDF file")
    if not prompt.strip():
        raise HTTPException(status_code=400, detail="Please provide a non-empty prompt")

    # Save uploaded file temporarily
    try:
        with open("temp.pdf", "wb") as f:
            f.write(await file.read())
        images = convert_from_path("temp.pdf")
    except Exception as e:
        logger.error(f"PDF conversion failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to convert PDF to images: {str(e)}")
    finally:
        if os.path.exists("temp.pdf"):
            os.remove("temp.pdf")

    num_pages = len(images)
    all_results = {}
    skipped_pages = []
    batch_size = 5
    model = "gemma3"
    client = get_openai_client(model)

    # Process batches concurrently
    batch_tasks = []
    for batch_start in range(0, num_pages, batch_size):
        batch_end = min(batch_start + batch_size, num_pages)
        batch_images = images[batch_start:batch_end]
        batch_messages = []

        for i, image in enumerate(batch_images, start=batch_start):
            try:
                image_bytes_io = BytesIO()
                image.save(image_bytes_io, format='JPEG', quality=85)
                image_bytes_io.seek(0)
                image_base64 = encode_image(image_bytes_io)
                batch_messages.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
                })
            except Exception as e:
                logger.error(f"Image processing failed for page {i}: {str(e)}")
                skipped_pages.append(i)
                continue

        if not batch_messages:
            logger.warning(f"Skipping batch {batch_start}-{batch_end-1}: No valid images")
            skipped_pages.extend(range(batch_start, batch_end))
            continue

        batch_messages.append({
            "type": "text",
            "text": (
                f"Extract plain text from these {batch_end - batch_start} PDF pages. "
                "Return the results as a valid JSON object where keys are page numbers "
                f"(starting from {batch_start}) and values are the extracted text for each page. "
                "Ensure the response is strictly JSON-formatted."
            )
        })

        batch_tasks.append(process_single_batch(client, model, batch_messages, batch_start, batch_end))

    batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)

    for batch_result in batch_results:
        if isinstance(batch_result, Exception):
            logger.error(f"Batch processing failed: {str(batch_result)}")
            continue
        batch_data, batch_skipped = batch_result
        if batch_data:
            all_results.update(batch_data)
        if batch_skipped:
            skipped_pages.extend(batch_skipped)

    # Retry skipped pages
    retry_tasks = []
    remaining_skipped = list(set(skipped_pages))
    for page_idx in remaining_skipped:
        retry_tasks.append(process_single_page(client, model, images[page_idx], page_idx))

    retry_results = await asyncio.gather(*retry_tasks, return_exceptions=True)
    successfully_processed = []

    for retry_result in retry_results:
        if isinstance(retry_result, Exception):
            logger.error(f"Retry processing failed: {str(retry_result)}")
            continue
        page_result, page_idx = retry_result
        if page_result:
            all_results.update(page_result)
            successfully_processed.append(page_idx)

    skipped_pages = [p for p in skipped_pages if p not in successfully_processed]

    if not all_results and skipped_pages:
        return JSONResponse(
            content={"error": "No valid text extracted from any pages", "skipped_pages": skipped_pages},
            status_code=400
        )

    # Process with the provided prompt
    dwani_prompt = (
        "You are Dwani, a helpful assistant. Answer questions considering India as base country "
        "and Karnataka as base state. Provide a concise response in one sentence maximum. "
        "If the answer contains numerical digits, convert the digits into words."
    )

    try:
        results_str = json.dumps(all_results)
    except Exception as e:
        logger.error(f"Failed to serialize all_results: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to serialize extracted text: {str(e)}")

    combined_prompt = f"{prompt} - Extracted text: {results_str}"

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