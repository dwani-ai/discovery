import gradio as gr
import logging
import json
from openai import AsyncOpenAI  # Use AsyncOpenAI for async support
import base64
from io import BytesIO
from pdf2image import convert_from_path
import os
import asyncio
import aiohttp
import re

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

vlm_base_url = os.getenv('VLLM_IP', "0.0.0.0")

def encode_image(image: BytesIO) -> str:
    """Encode image bytes to base64 string."""
    return base64.b64encode(image.read()).decode("utf-8")

# Dynamic AsyncOpenAI client based on model
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

def clean_response(raw_response):
    """Clean markdown code blocks or other non-JSON content from the response."""
    if not raw_response:
        return None
    # Remove markdown code blocks (e.g., ```json ... ``` or ``` ... ```)
    cleaned = re.sub(r'```(?:json)?\s*([\s\S]*?)\s*```', r'\1', raw_response)
    # Remove leading/trailing whitespace
    cleaned = cleaned.strip()
    return cleaned

async def process_batch(client, batch_messages, batch_start, batch_end, model):
    """Process a single batch of images asynchronously."""
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": batch_messages}],
            temperature=0.2,
            max_tokens=25000
        )
        raw_response = response.choices[0].message.content
        logger.debug(f"Raw response for batch {batch_start}-{batch_end-1}: {raw_response}")

        # Clean the response
        cleaned_response = clean_response(raw_response)
        if not cleaned_response:
            logger.warning(f"Empty response for batch {batch_start}-{batch_end-1}")
            return None, list(range(batch_start, batch_end))

        # Parse JSON
        try:
            batch_results = json.loads(cleaned_response)
            if not isinstance(batch_results, dict):
                logger.warning(f"Response is not a JSON object for batch {batch_start}-{batch_end-1}")
                return None, list(range(batch_start, batch_end))
            return batch_results, []
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing failed for batch {batch_start}-{batch_end-1}: {str(e)}")
            logger.debug(f"Cleaned response: {cleaned_response}")
            return None, list(range(batch_start, batch_end))
    except Exception as e:
        logger.error(f"API request failed for batch {batch_start}-{batch_end-1}: {str(e)}")
        return None, list(range(batch_start, batch_end))

async def process_pdf(pdf_file, prompt):
    if not pdf_file:
        return {"error": "Please upload a PDF file"}
    if not prompt.strip():
        return {"error": "Please provide a non-empty prompt"}

    file_path = pdf_file.name if hasattr(pdf_file, 'name') else pdf_file

    # Convert PDF to images with error handling
    try:
        images = convert_from_path(file_path)
    except Exception as e:
        logger.error(f"PDF conversion failed: {str(e)}")
        return {"error": f"Failed to convert PDF to images: {str(e)}"}

    num_pages = len(images)
    all_results = {}
    skipped_pages = []
    batch_size = 5  # Keep reduced batch size
    model = "gemma3"
    client = get_openai_client(model)

    # Prepare batches
    batches = []
    for batch_start in range(0, num_pages, batch_size):
        batch_end = min(batch_start + batch_size, num_pages)
        batch_images = images[batch_start:batch_end]
        batch_messages = []

        # Convert images to base64 for current batch
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

        # Add text instruction for current batch
        batch_page_count = batch_end - batch_start
        batch_messages.append({
            "type": "text",
            "text": (
                f"Extract plain text from these {batch_page_count} PDF pages. "
                "Return the results as a valid JSON object where keys are page numbers "
                f"(starting from {batch_start}) and values are the extracted text for each page. "
                "Ensure the response is strictly JSON-formatted and does not include markdown code blocks "
                "or any text outside the JSON object."
            )
        })

        batches.append((batch_messages, batch_start, batch_end))

    # Process batches concurrently
    tasks = [process_batch(client, messages, start, end, model) for messages, start, end in batches]
    batch_results = await asyncio.gather(*tasks, return_exceptions=True)

    # Combine results
    for batch_result, batch_skipped in batch_results:
        if batch_result:
            all_results.update(batch_result)
        if batch_skipped:
            skipped_pages.extend(batch_skipped)

    if not all_results and skipped_pages:
        return {"error": "No valid text extracted from any pages", "skipped_pages": skipped_pages}

    # Process the combined results with the provided prompt
    dwani_prompt = (
        "You are Dwani, a helpful assistant. Answer questions considering India as base country "
        "and Karnataka as base state. Provide a concise response in one sentence maximum. "
        "If the answer contains numerical digits, convert the digits into words."
    )

    # Convert all_results to string for the second API call
    try:
        results_str = json.dumps(all_results)
    except Exception as e:
        logger.error(f"Failed to serialize all_results: {str(e)}")
        return {"error": f"Failed to serialize extracted text: {str(e)}", "skipped_pages": skipped_pages}

    combined_prompt = f"{prompt} - Extracted text: {results_str}"

    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": [{"type": "text", "text": dwani_prompt}]
                },
                {"role": "user", "content": [{"type": "text", "text": combined_prompt}]}
            ],
            temperature=0.3,
            max_tokens=25000
        )
        generated_response = response.choices[0].message.content
        return {
            "response": generated_response,
            "extracted_text": all_results,
            "skipped_pages": skipped_pages
        }
    except Exception as e:
        logger.error(f"Final API request failed: {str(e)}")
        return {
            "error": f"Final API request failed: {str(e)}",
            "extracted_text": all_results,
            "skipped_pages": skipped_pages
        }

# --- Gradio Interface ---
css = """
.gradio-container {
    max-width: 1200px;
    margin: auto;
}
#chatbot {
    height: calc(100vh - 200px);
    max-height: 800px;
}
#conversations {
    max-height: 600px;
    overflow-y: auto;
}
"""

with gr.Blocks(title="dwani.ai - Discovery", css=css, fill_width=True) as demo:
    gr.Markdown("# Document Analytics")

    with gr.Tabs():
        # PDF Query Tab
        with gr.Tab("PDF Query"):
            gr.Markdown("Query PDF files with a custom prompt")
            with gr.Row():
                with gr.Column():
                    pdf_input = gr.File(label="Upload PDF", file_types=[".pdf"])
                    pdf_prompt = gr.Textbox(
                        label="Custom Prompt",
                        placeholder="e.g., List the key points",
                        value="List the key points",
                        lines=3
                    )
                    pdf_submit = gr.Button("Process")
                with gr.Column():
                    pdf_output = gr.JSON(label="PDF Response")
            pdf_submit.click(
                fn=process_pdf,
                inputs=[pdf_input, pdf_prompt],
                outputs=pdf_output
            )

# Launch the interface
if __name__ == "__main__":
    try:
        demo.launch(server_name="0.0.0.0", server_port=8000)
    except Exception as e:
        logger.error(f"Failed to launch Gradio interface: {str(e)}")
        print(f"Failed to launch Gradio interface: {str(e)}")