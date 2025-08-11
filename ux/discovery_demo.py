import gradio as gr
import logging

from openai import OpenAI

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from openai import OpenAI

import base64
from io import BytesIO

from pdf2image import convert_from_path


import os
vlm_base_url = os.getenv('VLLM_IP')
def encode_image(image: BytesIO) -> str:
    """Encode image bytes to base64 string."""
    return base64.b64encode(image.read()).decode("utf-8")

# Dynamic LLM client based on model
def get_openai_client(model: str) -> OpenAI:
    """Initialize OpenAI client with model-specific base URL."""
    valid_models = ["gemma3", "moondream", "qwen2.5vl", "qwen3", "sarvam-m", "deepseek-r1"]
    if model not in valid_models:
        raise ValueError(f"Invalid model: {model}. Choose from: {', '.join(valid_models)}")
    
    model_ports = {
        "qwen3": "9100",
        "gemma3": "9000",
        "moondream": "7882",
        "qwen2.5vl": "7883",
    }
    base_url = f"http://{vlm_base_url}:{model_ports[model]}/v1"

    return OpenAI(api_key="http", base_url=base_url)


def ocr_page_with_rolm(img_base64: str, model: str) -> str:
    """Perform OCR on the provided base64 image using the specified model."""
    try:
        client = get_openai_client(model)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{img_base64}"}
                        },
                        {"type": "text", "text": "Return the plain text extracted from this image."}
                    ]
                }
            ],
            temperature=0.2,
            max_tokens=4096
        )
        return response.choices[0].message.content
    except Exception as e:
        raise Exception(status_code=500, detail=f"OCR processing failed: {str(e)}")
    


# --- PDF Processing Module ---
def process_pdf(pdf_file, page_number, prompt):
    if not pdf_file:
        return {"error": "Please upload a PDF file"}
    if not prompt.strip():
        return {"error": "Please provide a non-empty prompt"}
    try:
        page_number = int(page_number)
        if page_number < 1:
            raise ValueError("Page number must be at least 1")
    except (ValueError, TypeError):
        return {"error": "Page number must be a positive integer"}
    file_path = pdf_file.name if hasattr(pdf_file, 'name') else pdf_file


    # Change 'your_pdf_file.pdf' to the path of your PDF
    images = convert_from_path(file_path)

    for i, image in enumerate(images):
        #image.save(f'page_{i + 1}.jpg', 'JPEG')
        image_bytes_io = BytesIO()
        image.save(image_bytes_io, format='JPEG')
        #image_bytes = image_bytes_io.getvalue()

        image_bytes_io.seek(0)  # Reset cursor to start if needed
        img_base64 = encode_image(image_bytes_io)
        text = ocr_page_with_rolm(img_base64, model="gemma3")
        logger.debug(text)
        return {"extracted_text": text}


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
                    pdf_page = gr.Number(label="Page Number", value=1, minimum=1, precision=0)
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
                inputs=[pdf_input, pdf_page, pdf_prompt],
                outputs=pdf_output
            )

# Launch the interface
if __name__ == "__main__":
    try:
        demo.launch(server_name="0.0.0.0", server_port=80)
    except Exception as e:
        logger.error(f"Failed to launch Gradio interface: {str(e)}")
        print(f"Failed to launch Gradio interface: {str(e)}")