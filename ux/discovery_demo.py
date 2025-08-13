import gradio as gr
import logging
import json
from openai import OpenAI
import base64
from io import BytesIO
from pdf2image import convert_from_path
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

vlm_base_url = os.getenv('VLLM_IP', "0.0.0.0")
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

# --- PDF Processing Module ---
def process_pdf(pdf_file, prompt):
    if not pdf_file:
        return {"error": "Please upload a PDF file"}
    if not prompt.strip():
        return {"error": "Please provide a non-empty prompt"}

    file_path = pdf_file.name if hasattr(pdf_file, 'name') else pdf_file

    # Change 'your_pdf_file.pdf' to the path of your PDF
    images = convert_from_path(file_path)
    num_pages = len(images)
    messages = []

    for i, image in enumerate(images):
        image_bytes_io = BytesIO()
        image.save(image_bytes_io, format='JPEG')

        image_bytes_io.seek(0)  # Reset cursor to start if needed
        image_base64 = encode_image(image_bytes_io)
        messages.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_base64}"}
        })
        
    messages.append({
            "type": "text",
            "text": (
                f"Extract plain text from these {num_pages} PDF pages. "
                "Return the results as a valid JSON object where keys are page numbers (starting from 0) "
                "and values are the extracted text for each page. Ensure the response is strictly JSON-formatted "
                "and does not include markdown code blocks or any text outside the JSON object."
            )
        })

    model = "gemma3"    
    client = get_openai_client(model)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": messages}],
        temperature=0.2,
        max_tokens=8000
    )
    num_pages = len(images)

    raw_response = response.choices[0].message.content

    print(raw_response)
    # Clean markdown code blocks
    
    cleaned_response = prompt + " - " +  raw_response



    dwani_prompt = f"You are a helpful assistant"

    client = get_openai_client(model)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
            #    "content": [{"type": "text", "text": f"You are Dwani, a helpful assistant. Answer questions considering India as base country and Karnataka as base state. Provide a concise response in one sentence maximum. If the answer contains numerical digits, convert the digits into words. If user asks the time, then return answer as {current_time}"}]
                "content": [{"type": "text", "text": dwani_prompt }]
            
            },
            {"role": "user", "content": [{"type": "text", "text": cleaned_response}]}
        ],
        temperature=0.3,
        max_tokens=8000
    )
    generated_response = response.choices[0].message.content
    logger.debug(f"Generated response: {generated_response}")

    '''
    if raw_response.startswith("```json") and raw_response.endswith("```"):
        cleaned_response = raw_response[7:-3].strip()
    elif raw_response.startswith("```") and raw_response.endswith("```"):
        cleaned_response = raw_response[3:-3].strip()
    
 
    page_contents = json.loads(cleaned_response)

    logger.debug(page_contents)
    '''
    return {"extracted_text": generated_response}


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
                inputs=[pdf_input,pdf_prompt],
                outputs=pdf_output
            )

# Launch the interface
if __name__ == "__main__":
    try:
        demo.launch(server_name="0.0.0.0", server_port=8000)
    except Exception as e:
        logger.error(f"Failed to launch Gradio interface: {str(e)}")
        print(f"Failed to launch Gradio interface: {str(e)}")