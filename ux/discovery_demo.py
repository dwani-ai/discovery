import gradio as gr
import requests
import dwani
import os
import tempfile
import logging
from PIL import Image
import urllib.parse
import json
import time
import uuid
from openai import OpenAI

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure dwani API settings
dwani.api_key = os.getenv("DWANI_API_KEY")
dwani.api_base = os.getenv("DWANI_API_BASE_URL")


# Validate API configuration
if not dwani.api_key or not dwani.api_base:
    logger.error("API key or base URL not set. Please set DWANI_API_KEY and DWANI_API_BASE_URL environment variables.")
    raise ValueError("Please set DWANI_API_KEY and DWANI_API_BASE_URL environment variables.")


from pdf2image import convert_from_path

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

    # Save each page as an image
    for i, image in enumerate(images):
        image.save(f'page_{i + 1}.jpg', 'JPEG')


    

    '''
    try:
        result = dwani.Documents.query_all(
            file_path, model="gemma3", prompt=prompt
        )
        return {
            "Original Text": result.get("original_text", "N/A"),
            "Response": result.get("query_answer", "N/A"),
            "Translated Response": result.get("translated_query_answer", "N/A")
        }
    except Exception as e:
        return {"error": f"PDF API error: {str(e)}"}
    '''


# --- Chatbot from File 2 ---
# Initialize OpenAI client for Chatbot
gemma_base_url = os.getenv('GEMMA_VLLM_IP', 'http://localhost:9000/v1')
api_key = os.getenv('OPENAI_API_KEY', 'your-api-key')
client = OpenAI(api_key=api_key, base_url=gemma_base_url)

# Configuration for Chatbot
DEFAULT_SYS_PROMPT = "You are a helpful and harmless assistant. Respond concisely but meaningfully to short inputs, and provide detailed answers when appropriate."
DEFAULT_MODEL = "gemma3"
MODEL_OPTIONS = [{"label": "Gemma3", "value": "gemma3"}]
MODEL_OPTIONS_MAP = {model["value"]: model for model in MODEL_OPTIONS}
DEFAULT_SETTINGS = {"model": DEFAULT_MODEL, "sys_prompt": DEFAULT_SYS_PROMPT}

def format_history(history, sys_prompt):
    messages = [{"role": "system", "content": sys_prompt}] + history
    return messages

class Gradio_Events:
    @staticmethod
    def submit(state_value, user_input, model_value, sys_prompt_value):
        conversation_id = state_value["conversation_id"]
        history = state_value["conversation_contexts"][conversation_id]["history"]
        settings = {"model": model_value, "sys_prompt": sys_prompt_value}
        state_value["conversation_contexts"][conversation_id]["settings"] = settings

        history.append({"role": "user", "content": user_input})
        messages = format_history(history, sys_prompt_value)

        try:
            response = client.chat.completions.create(
                model=model_value,
                messages=messages,
                stream=False
            )
            start_time = time.time()
            answer_content = response.choices[0].message.content
            history.append({"role": "assistant", "content": f"{answer_content}\n\n*Generated in {time.time() - start_time:.2f}s*"})
        except Exception as e:
            history.append({"role": "assistant", "content": f"Error: {str(e)}"})

        return (
            gr.update(value=history),
            gr.update(value=state_value),
            gr.update(interactive=True),
            gr.update(interactive=True),
            gr.update(interactive=True),
            gr.update(choices=[(c["label"], c["key"]) for c in state_value["conversations"]], 
                     visible=bool(state_value["conversations"]), 
                     value=state_value["conversation_id"])
        )

    @staticmethod
    def add_message(user_input, model_value, sys_prompt_value, state_value):
        if not user_input.strip():
            return (
                gr.skip(),
                state_value,
                user_input,
                gr.skip(),
                gr.skip(),
                gr.update(choices=[(c["label"], c["key"]) for c in state_value["conversations"]], 
                         visible=bool(state_value["conversations"]), 
                         value=state_value["conversation_id"])
            )

        if not state_value["conversation_id"]:
            random_id = str(uuid.uuid4())
            state_value["conversation_id"] = random_id
            state_value["conversation_contexts"][random_id] = {
                "history": [],
                "settings": {"model": model_value, "sys_prompt": sys_prompt_value}
            }
            state_value["conversations"].append({
                "label": user_input[:30] + "..." if len(user_input) > 30 else user_input,
                "key": random_id
            })

        return Gradio_Events.submit(state_value, user_input, model_value, sys_prompt_value)

    @staticmethod
    def new_chat(state_value):
        state_value["conversation_id"] = ""
        return (
            gr.update(value=[]),
            gr.update(value=state_value),
            gr.update(value=DEFAULT_SETTINGS["model"]),
            gr.update(value=DEFAULT_SETTINGS["sys_prompt"]),
            gr.update(choices=[], visible=False)
        )

    @staticmethod
    def select_conversation(state_value, evt: gr.EventData):
        conversation_id = evt._data
        if conversation_id not in state_value["conversation_contexts"]:
            return gr.skip(), gr.skip(), gr.skip(), gr.skip()
        state_value["conversation_id"] = conversation_id
        history = state_value["conversation_contexts"][conversation_id]["history"]
        settings = state_value["conversation_contexts"][conversation_id]["settings"]
        return (
            gr.update(value=history),
            gr.update(value=state_value),
            gr.update(value=settings["model"]),
            gr.update(value=settings["sys_prompt"])
        )

    @staticmethod
    def delete_conversation(state_value, evt: gr.EventData):
        conversation_id = evt._data
        if conversation_id in state_value["conversation_contexts"]:
            del state_value["conversation_contexts"][conversation_id]
            state_value["conversations"] = [c for c in state_value["conversations"] if c["key"] != conversation_id]
            if state_value["conversation_id"] == conversation_id:
                state_value["conversation_id"] = ""
                return (
                    gr.update(value=[]),
                    gr.update(value=state_value),
                    gr.update(choices=[], visible=False)
                )
        return gr.skip(), gr.update(value=state_value), gr.update(choices=[(c["label"], c["key"]) for c in state_value["conversations"]], 
                                                                visible=bool(state_value["conversations"]), 
                                                                value=state_value["conversation_id"])

    @staticmethod
    def clear_conversation(state_value):
        if state_value["conversation_id"]:
            state_value["conversation_contexts"][state_value["conversation_id"]]["history"] = []
            return gr.update(value=[]), gr.update(value=state_value)
        return gr.skip(), gr.skip()

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
        demo.launch(server_name="0.0.0.0", server_port=8000)
    except Exception as e:
        logger.error(f"Failed to launch Gradio interface: {str(e)}")
        print(f"Failed to launch Gradio interface: {str(e)}")