import gradio as gr
import requests
import logging
import os
import json
import concurrent.futures

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI server URLs
vlm_base_url = os.getenv('VLLM_IP', "0.0.0.0")
API_URL_PDF = f"http://{vlm_base_url}:18889/process_pdf"
API_URL_MESSAGE = f"http://{vlm_base_url}:18889/process_message"

# Global state to store PDF paths and extracted text
state = {
    "pdf_paths": [],
    "extracted_text": None
}

def extract_texts(pdf_paths):
    """Extracts text from multiple PDFs in parallel using a dummy prompt."""
    def extract_single(pdf_path):
        try:
            with open(pdf_path, "rb") as f:
                files = {"file": (os.path.basename(pdf_path), f, "application/pdf")}
                data = {"prompt": "Extract all text from this PDF."}
                response = requests.post(API_URL_PDF, files=files, data=data)
                response.raise_for_status()
                return response.json().get('extracted_text', '')
        except requests.RequestException as e:
            logger.error(f"Failed to extract text from {pdf_path}: {str(e)}")
            return ''

    if not pdf_paths:
        return ''

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(pdf_paths)) as executor:
        texts = list(executor.map(extract_single, pdf_paths))

    # Combine texts with separators for clarity
    combined = "\n\n---\n\n".join(
        [f"Text from {os.path.basename(path)}:\n{text}" for path, text in zip(pdf_paths, texts) if text]
    )
    return combined

def process_message(history, message, pdf_files=None):
    """Handles chat messages, reusing combined extracted text for follow-up questions."""
    if pdf_files is None:
        pdf_files = []

    current_paths = sorted(pdf_files)

    # Update state only if PDFs have changed
    if current_paths != state["pdf_paths"]:
        state["pdf_paths"] = current_paths
        state["extracted_text"] = None  # Reset to force re-extraction

    # Check if any PDFs are uploaded
    if not state["pdf_paths"]:
        return history + [{"role": "user", "content": message}, {"role": "assistant", "content": "⚠️ Please upload at least one PDF first!"}], ""

    try:
        # Extract texts if not yet done
        if state["extracted_text"] is None:
            state["extracted_text"] = extract_texts(state["pdf_paths"])

        # Use cached combined extracted text for the query
        data = {"prompt": message, "extracted_text": json.dumps(state["extracted_text"])}
        response = requests.post(API_URL_MESSAGE, data=data)
        response.raise_for_status()
        result = response.json()
        return history + [{"role": "user", "content": message}, {"role": "assistant", "content": result['response']}], ""
    except requests.RequestException as e:
        logger.error(f"API request failed: {str(e)}")
        return history + [{"role": "user", "content": message}, {"role": "assistant", "content": f"❌ Error: {str(e)}"}], ""

def clear_chat():
    """Clears the chat history."""
    return []

def new_chat():
    """Clears chat history and resets PDF state."""
    state["pdf_paths"] = []
    state["extracted_text"] = None
    return [], None

# Custom styling
css = """
.gradio-container { max-width: 1200px; margin: auto; }
#chatbot { height: calc(100vh - 200px); max-height: 800px; }
"""

with gr.Blocks(title="dwani.ai - Discovery", css=css, fill_width=True) as demo:
    gr.Markdown("# 📄 Document Chat - Query Your PDFs")

    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                [],
                elem_id="chatbot",
                label="Document Assistant",
                type="messages"
            )
            msg = gr.Textbox(
                placeholder="Ask something about the document...",
                label="Your Question"
            )
            pdf_input = gr.File(
                label="Attach PDFs (upload once per session)",
                file_types=[".pdf"],
                file_count="multiple"
            )
            with gr.Row():
                clear = gr.Button("Clear Chat")
                new_chat_button = gr.Button("New Chat")

        with gr.Column(scale=1):
            gr.Markdown("### Instructions")
            gr.Markdown(
                """
                1. Upload one or more PDF documents.  
                2. Ask questions about the documents in the chat box.  
                3. The assistant will respond based on the documents' content.  
                4. Use 'Clear Chat' to reset the conversation history.  
                5. Use 'New Chat' to start a new session (clears chat and PDFs).  
                """
            )

    # Event bindings
    msg.submit(process_message, inputs=[chatbot, msg, pdf_input], outputs=[chatbot, msg])
    clear.click(clear_chat, outputs=chatbot)
    new_chat_button.click(new_chat, outputs=[chatbot, pdf_input])

if __name__ == "__main__":
    try:
        demo.launch(server_name="0.0.0.0", server_port=8000)
    except Exception as e:
        logger.error(f"Failed to launch Gradio interface: {str(e)}")
        print(f"Failed to launch Gradio interface: {str(e)}")