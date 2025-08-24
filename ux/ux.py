import gradio as gr
import requests
import logging
import os
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI server URLs
vlm_base_url = os.getenv('VLLM_IP', "0.0.0.0")
API_URL_PDF = f"http://{vlm_base_url}:18889/process_pdf"
API_URL_MESSAGE = f"http://{vlm_base_url}:18889/process_message"

# Global state to store PDF path and extracted text
state = {
    "pdf_path": None,
    "extracted_text": None
}

def process_pdf(pdf_file, message):
    """Processes a PDF file and extracts text via API."""
    try:
        with open(pdf_file, "rb") as f:
            files = {"file": (os.path.basename(pdf_file), f, "application/pdf")}
            data = {"prompt": message}
            response = requests.post(API_URL_PDF, files=files, data=data)
            response.raise_for_status()
            result = response.json()
            state["extracted_text"] = result.get('extracted_text', '')  # Store extracted text
            return result['response']
    except requests.RequestException as e:
        logger.error(f"PDF processing failed: {str(e)}")
        return f"❌ Error processing PDF: {str(e)}"

def process_message(history, message, pdf_file=None):
    """Handles chat messages, reusing extracted text for follow-up questions."""
    # Update PDF path only if a new file is uploaded and different
    if pdf_file is not None and pdf_file != state["pdf_path"]:
        state["pdf_path"] = pdf_file
        state["extracted_text"] = None  # Reset extracted text only for new PDF

    # Check if PDF is uploaded
    if not state["pdf_path"]:
        return history + [{"role": "user", "content": message}, {"role": "assistant", "content": "⚠️ Please upload a PDF first!"}], ""

    try:
        # Process PDF if text is not yet extracted
        if state["extracted_text"] is None:
            response = process_pdf(state["pdf_path"], message)
        else:
            # Use cached extracted text for follow-up questions
            data = {"prompt": message, "extracted_text": json.dumps(state["extracted_text"])}
            response = requests.post(API_URL_MESSAGE, data=data)
            response.raise_for_status()
            response = response.json()['response']

        return history + [{"role": "user", "content": message}, {"role": "assistant", "content": str(response)}], ""
    except requests.RequestException as e:
        logger.error(f"API request failed: {str(e)}")
        return history + [{"role": "user", "content": message}, {"role": "assistant", "content": f"❌ Error: {str(e)}"}], ""

def clear_chat():
    """Clears the chat history."""
    return []

def new_chat():
    """Clears chat history and resets PDF state."""
    state["pdf_path"] = None
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
                label="Attach PDF (upload once per session)",
                file_types=[".pdf"]
            )
            with gr.Row():
                clear = gr.Button("Clear Chat")
                new_chat_button = gr.Button("New Chat")

        with gr.Column(scale=1):
            gr.Markdown("### Instructions")
            gr.Markdown(
                """
                1. Upload a PDF document.  
                2. Ask questions about the document in the chat box.  
                3. The assistant will respond based on the document's content.  
                4. Use 'Clear Chat' to reset the conversation history.  
                5. Use 'New Chat' to start a new session (clears chat and PDF).  
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