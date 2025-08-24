import gradio as gr
import requests
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI server URL
vlm_base_url = os.getenv('VLLM_IP', "0.0.0.0")
API_URL = f"http://{vlm_base_url}:18889/process_pdf"

# Store uploaded PDFs to allow multiple queries
uploaded_pdf = {"path": None}

def process_pdf_message(history, message, pdf_file=None):
    """Handles a chat message with optional PDF file, talks to backend API."""
    # If a new PDF is uploaded, store its path
    if pdf_file is not None:
        uploaded_pdf["path"] = pdf_file

    pdf_path = uploaded_pdf.get("path")
    if not pdf_path:
        return history + [{"role": "user", "content": message}, {"role": "assistant", "content": "⚠️ Please upload a PDF first!"}], ""

    try:
        with open(pdf_path, "rb") as f:
            files = {"file": (os.path.basename(pdf_path), f, "application/pdf")}
            data = {"prompt": message}
            response = requests.post(API_URL, files=files, data=data)
            response.raise_for_status()
            full_result = response.json()

            result = full_result['response']
            return history + [{"role": "user", "content": message}, {"role": "assistant", "content": str(result)}], ""
    except requests.RequestException as e:
        logger.error(f"API request failed: {str(e)}")
        return history + [{"role": "user", "content": message}, {"role": "assistant", "content": f"❌ Error: {str(e)}"}], ""

def clear_chat():
    """Clears the chat history."""
    return []

def new_chat():
    """Clears the chat history and resets the uploaded PDF."""
    uploaded_pdf["path"] = None
    return [], None  # Clears chatbot history and PDF input

# Custom styling
css = """
.gradio-container {
    max-width: 1200px;
    margin: auto;
}
#chatbot {
    height: calc(100vh - 200px);
    max-height: 800px;
}
"""

with gr.Blocks(title="dwani.ai - Discovery", css=css, fill_width=True) as demo:
    gr.Markdown("# 📄 Document Chat - Query Your PDFs")

    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                [],
                elem_id="chatbot",
                label="Document Assistant",
                type="messages"  # Explicitly set to 'messages' to avoid deprecation warning
            )

            msg = gr.Textbox(
                placeholder="Ask something about the document...",
                label="Your Question"
            )
            pdf_input = gr.File(
                label="Attach PDF (only needs to be uploaded once per session)",
                file_types=[".pdf"]
            )
            with gr.Row():
                clear = gr.Button("Clear Chat")
                new_chat_button = gr.Button("New Chat")  # New button for starting a new chat

        with gr.Column(scale=1):
            gr.Markdown("### Instructions")
            gr.Markdown(
                """
                1. Upload a PDF document.  
                2. Ask questions about the document in the chat box.  
                3. The assistant will return structured responses from the backend.  
                4. Use 'Clear Chat' to reset the conversation history.  
                5. Use 'New Chat' to start a new session (clears chat and PDF).  
                """
            )

    # Event binding
    msg.submit(
        process_pdf_message,
        inputs=[chatbot, msg, pdf_input],
        outputs=[chatbot, msg]  # Include msg to clear the textbox
    )
    clear.click(clear_chat, inputs=None, outputs=chatbot)
    new_chat_button.click(new_chat, inputs=None, outputs=[chatbot, pdf_input])  # Bind new chat button

if __name__ == "__main__":
    try:
        demo.launch(server_name="0.0.0.0", server_port=8000)
    except Exception as e:
        logger.error(f"Failed to launch Gradio interface: {str(e)}")
        print(f"Failed to launch Gradio interface: {str(e)}")