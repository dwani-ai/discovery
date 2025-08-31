import gradio as gr
import requests
import logging
import os
import json
import concurrent.futures
from typing import List, Dict, Tuple, Optional
from time import time
import argparse
import hashlib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Configuration
VLLM_IP = os.getenv('VLLM_IP', '0.0.0.0')
API_URL_PDF = f"http://{VLLM_IP}:18889/process_pdf"
API_URL_MESSAGE = f"http://{VLLM_IP}:18889/process_message"
MAX_FILE_SIZE_MB = 10  # Max PDF size in MB
MAX_CONCURRENT_PDFS = 5  # Max PDFs to process concurrently
CACHE_TTL = 3600  # Cache extracted text for 1 hour
SESSION_CACHE = {}  # In-memory session cache: {session_id: {pdf_hash: extracted_text, timestamp}}

def validate_config() -> None:
    """Validate environment configuration at startup."""
    if VLLM_IP == '0.0.0.0':
        logger.warning("VLLM_IP not set, using default '0.0.0.0'. This may cause issues in production.")
    try:
        response = requests.get(f"http://{VLLM_IP}:18889/health", timeout=30)
        response.raise_for_status()
    except requests.RequestException as e:
        logger.error(f"Failed to connect to API server at {VLLM_IP}: {str(e)}")
        raise

def validate_pdf(file_path: str) -> bool:
    """Validate PDF file for type and size."""
    if not file_path.lower().endswith('.pdf'):
        logger.warning(f"Invalid file type: {file_path}")
        return False
    try:
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        if file_size_mb > MAX_FILE_SIZE_MB:
            logger.warning(f"File {file_path} exceeds size limit of {MAX_FILE_SIZE_MB}MB")
            return False
        return True
    except OSError as e:
        logger.error(f"Error accessing file {file_path}: {str(e)}")
        return False

def extract_single_pdf(pdf_path: str) -> str:
    """Extract text from a single PDF."""
    if not validate_pdf(pdf_path):
        return ''
    try:
        with open(pdf_path, "rb") as f:
            files = {"file": (os.path.basename(pdf_path), f, "application/pdf")}
            data = {"prompt": "Extract all text from this PDF."}
            response = requests.post(API_URL_PDF, files=files, data=data, timeout=90)
            response.raise_for_status()
            return response.json().get('extracted_text', '')
    except requests.RequestException as e:
        logger.error(f"Failed to extract text from {pdf_path}: {str(e)}")
        return ''
    except Exception as e:
        logger.error(f"Unexpected error extracting text from {pdf_path}: {str(e)}")
        return ''

def extract_texts(pdf_paths: List[str], session_id: str) -> str:
    """Extract text from multiple PDFs in parallel and cache results."""
    valid_paths = [p for p in pdf_paths if validate_pdf(p)]
    if not valid_paths:
        return "No valid PDFs provided."

    # Generate cache key based on PDF paths
    pdf_hash = hashlib.md5("".join(sorted(valid_paths)).encode()).hexdigest()
    session_data = SESSION_CACHE.get(session_id, {})
    cached = session_data.get('cache', {}).get(pdf_hash)

    # Check cache and TTL
    if cached and (time() - session_data.get('timestamp', 0)) < CACHE_TTL:
        logger.info(f"Returning cached text for session {session_id}")
        return cached['text']

    # Extract texts in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(valid_paths), MAX_CONCURRENT_PDFS)) as executor:
        texts = list(executor.map(extract_single_pdf, valid_paths))

    # Combine texts
    combined = "\n\n---\n\n".join(
        [f"Text from {os.path.basename(path)}:\n{text}" for path, text in zip(valid_paths, texts) if text]
    )

    # Update cache
    SESSION_CACHE[session_id] = {
        'cache': {pdf_hash: {'text': combined, 'timestamp': time()}},
        'timestamp': time(),
        'pdf_paths': valid_paths
    }
    return combined

def process_message(history: List[Dict], message: str, pdf_files: Optional[List[str]], session_id: str) -> Tuple[List[Dict], str]:
    """Handle chat messages, reusing cached extracted text."""
    pdf_files = pdf_files or []
    current_paths = sorted(pdf_files)

    # Validate input
    if not message.strip():
        return history + [{"role": "user", "content": message}, {"role": "assistant", "content": "⚠️ Please enter a valid question!"}], ""

    if not current_paths:
        return history + [{"role": "user", "content": message}, {"role": "assistant", "content": "⚠️ Please upload at least one PDF first!"}], ""

    try:
        # Extract or retrieve cached text
        extracted_text = extract_texts(current_paths, session_id)
        if not extracted_text:
            return history + [{"role": "user", "content": message}, {"role": "assistant", "content": "⚠️ No text could be extracted from the provided PDFs!"}], ""

        # Send query to API
        data = {"prompt": message, "extracted_text": json.dumps(extracted_text)}
        response = requests.post(API_URL_MESSAGE, data=data, timeout=90)
        response.raise_for_status()
        result = response.json()

        return history + [{"role": "user", "content": message}, {"role": "assistant", "content": result['response']}], ""
    except requests.RequestException as e:
        logger.error(f"API request failed for session {session_id}: {str(e)}")
        return history + [{"role": "user", "content": message}, {"role": "assistant", "content": f"❌ Error: Failed to process your request. Please try again later."}], ""
    except Exception as e:
        logger.error(f"Unexpected error for session {session_id}: {str(e)}")
        return history + [{"role": "user", "content": message}, {"role": "assistant", "content": f"❌ Error: {str(e)}"}], ""

def clear_chat(session_id: str) -> List:
    """Clear the chat history for a session."""
    return []

def new_chat(session_id: str) -> Tuple[List, None]:
    """Clear chat history and reset PDF state for a session."""
    SESSION_CACHE.pop(session_id, None)
    return [], None

# Custom styling
css = """
.gradio-container { max-width: 1200px; margin: auto; }
#chatbot { height: calc(100vh - 200px); max-height: 800px; }
#message { resize: none; }
"""

def create_gradio_app() -> gr.Blocks:
    """Create and configure the Gradio application."""
    with gr.Blocks(title="dwani.ai - Discovery", css=css, fill_width=True) as demo:
        gr.Markdown("# 📄 Document Chat - Query Your PDFs")

        # Generate a unique session ID
        session_id = gr.State(value=f"session_{int(time())}")

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
                    label="Your Question",
                    elem_id="message"
                )
                pdf_input = gr.File(
                    label=f"Attach PDFs (max {MAX_FILE_SIZE_MB}MB each, upload once per session)",
                    file_types=[".pdf"],
                    file_count="multiple"
                )
                with gr.Row():
                    clear = gr.Button("Clear Chat")
                    new_chat_button = gr.Button("New Chat")

            with gr.Column(scale=1):
                gr.Markdown("### Instructions")
                gr.Markdown(
                    f"""
                    1. Upload one or more PDF documents (max {MAX_FILE_SIZE_MB}MB each).  
                    2. Ask questions about the documents in the chat box.  
                    3. The assistant will respond based on the documents' content.  
                    4. Use 'Clear Chat' to reset the conversation history.  
                    5. Use 'New Chat' to start a new session (clears chat and PDFs).  
                    """
                )

        # Event bindings
        msg.submit(
            process_message,
            inputs=[chatbot, msg, pdf_input, session_id],
            outputs=[chatbot, msg]
        )
        clear.click(
            clear_chat,
            inputs=[session_id],
            outputs=[chatbot]
        )
        new_chat_button.click(
            new_chat,
            inputs=[session_id],
            outputs=[chatbot, pdf_input]
        )

    return demo

def main():
    """Main function to run the application."""
    parser = argparse.ArgumentParser(description="Document Chat Application")
    parser.add_argument("--host", default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    args = parser.parse_args()

    try:
        validate_config()
        demo = create_gradio_app()
        demo.launch(server_name=args.host, server_port=args.port, show_error=True)
    except Exception as e:
        logger.error(f"Failed to launch Gradio interface: {str(e)}")
        print(f"Failed to launch Gradio interface: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main()