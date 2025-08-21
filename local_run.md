sudo apt-get update
sudo apt-get install tesseract-ocr
sudo apt-get install poppler-utils -y

- server/vlm/llama.md
    - Till line - 22


python3.10 -m venv venv
source venv/bin/activate

pip install -r server-requirements.txt

pip install -r client-requirements.txt

For server - 
- uvicorn server.local_main:app --host 0.0.0.0 --port 18888


For Client
- python ux/ux.py