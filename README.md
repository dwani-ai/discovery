## Discovery - Document Analytics

- Setup
```bash
sudo apt-get install poppler-utils

python3.10 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

- Server
    - vllm serve HuggingFaceTB/SmolVLM-256M-Instruct --gpu-memory-utilization 0.4 --served-model-name gemma3 --host 0.0.0.0 --port 9000 --disable-log-requests

- Client

    - python ux/discovery_demo.py