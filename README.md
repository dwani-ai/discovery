## Discovery - Document Analytics

- Setup
    ```bash
    python3.10 -m venv venv
    source venv/bin/activate
    ```
    - Client 
    ```bash
    sudo apt-get install poppler-utils

    pip install -r client-requirements.txt
    ```

    - Server
    ```
    pip install -r server-requirements.txt
    ```

- Run 

    - Server
    ```bash
    vllm serve HuggingFaceTB/SmolVLM-256M-Instruct --gpu-memory-utilization 0.4 --served-model-name gemma3 --host 0.0.0.0 --port 9000 --disable-log-requests
    ```
    - Client

    ```bash
    python ux/discovery_demo.py
    ```