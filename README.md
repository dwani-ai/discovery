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

    for x86
        - pip install https://github.com/dwani-ai/vllm-arm64/releases/download/v0.0.0.8/vllm-0.10.1.dev603+ga01e0018b.d20250813-cp312-cp312-linux_x86_64.whl

- Run 

    - Server
    ```bash
    vllm serve HuggingFaceTB/SmolVLM-256M-Instruct --gpu-memory-utilization 0.4 --served-model-name gemma3 --host 0.0.0.0 --port 9000 --disable-log-requests
    ```
    - Client

    ```bash
    python ux/discovery_demo.py
    ```


<!-- 

docker build -t dwani/discovery_ux:latest -f client.Dockerfile .
docker push dwani/discovery_ux:latest

docker run -p 80:80 --env DWANI_API_KEY=<your_key> --env DWANI_API_BASE_URL=<your_url>  --env GPT_OSS_API_URL=<gpt_url> --env GEMMA_VLLM_IP=<gemma_ip> dwani/workshop:latest

-->