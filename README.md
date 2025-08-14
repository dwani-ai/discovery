## Discovery - Document Analytics


- Visit : [https://app.dwani.ai](https://app.dwani.ai)

- Client Docker Run
    - docker run -p 80:8000 --env VLLM_IP=<server_ip> dwani/discovery_ux:latest

- Server Run steps - [server/vlm/README.md](server/vlm/README.md)


---

![Discovery](docs/images/document_extract.png "Discovery") 


---

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
<!-- 
    for x86
        - pip install https://github.com/dwani-ai/vllm-arm64/releases/download/v0.0.0.8/vllm-0.10.1.dev603+ga01e0018b.d20250813-cp312-cp312-linux_x86_64.whl
-->
   - Server
        - Follow Steps in [server/vlm/README.md](server/vlm/README.md)
    - Client

    ```bash
    python ux/discovery_demo.py
    ```


<!-- 

docker build -t dwani/discovery_ux:latest -f client.Dockerfile .
docker push dwani/discovery_ux:latest

docker run -p 80:8000 --env VLLM_IP=<gemma_ip> dwani/discovery_ux:latest

-->