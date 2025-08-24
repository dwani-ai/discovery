## Discovery - Document Analytics


- Visit : [https://app.dwani.ai](https://app.dwani.ai)


![Discovery](docs/images/document_extract.png "Discovery") 


```bash
sudo apt-get update
sudo apt-get install poppler-utils -y
```


- Client
    - python ux/ux.py
- Server
    - export VLLM_IP="your_vllm_ip"
    - uvicorn server.main:app --host 0.0.0.0 --port 18889
- VLLM Server setup - [server/vlm/README.md](server/vlm/README.md)


- Client Docker Run
    - docker run -p 80:8000 --env VLLM_IP=<server_ip> dwani/discovery_ux:latest

- Client Server Run 

---

### To Run locally  - English only 
```bash
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
- uvicorn server.local_main:app --host 0.0.0.0 --port 18889


For Client
- python ux/ux.py

```
---


---

- Setup
    ```bash
    python3.10 -m venv venv
    source venv/bin/activate
    ```
    - Client 
    ```bash
    pip install -r client-requirements.txt
    ```
<!-- 
    for x86
        - pip install https://github.com/dwani-ai/vllm-arm64/releases/download/v0.0.0.8/vllm-0.10.1.dev603+ga01e0018b.d20250813-cp312-cp312-linux_x86_64.whl
-->
   - Server
    -     sudo apt-get install poppler-utils -y

        - Follow Steps in [server/vlm/README.md](server/vlm/README.md)
    - Client

    ```bash
    python ux/discovery_demo.py
    ```


<!-- 
Client 
docker build -t dwani/discovery_ux:latest -f client.Dockerfile .
docker push dwani/discovery_ux:latest

docker run -p 80:8000 --env VLLM_IP=$VLLM_IP dwani/discovery_ux:latest

Server


docker build -t dwani/discovery_server:latest -f server.Dockerfile .
docker push dwani/discovery_server:latest

docker run -p 18888:18888 --env VLLM_IP=$VLLM_IP dwani/discovery_server:latest

-- arm64 - on GH200

sudo apt-get update
sudo apt-get install tesseract-ocr


-->