Discovery - Agentic Intelligence Platform 

Try Demo :  [https://app.dwani.ai](https://app.dwani.ai)

for server
    - [README.md](backend/README.md)

for client
    - [README.md](frontend/README.md)

for VLM
    - [README.md](vlm/README.md)



To run locally
    - update the environment with your local vllm/llama-cpp IP/port 
         
```
    export DWANI_API_BASE_URL=vllm/llama.cpp/IP
```  
    
- Run Docker 

```
    docker compose -f docker-compose.yml up -d
```