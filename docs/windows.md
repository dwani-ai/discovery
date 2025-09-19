To - run discovery server on windows


- run Powershell as Administrator and run below command

Set-ExecutionPolicy RemoteSigned

python.exe -m venv venv

.\venv\Scripts\Activate.ps1

pip.exe install -r server-requirements.txt


---
As - powershell admin
choco install poppler


Add to PATH
C:\ProgramData\chocolatey\lib\poppler\tools


---

Install Docker desktop

as powershell admin
wsl --update

 docker build -t dwani/discovery_server:latest -f server.Dockerfile .
//docker run -p 18889:18889 --env DWANI_API_BASE_URL=0.0.0.0 dwani/discovery_server:latest

docker-compose.exe up -d

---

Install cuda toolkit for windows


---
llama-cpp
In powershell

winget install llama.cpp


llama-server -hf ggml-org/gemma-3-4b-it-GGUF --host 0.0.0.0 --port 9000 --n-gpu-layers 99 --ctx-size 8192 --alias gemma3 


llama-server -hf ggml-org/gemma-3-4b-it-GGUF --host 0.0.0.0 --port 9000 --n-gpu-layers 99 --ctx-size 1008 --alias gemma3 


Q4 : 2.37
./build/bin/llama-server -hf google/gemma-3-4b-it-qat-q4_0-gguf --host 0.0.0.0 --port 9000 --n-gpu-layers 99 --ctx-size 1008 --alias gemma3


Q4 : 2.37
./build/bin/llama-server -hf ggml-org/gemma-3-4b-it-qat-GGUF --host 0.0.0.0 --port 9000 --n-gpu-layers 99 --ctx-size 1008 --alias gemma3


Q2 :   1.73 GB

./build/bin/llama-server -hf unsloth/gemma-3-4b-it-GGUF:Q2_k --host 0.0.0.0 --port 8000 --n-gpu-layers 99 --ctx-size 1008 --alias gemma3
