## Discovery - VLM


sudo apt update
sudo apt upgrade -y

sudo apt-get install libssl-dev libcurl4-openssl-dev python3.12 python3.12-venv python3.12-dev -y

<!-- 
wget https://developer.download.nvidia.com/compute/cudnn/9.12.0/local_installers/cudnn-local-repo-ubuntu2204-9.12.0_1.0-1_arm64.deb

sudo dpkg -i cudnn-local-repo-ubuntu2204-9.12.0_1.0-1_arm64.deb

sudo cp /var/cudnn-local-repo-ubuntu2204-9.12.0/cudnn-*-keyring.gpg /usr/share/keyrings/


sudo apt-get update

sudo apt-get -y install cudnn

sudo apt-get -y install cudnn9-cuda-12

-->


python3.12 -m venv venv

source venv/bin/activate


pip install torch==2.7.1 torchaudio==2.7.1 torchvision --index-url https://download.pytorch.org/whl/cu128

<!-- 

pip install https://github.com/sachinsshetty/flashinfer-arm64/releases/download/v0.0.0.1/flashinfer_python-0.2.11.post1-py3-none-any.whl
-->


pip install https://github.com/dwani-ai/vllm-arm64/releases/download/v.0.0.4/vllm-0.10.1.dev0+g6d8d0a24c.d20250726-cp312-cp312-linux_aarch64.whl

vllm serve RedHatAI/gemma-3-27b-it-FP8-dynamic --served-model-name gemma3 --host 0.0.0.0 --port 9000 --gpu-memory-utilization 0.4 --tensor-parallel-size 1 --max-model-len 32768 --disable-log-requests  --dtype bfloat16


vllm serve RedHatAI/gemma-3-27b-it-FP8-dynamic --served-model-name gemma3 --host 0.0.0.0 --port 9100 --gpu-memory-utilization 0.4 --tensor-parallel-size 1 --max-model-len 32768 --disable-log-requests  --dtype bfloat16


---



for PC

- vllm serve HuggingFaceTB/SmolVLM-256M-Instruct --gpu-memory-utilization 0.4 --served-model-name gemma3 --host 0.0.0.0 --port 9000 --disable-log-requests

vllm serve RedHatAI/gemma-3-4b-it-FP8-dynamic --served-model-name gemma3 --host 0.0.0.0 --port 9000 --gpu-memory-utilization 0.4 --tensor-parallel-size 1 --max-model-len 4096 --disable-log-requests


vllm serve Qwen/Qwen2.5-VL-3B-Instruct-AWQ --gpu-memory-utilization 0.4

for Server
- vllm serve RedHatAI/gemma-3-27b-it-FP8-dynamic --served-model-name gemma3 --host 0.0.0.0 --port 9000 --gpu-memory-utilization 0.4 --tensor-parallel-size 1 --max-model-len 65536 --disable-log-requests

- vllm serve RedHatAI/gemma-3-4b-it-FP8-dynamic --served-model-name gemma3 --host 0.0.0.0 --port 9000 --gpu-memory-utilization 0.4 --tensor-parallel-size 1 --max-model-len 8192 --disable-log-requests

- For A100
    - vllm serve RedHatAI/gemma-3-12b-it-FP8-dynamic --served-model-name gemma3 --host 0.0.0.0 --port 9000 --gpu-memory-utilization 0.8 --tensor-parallel-size 1 --max-model-len 32768 --disable-log-requests



- with llama-cpp
    - docker compose -f llama-cpp-compose.yaml up -d