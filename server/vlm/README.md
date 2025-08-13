## Discovery - VLM

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