## Discovery - VLM

python3.10 -m venv venv
source venv/bin/activate

pip install vllm


vllm serve RedHatAI/gemma-3-4b-it-FP8-dynamic --served-model-name gemma3 --host 0.0.0.0 --port 9000 --gpu-memory-utilization 0.4 --tensor-parallel-size 1 --max-model-len 65536    --disable-log-requests

Gemma3-4B-it-fp8 with vllm
gpt-oss-20b with llama.cpp
