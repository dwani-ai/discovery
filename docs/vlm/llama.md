
- To Run - gpt-oss - on PC
    - llama.cpp for gpt-oss models from OpenAI


```bash
sudo apt install libcurl4-openssl-dev

git clone --depth 1 https://github.com/ggml-org/llama.cpp.git
cd llama.cpp


```
- With NVIDIA GPU and CUDA Toolkit
```bash     
    cmake -B build -DGGML_CUDA=ON
```    
- for CPU 
```bash
cmake -B build
```

-  build llama.cpp executable
```bash 
cmake --build build --config Release -j4
```

- For Laptop / PC -  gpt-oss-20b
```bash
./build/bin/llama-server -hf ggml-org/gpt-oss-20b-GGUF -c 0 -fa --jinja --reasoning-format none --port 9500
```

- For H100/H200 - gpt-oss-120b
```bash
    ./build/bin/llama-server -hf ggml-org/gpt-oss-120b-GGUF -c 0 -fa --jinja --reasoning-format none --port 9500
```

- Then, access http://localhost:8080


- For Vision + Text : Use Gemma

./build/bin/llama-server -hf ggml-org/gemma-3-4b-it-GGUF --host 0.0.0.0 --port 8080 --n-gpu-layers 99 --ctx-size 8192 --alias gemma3
