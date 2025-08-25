
- To Run - gpt-oss - on PC
    - llama.cpp for gpt-oss models from OpenAI


```bash
sudo apt install libcurl4-openssl-dev

git clone --depth 1 https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
git checkout 0d8831543cdc368fb248bae6f1b4aa5516684edc

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


- Reference 
  - https://blogs.nvidia.com/blog/rtx-ai-garage-openai-oss/
  - HF model repo - https://huggingface.co/collections/ggml-org/gpt-oss-68923b60bee37414546c70bf



To Run - gemma
 ./build/bin/llama-server \
    --hf-repo unsloth/gemma-3-4b-it-GGUF \
    --hf-file gemma-3-4b-it-Q4_K_M.gguf

./build/bin/llama-server \
    --hf-repo unsloth/gemma-3-4b-it-GGUF \
    --hf-file mmproj-BF16.gguf


./build/bin/llama-server     --hf-repo unsloth/gemma-3-4b-it-GGUF     --hf-file gemma-3-4b-it-Q4_K_M.gguf --port 9500 --host 0.0.0.0

./build/bin/llama-server     --hf-repo ggml-org/gemma-3-12b-it-GGUF


