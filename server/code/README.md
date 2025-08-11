## Discovery - Code Gen

- sudo apt install libcurl4-openssl-dev

- git clone https://github.com/ggml-org/llama.cpp.git
- cd llama.cpp
- git checkout 0d8831543cdc368fb248bae6f1b4aa5516684edc

- With NVIDIA GPU and CUDA Toolkit
    - cmake -B build -DGGML_CUDA=ON
- for CPU 
    - cmake -B build

- cmake --build build --config Release -j4

- For Laptop / PC -  gpt-oss-20b

    - ./build/bin/llama-server -hf ggml-org/gpt-oss-20b-GGUF -c 0 -fa --jinja --reasoning-format none --port 9500

