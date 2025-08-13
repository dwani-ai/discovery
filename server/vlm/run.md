

sudo apt update
sudo apt upgrade -y

sudo apt-get install libssl-dev libcurl4-openssl-dev python3.12 python3.12-venv python3.12-dev -y


python3.12 -m venv venv

source venv/bin/activate

pip install https://github.com/dwani-ai/vllm-arm64/releases/download/v.0.0.4/vllm-0.10.1.dev0+g6d8d0a24c.d20250726-cp312-cp312-linux_aarch64.whl