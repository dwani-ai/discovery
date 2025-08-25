torch 2.8

curl -LsSf https://astral.sh/uv/install.sh | INSTALLER_DOWNLOAD_URL=https://wheelnext.astral.sh sh
uv venv venv
source venv/bin/activate

uv pip install torch torchvision

pip install https://github.com/dwani-ai/vllm-arm64/releases/download/v0.0.7/vllm-0.1.dev1+ge5d3d63c4.d20250812.cpu-cp312-cp312-linux_aarch64.whl
