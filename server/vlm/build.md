
sudo apt update
sudo apt upgrade -y

sudo apt-get install libssl-dev libcurl4-openssl-dev python3.12 python3.12-venv python3.12-dev -y


wget https://github.com/Kitware/CMake/releases/download/v4.1.0/cmake-4.1.0.tar.gz
tar -zxvf cmake-4.1.0.tar.gz
cd cmake-4.1.0
./bootstrap
make -j4
sudo make install



python3.12 -m venv venv

source venv/bin/activate

sudo apt-get install build-essential libnuma-dev -y

pip install torch==2.7.1 torchaudio==2.7.1 torchvision --index-url https://download.pytorch.org/whl/cu128

git clone https://github.com/vllm-project/vllm.git

cd vllm


python use_existing_torch.py 

pip install --upgrade setuptools twine setuptools-scm


pip install -r requirements/cuda.txt

export MAX_JOBS=4
export NVCC_THREADS=2
export TORCH_CUDA_ARCH_LIST=""
export VLLM_TARGET_DEVICE=cuda

python setup.py bdist_wheel
pip install dist/*.whl

---


To Run

pip install xformers

vllm serve

vllm serve RedHatAI/gemma-3-27b-it-FP8-dynamic --served-model-name gemma3 --host 0.0.0.0 --port 9000 --gpu-memory-utilization 0.8 --tensor-parallel-size 1 --max-model-len 65536 --disable-log-requests


vllm serve google/gemma-3-4b-it --served-model-name gemma3 --host 0.0.0.0 --port 9000 --gpu-memory-utilization 0.8 --tensor-parallel-size 1 --max-model-len 65536     --dtype bfloat16 --disable-log-requests