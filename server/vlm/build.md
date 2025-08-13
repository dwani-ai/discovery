
sudo apt update
sudo apt upgrade -y

sudo apt install python3.12 python3.12-venv python3.12-dev -y



wget https://github.com/Kitware/CMake/releases/download/v4.1.0/cmake-4.1.0.tar.gz
tar -zxvf cmake-4.1.0.tar.gz
cd cmake-4.1.0
./bootstrap
make
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

