

sudo apt-get update
sudo apt upgrade -y
sudo apt-get install libssl-dev libcurl4-openssl-dev python3.12 python3.12-venv python3.12-dev -y
sudo apt-get install build-essential 
ninja-build python3-dev


wget https://developer.download.nvidia.com/compute/cudnn/9.12.0/local_installers/cudnn-local-repo-ubuntu2204-9.12.0_1.0-1_arm64.deb

 sudo dpkg -i cudnn-local-repo-ubuntu2204-9.12.0_1.0-1_arm64.deb
 sudo cp /var/cudnn-local-repo-ubuntu2204-9.12.0/cudnn-*-keyring.gpg /usr/share/keyrings/

 sudo apt-get update
sudo apt-get -y install cudnn
sudo apt-get -y install cudnn9-cuda-12


python3.12 -m venv venv
source venv/bin/activate


pip install --upgrade pip setuptools wheel

curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

source $HOME/.cargo/env


git clone https://github.com/facebookresearch/xformers.git
cd xformers
git submodule update --init --recursive



pip install ninja
pip install --upgrade setuptools twine setuptools-scm wheel setuptools



export TORCH_CUDA_ARCH_LIST="8.0;8.6;9.0"
pip install torch==2.7.1 torchaudio==2.7.1 torchvision --index-url https://download.pytorch.org/whl/cu128



export MAX_JOBS=3
python setup.py bdist_wheel


pip install xformer*.whl