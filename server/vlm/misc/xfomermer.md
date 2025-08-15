

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


curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

source $HOME/.cargo/env


git clone https://github.com/facebookresearch/xformers.git
cd xformers

- for pytorch 2.8 
    - git checkout 5d4b92a5e5a9c6c6d4878283f47d82e17995b468

- fpr pytorch 2.7
    - git checkout eb0946a363464da96ea40afd1a7f72a907c25497


git submodule update --init --recursive



pip install ninja
pip install --upgrade setuptools twine setuptools-scm wheel setuptools ninja



export TORCH_CUDA_ARCH_LIST="8.0;8.6;9.0"
pip install torch==2.7.1 torchaudio==2.7.1 torchvision --index-url https://download.pytorch.org/whl/cu128

export CUDA_HOME=/usr/local/cuda/

export MAX_JOBS=3
python setup.py bdist_wheel


pip install xformer*.whl