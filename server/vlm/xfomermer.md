git clone https://github.com/facebookresearch/xformers.git
cd xformers

python3.12 -m venv venv
source venv/bin/activate


pip install ninja
pip install --upgrade setuptools twine setuptools-scm wheel setuptools


export TORCH_CUDA_ARCH_LIST="8.0;8.6;9.0"
pip install torch==2.7.1 torchaudio==2.7.1 torchvision --index-url https://download.pytorch.org/whl/cu128




python setup.py bdist_wheel


or 
pip install -v --no-build-isolation -U git+https://github.com/facebookresearch/xformers.git@main#egg=xformers

