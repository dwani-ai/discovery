
python3.12 -m venv venv
source venv/bin/activate
git clone https://github.com/flashinfer-ai/flashinfer.git --recursive
cd flashinfer
python -m pip install -v .

pip install --upgrade setuptools twine setuptools-scm wheel setuptools build

python -m build --no-isolation --wheel
python -m pip install dist/flashinfer_*.whl