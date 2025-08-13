
git clone https://github.com/flashinfer-ai/flashinfer.git --recursive
cd flashinfer
python -m pip install -v .

python -m build --no-isolation --wheel
python -m pip install dist/flashinfer_*.whl