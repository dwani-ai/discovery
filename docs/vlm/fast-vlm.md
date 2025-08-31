Fast VLm

https://huggingface.co/apple/FastVLM-0.5B

pip install timm


For CPU

uvicorn fast-vlm-api:app --reload --host 0.0.0.0 --port 8000 -- --device cpu

for GPU
uvicorn fast-vlm-api:app --reload --host 0.0.0.0 --port 8000 -- --device cuda

default with cuda 

uvicorn fast-vlm-api:app --reload --host 0.0.0.0 --port 8000

python fast-vlm-api.py --device cpu

python fast-vlm-api.py --device cuda
