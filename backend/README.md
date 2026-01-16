Server for Discovery

- Python : 3.10 

- Docker run
```bash
docker build -t dwani/discovery-server:latest -f Dockerfile .
docker compose -f discovery-server.yml up -d
```

- Local run
```bash
```bash
python3.10 -m venv venv
source venv/bin/activate

pip install --upgrade pip setuptools wheel
pip install --no-deps   torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

pip install -r cpu-requirements.txt

python src/server/main.py
```
```

--- 

Test

- LLM Mocked Test
pytest tests/test_services/test_extraction.py -v
pytest tests/test_routes/test_chat.py -v


--



