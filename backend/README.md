Server for Discovery

- Production 
```bash
docker build -t dwani/discovery-server:latest -f app.Dockerfile .
docker compose -f discovery-server.yml up -d
```

- Local run
```bash
python3 -m venv venv
source venv/bin/activate

pip install --upgrade pip setuptools wheel

pip install -r requirements.txt

python src/server/main.py
```

--- 

Test

- LLM Mocked Test
pytest tests/test_services/test_extraction.py -v
pytest tests/test_routes/test_chat.py -v


--

docker multi-stage build

- cpu
    - docker build -t dwani/discovery-server-prod-cpu:v-0-0-1-dec-2025 -f cpu.Dockerfile .
- cuda
    - docker build -t dwani/discovery-server-prod:v-0-0-1-dec-2025 -f Dockerfile.prod .

 
 -- 

- cpu only optimisation

 ```bash
python3.10 -m venv venv
source venv/bin/activate

pip install --upgrade pip setuptools wheel
pip install --no-deps   torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

pip install -r cpu-requirements.txt

python src/server/main.py
```
