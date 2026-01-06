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