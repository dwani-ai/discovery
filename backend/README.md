

docker build -t dwani/discovery-server:latest -f app.Dockerfile .
docker compose -f discovery-server.yml up -d


---

python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt

python src/server/main.py