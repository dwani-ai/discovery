Client 


docker build -t dwani/discovery_ux:latest -f client.Dockerfile .
docker push dwani/discovery_ux:latest

docker run -p 80:8000 --env DWANI_API_BASE_URL=$DWANI_API_BASE_URL dwani/discovery_ux:latest

Server


docker build -t dwani/discovery_server:latest -f server.Dockerfile .
docker push dwani/discovery_server:latest

docker run -p 18888:18888 --env DWANI_API_BASE_URL=$DWANI_API_BASE_URL dwani/discovery_server:latest


--


- Client Docker Run
```bash
docker run -p 80:8000 --env DWANI_API_BASE_URL=<server_ip> dwani/discovery_ux:latest
```
- Server  Docker Run 

---

