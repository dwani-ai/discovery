UX for discovery


- export DWANI_IP=0.0.0.0 / SERVER_IP
- python ux.py

- With Docker
  - docker run -p 80:8000 --env DWANI_IP=$DWANI_IP dwani/discovery_ux:latest


- build with Docker
    docker build -t dwani/discovery_ux:latest -f client.Dockerfile .
    docker push dwani/discovery_ux:latest


- With Docker Install - 

Docker Setup

https://docs.docker.com/engine/install/

```bash
for pkg in docker.io docker-doc docker-compose docker-compose-v2 podman-docker containerd runc; do sudo apt-get remove $pkg; done
```

# Add Docker's official GPG key:
```bash

sudo apt-get update
sudo apt-get install ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc
```

# Add the repository to Apt sources:
```bash

echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
```

```bash
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
```

```bash

sudo docker run hello-world


sudo groupadd docker

sudo usermod -aG docker $USER

newgrp docker

docker run hello-world

```


Client - Server 



docker run -p 80:8000 --env VLLM_IP=$VLLM_IP dwani/discovery_ux:latest
