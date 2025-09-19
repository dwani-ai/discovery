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

Cuda -d Container Toolkit


-- 



https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html


curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update

export NVIDIA_CONTAINER_TOOLKIT_VERSION=1.17.8-1
  sudo apt-get install -y \
      nvidia-container-toolkit=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
      nvidia-container-toolkit-base=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
      libnvidia-container-tools=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
      libnvidia-container1=${NVIDIA_CONTAINER_TOOLKIT_VERSION}

update - daemon.json : /etc/docker/daemon.json


sudo systemctl restart docker

sudo docker run --rm --runtime=nvidia --gpus all ubuntu nvidia-smi


