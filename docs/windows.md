To - run discovery server on windows


- run Powershell as Administrator and run below command

Set-ExecutionPolicy RemoteSigned

python.exe -m venv venv

.\venv\Scripts\Activate.ps1

pip.exe install -r server-requirements.txt


---
As - powershell admin
choco install poppler


Add to PATH
C:\ProgramData\chocolatey\lib\poppler\tools


---

Install Docker desktop

as powershell admin
wsl --update

 docker build -t dwani/discovery_server:latest -f server.Dockerfile .
docker run -p 18889:18889 --env DWANI_API_BASE_URL=0.0.0.0 dwani/discovery_server:latest


---

Install cuda toolkit for windows


---
llama-cpp
In powershell

winget install llama.cpp


