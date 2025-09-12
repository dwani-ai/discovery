To - run discovery server on windows


- run Powershell as Administrator and run below command

Set-ExecutionPolicy RemoteSigned

python.exe -m venv venv

.\venv\Scripts\Activate.ps1

pip.exe install -r server-requirements.txt


---
As - powershell admin
choco install poppler


