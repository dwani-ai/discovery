

docker build -t my-go-app .

docker run -p 8080:8080 --env VLLM_IP=<your-vllm-ip> my-go-app

