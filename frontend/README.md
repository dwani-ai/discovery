client for Discovery

- Production 
```bash
npm install
npm run build
npm run deploy
```

- Local run
```bash
npm install
npm run dev
```


--

- docker 
    - Developer Environment
    ```bash
        docker build -t dwani/discovery-ux-dev:latest -f Dockerfile.dev .

        docker push dwani/discovery-ux-dev:latest
    ```
    - Production Environment
    ```bash
        docker build -t dwani/discovery-ux-prod:latest -f Dockerfile.prod .

        docker push dwani/discovery-ux-prod:latest
    ```