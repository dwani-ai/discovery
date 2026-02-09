Google Agent Development Kit

pip install google-adk

adk create loan_agent

--
LITELLM_MODEL_NAME="openai/qwen3-coder" # 'openai/' prefix treats it as generic OpenAI format
LITELLM_API_BASE="http://localhost:8000/v1"
LITELLM_API_KEY="sk-dummy"

--
- For Terminal 
adk run loan_agent


- For Web Browser
adk web --port 8000


--

REference

https://google.github.io/adk-docs/get-started/quickstart/#run-your-agent

    