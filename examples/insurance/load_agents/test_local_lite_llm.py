import os
from dotenv import load_dotenv
from litellm import completion

load_dotenv()

response = completion(
    model=os.getenv("LITELLM_MODEL_NAME"),
    api_base=os.getenv("LITELLM_API_BASE"),
    api_key=os.getenv("LITELLM_API_KEY"),
    messages=[{"role": "user", "content": "Hello Gemma4, are you ready to write some python code?"}]
)

print("Response from Gemma4:", response.choices[0].message.content)
