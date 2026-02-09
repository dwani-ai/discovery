from google.adk.agents.llm_agent import Agent

# Mock tool implementation
def get_current_time(city: str) -> dict:
    """Returns the current time in a specified city."""
    return {"status": "success", "city": city, "time": "10:30 AM"}


import os
from google.adk.models.lite_llm import LiteLlm

# ADK's LiteLLM wrapper reads specific env vars, but explicit config is safer for local models
model = LiteLlm(
    model=os.getenv("LITELLM_MODEL_NAME"),          # ← changed from model_name
    api_base=os.getenv("LITELLM_API_BASE"),
    api_key=os.getenv("LITELLM_API_KEY"),
)

root_agent = Agent(
    model=model,
    name='root_agent',
    description="Tells the current time in a specified city.",
    instruction="You are a helpful assistant that tells the current time in cities. Use the 'get_current_time' tool for this purpose.",
    tools=[get_current_time],
)