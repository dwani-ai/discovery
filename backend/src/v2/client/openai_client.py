from openai import AsyncOpenAI
from config.settings import settings

def get_openai_client(model: str = "gemma3") -> AsyncOpenAI:
    valid_models = {"gemma3", "gpt-oss"}
    if model not in valid_models:
        raise ValueError(f"Invalid model: {model}")
    return AsyncOpenAI(api_key="http", base_url=settings.DWANI_API_BASE_URL)