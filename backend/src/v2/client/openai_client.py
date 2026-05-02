from openai import AsyncOpenAI
from config.settings import settings

def get_openai_client(model: str | None = None) -> AsyncOpenAI:
    model = model or settings.DWANI_LLM_MODEL
    valid_models = {settings.DWANI_LLM_MODEL, "gpt-oss"}
    if model not in valid_models:
        raise ValueError(f"Invalid model: {model}")
    return AsyncOpenAI(api_key="http", base_url=settings.DWANI_API_BASE_URL)