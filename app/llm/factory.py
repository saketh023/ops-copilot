import os
from app.llm.ollama_client import OllamaClient


def get_llm_client():
    provider = os.getenv("LLM_PROVIDER", "ollama").lower()

    if provider == "ollama":
        base_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
        model = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
        return OllamaClient(base_url=base_url, model=model)

    raise ValueError(f"Unsupported LLM provider: {provider}")
