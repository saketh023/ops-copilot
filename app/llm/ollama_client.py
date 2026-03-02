import requests
from typing import List, Dict, Any


class OllamaClient:
    def __init__(self, base_url: str, model: str):
        self.name = "ollama"
        self.base_url = base_url.rstrip("/")
        self.model = model

    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.2, num_ctx: int = 4096) -> str:
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_ctx": num_ctx,
            },
        }

        try:
            resp = requests.post(f"{self.base_url}/api/chat", json=payload, timeout=120)
        except requests.RequestException as e:
            raise RuntimeError(f"Failed to reach Ollama: {e}")

        if resp.status_code != 200:
            raise RuntimeError(f"Ollama error {resp.status_code}: {resp.text}")

        try:
            data = resp.json()
        except ValueError:
            raise RuntimeError(f"Invalid JSON from Ollama: {resp.text}")

        if "message" not in data:
            raise RuntimeError(f"Unexpected response shape from Ollama: {data}")

        return data["message"].get("content", "")