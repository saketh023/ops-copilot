from fastapi import FastAPI
import requests

app = FastAPI()

OLLAMA_URL = "http://localhost:11434"
MODEL_NAME = "qwen2.5-7B"


@app.get("/health")
def health():
    return {"status": "running"}


@app.get("/ask")
def ask(question: str):
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": question}],
        "stream": False,
    }

    response = requests.post(f"{OLLAMA_URL}/api/chat", json=payload)
    result = response.json()

    return {"answer": result["message"]["content"]}
