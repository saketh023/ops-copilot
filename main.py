from fastapi import FastAPI, HTTPException
import requests

app = FastAPI()

OLLAMA_URL = "http://localhost:11434"
MODEL_NAME = "qwen2.5:7b"


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
    
    try:
        response = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=120)
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Failed to reach Ollama: {e}")
    
    if response.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail=f"Ollama error {response.status_code}: {response.text}",
        )

    try:
        result = response.json()
    except ValueError:
        raise HTTPException(status_code=502, detail=f"Invalid JSON from Ollama: {response.text}")

    # Debug: if shape is not what we expect, return it
    if "message" not in result:
        raise HTTPException(
            status_code=502,
            detail={
                "error": "Unexpected response shape from Ollama",
                "ollama_response": result,
            },
        )

    return {"answer": result["message"].get("content", "")}
