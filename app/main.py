from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from app.llm.factory import get_llm_client
from dotenv import load_dotenv

app = FastAPI(title="Ops Copilot", version="0.2")

load_dotenv()
llm = get_llm_client()

class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    temperature: float = Field(0.2, ge=0.0, le=1.0)
    num_ctx: int = Field(4096, ge=512, le=16384)

OLLAMA_URL = "http://localhost:11434"
MODEL_NAME = "qwen2.5:7b"


@app.get("/health")
def health():
    return {"status": "running"}


@app.post("/ask")
def ask(req: AskRequest):
    try:
        answer = llm.chat(
            messages=[{"role": "user", "content": req.question}],
            temperature=req.temperature,
            num_ctx=req.num_ctx,
        )

        return {
            "provider": getattr(llm, "name", "unknown"),
            "model": getattr(llm, "model", "unknown"),
            "answer": answer,
        }

    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))
    