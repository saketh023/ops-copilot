from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from app.llm.factory import get_llm_client
from dotenv import load_dotenv

from app.tools.log_tools import summarize_logs

from app.tools.detector import detect_tool
from app.tools.registry import get_tool, register_tool

import json
import re

SYSTEM_PROMPT_JSON = """
You are Ops Copilot, a backend reliability assistant.

Return ONLY valid JSON.
Do not include markdown.
Do not include code fences.

The JSON must match this schema:

{
  "diagnosis": "string",
  "possible_causes": ["string", "..."],
  "next_checks": ["string", "..."],
  "confidence": "low|medium|high"
}

Rules:
- possible_causes must contain 2 to 6 items
- next_checks must contain 2 to 6 items
- confidence must be exactly one of: low, medium, high
"""


def extract_json(text: str) -> dict:
    text = text.strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError("No JSON found in model output")

    return json.loads(match.group(0))


app = FastAPI(title="Ops Copilot", version="0.2")

load_dotenv()
llm = get_llm_client()

register_tool("summarize_logs", summarize_logs)


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
        tool_name = detect_tool(req.question)

        if tool_name:
            tool_fn = get_tool(tool_name)
            tool_result = tool_fn(req.question)

            tool_prompt = f"""
        User provided input that requires tool: {tool_name}

        Tool Result:
        {tool_result}

        Use the tool result to diagnose the issue.
        """

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT_JSON},
                {"role": "user", "content": tool_prompt},
            ]
        else:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT_JSON},
                {"role": "user", "content": req.question},
            ]

        raw = llm.chat(
            messages=messages,
            temperature=req.temperature,
            num_ctx=req.num_ctx,
        )

        structured = extract_json(raw)

        return {
            "provider": getattr(llm, "name", "unknown"),
            "model": getattr(llm, "model", "unknown"),
            "result": structured,
        }

    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))
