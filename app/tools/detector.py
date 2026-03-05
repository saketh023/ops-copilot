from typing import Optional

LOG_KEYWORDS = ["ERROR", "WARN", "EXCEPTION", "TRACEBACK"]


def detect_tool(text: str) -> Optional[str]:
    t = text.upper()
    if any(k in t for k in LOG_KEYWORDS):
        return "summarize_logs"
    return None