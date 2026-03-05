import re
from collections import Counter


def summarize_logs(log_text: str):
    """
    Basic log summarization tool.
    Extracts error types and frequency.
    """

    lines = log_text.strip().split("\n")

    error_lines = [l for l in lines if "ERROR" in l or "WARN" in l]

    messages = []

    for line in error_lines:
        # remove timestamp
        cleaned = re.sub(r"^\S+\s+\S+\s+", "", line)
        messages.append(cleaned)

    counts = Counter(messages)

    return {
        "total_lines": len(lines),
        "error_lines": len(error_lines),
        "top_errors": dict(counts.most_common(5))
    }