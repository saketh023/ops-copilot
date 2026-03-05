from typing import Callable, Dict, Any

ToolFn = Callable[[str], Dict[str, Any]]

TOOLS: Dict[str, ToolFn] = {}


def register_tool(name: str, fn: ToolFn) -> None:
    TOOLS[name] = fn


def get_tool(name: str) -> ToolFn:
    if name not in TOOLS:
        raise KeyError(f"Tool not found: {name}")
    return TOOLS[name]


def list_tools() -> list[str]:
    return sorted(TOOLS.keys())