"""Pure message/response extraction helpers for the middleware hooks.

Split out of ``middleware_hooks.py`` to stay under 200 lines per file. No
middleware coupling — these operate on raw LangChain message/response shapes
only, so they are trivially unit-testable in isolation.
"""

from __future__ import annotations

from typing import Any

__all__ = ["extract_last_user_message", "extract_response_metadata"]


def extract_last_user_message(messages: list[Any]) -> str | None:
    """Extract last human/user message text from agent state messages."""
    for msg in reversed(messages):
        if isinstance(msg, dict):
            if msg.get("role") in ("user", "human"):
                content = msg.get("content")
                return content if isinstance(content, str) else None
        elif hasattr(msg, "type") and msg.type in ("human", "generic"):
            content = msg.content
            return content if isinstance(content, str) else None
    return None


def extract_response_metadata(response: Any) -> dict[str, Any]:
    """Extract tokens, model name, completion from a model response."""
    result: dict[str, Any] = {}
    ai_msg = response
    if hasattr(response, "message"):
        ai_msg = response.message

    if hasattr(ai_msg, "response_metadata"):
        meta = ai_msg.response_metadata or {}
        result["llm_model"] = meta.get("model_name") or meta.get("model")

    usage = getattr(ai_msg, "usage_metadata", None) or {}
    if isinstance(usage, dict):
        result["input_tokens"] = usage.get("input_tokens") or usage.get("prompt_tokens")
        result["output_tokens"] = usage.get("output_tokens") or usage.get("completion_tokens")
        inp = result.get("input_tokens") or 0
        out = result.get("output_tokens") or 0
        result["total_tokens"] = inp + out if (inp or out) else None

    content = getattr(ai_msg, "content", None)
    if isinstance(content, str):
        result["completion"] = content
    elif isinstance(content, list):
        text_parts = [
            p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"
        ]
        result["completion"] = " ".join(text_parts) if text_parts else None

    result["has_tool_calls"] = bool(getattr(ai_msg, "tool_calls", None))
    return result
