"""Prompt extraction, redaction application, and ``__openbox`` enrichment —
split out of ``lifecycle_events.py`` to stay under 200 lines. Re-exported
there; not a separate public surface.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

__all__ = ["apply_redaction_to_messages", "enrich_activity_input", "extract_human_turn_prompt"]


# ─── Human-turn prompt extraction ──────────────────────────────────────────
#
# Documented gap (M14): when no human/user message is found (e.g. a
# system-only prompt, or a tool-only re-invocation), this returns "" rather
# than None or raising. Callers treat "" as "no prompt to redact/send" — this
# is a deliberate simplification, not a crash-worthy edge case, but it means
# an all-empty prompt sends an ActivityStarted with an empty activity_input
# entry rather than omitting the field. Revisit if a policy needs to
# distinguish "empty human turn" from "no human turn at all".


def extract_human_turn_prompt(messages: Any) -> str:
    """Extract the human/user turn text from a LangChain messages structure.

    Accepts the ``on_chat_model_start`` shape (``list[list[BaseMessage]]``),
    a flat ``list[BaseMessage]``, or dict-shaped messages (``{"role": ...,
    "content": ...}``). Joins all human-authored text found, in order.
    Returns "" when no human message is found (gap M14 — see module note).
    """
    if not isinstance(messages, (list, tuple)):
        return ""
    parts: list[str] = []
    for item in messages:
        if isinstance(item, (list, tuple)):
            for inner in item:
                _append_human_text(inner, parts)
        else:
            _append_human_text(item, parts)
    return "\n".join(parts)


def _append_human_text(msg: Any, parts: list[str]) -> None:
    """Append human-authored text content from a single message to ``parts``."""
    role: str | None = None
    content: Any = None
    if hasattr(msg, "type"):
        role = msg.type
        content = getattr(msg, "content", None)
    elif isinstance(msg, dict):
        role = msg.get("role") or msg.get("type")
        content = msg.get("content")
    if role not in ("human", "user", "generic"):
        return
    if isinstance(content, str):
        if content:
            parts.append(content)
    elif isinstance(content, list):
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                text = part.get("text", "")
                if text:
                    parts.append(text)


# ─── Redaction application ─────────────────────────────────────────────────


def apply_redaction_to_messages(messages: Sequence[Any], redacted_input: Any) -> bool:
    """Mutate ``messages`` in place, replacing the last human turn's content
    with ``redacted_input`` (from ``GuardrailsResult.redacted_input``).

    Returns True if a message was mutated, False otherwise (no human turn
    found, or ``redacted_input`` did not yield replacement text).
    """
    redacted_text = _coerce_redacted_text(redacted_input)
    if not redacted_text:
        return False
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        if hasattr(msg, "type") and msg.type in ("human", "user", "generic"):
            msg.content = redacted_text
            return True
        if isinstance(msg, dict) and msg.get("role") in ("user", "human"):
            msg["content"] = redacted_text
            return True
    return False


def _coerce_redacted_text(redacted_input: Any) -> str | None:
    """Normalize a ``GuardrailsResult.redacted_input`` value to plain text."""
    if isinstance(redacted_input, str):
        return redacted_input or None
    if isinstance(redacted_input, list) and redacted_input:
        first = redacted_input[0]
        if isinstance(first, dict):
            text = first.get("prompt") or first.get("text")
            return text or None
        if isinstance(first, str):
            return first or None
    return None


# ─── __openbox activity-input enrichment (wire parity, C9 Non-Goal) ────────
#
# Mirrors `openbox_langgraph/langgraph_handler.py:1249-1278`'s
# `_enrich_activity_input` verbatim in shape and placement: appends a single
# `{"__openbox": {...}}` sentinel to the END of the activity_input list so
# Rego policies can classify tools/subagents without Core changes. Forgery of
# this sentinel by a malicious tool payload (C9) is a documented Non-Goal —
# callers append it themselves, after building activity_input from trusted
# fields, so this function does NOT sanitize or strip existing __openbox
# entries from caller-supplied input.


def enrich_activity_input(
    base_input: list[Any] | None,
    *,
    tool_type: str | None,
    subagent_name: str | None,
) -> list[Any] | None:
    """Append an ``__openbox`` metadata entry to ``activity_input``.

    Only appended when ``tool_type`` or ``subagent_name`` is set — unclassified
    tools get no sentinel (matches the LangGraph handler's skip condition).
    """
    if tool_type is None and subagent_name is None:
        return base_input
    meta: dict[str, Any] = {}
    if tool_type is not None:
        meta["tool_type"] = tool_type
    if subagent_name is not None:
        meta["subagent_name"] = subagent_name
    result = list(base_input) if base_input else []
    result.append({"__openbox": meta})
    return result
