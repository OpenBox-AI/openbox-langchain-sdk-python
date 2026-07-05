"""Local structural stand-in for ``langgraph.prebuilt.tool_node.ToolCallRequest``.

Split out of ``middleware.py`` (keeps that file under 200 lines) and to make
the langgraph-avoidance rationale a single, easy-to-find place. Only the
fields the middleware reads are declared — ``wrap_tool_call``/``awrap_tool_call``
accept the REAL LangGraph-produced request object, which satisfies this
Protocol structurally at runtime (duck typing), so no ``langgraph`` import is
needed in this package's boundary (P2-new-C).

``state``/``runtime`` were added alongside ``tool_call`` once the M17 turn-state
fix moved turn identity into the graph state (``OpenBoxAgentState.ob_turn``,
see ``middleware.py``) — the wrap hooks now read ``request.state["ob_turn"]``
instead of a ContextVar, so the Protocol must expose ``state`` structurally too.
"""

from __future__ import annotations

from typing import Any, Protocol

__all__ = ["ToolCallRequestLike"]


class ToolCallRequestLike(Protocol):
    """Structural stand-in exposing the fields the middleware reads.

    ``runtime`` is declared as ``Any`` (optional in practice — accessed only
    via ``getattr`` with a default by callers) since the middleware never
    inspects its shape, only ``state``.
    """

    tool_call: dict[str, Any]
    state: Any
    runtime: Any
