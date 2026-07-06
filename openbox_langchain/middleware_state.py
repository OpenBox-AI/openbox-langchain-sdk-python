"""Graph-state schema for ``OpenBoxLangChainMiddleware`` (M17 turn-state carriage).

Split out of ``middleware.py`` to stay under 200 lines per file. Requires the
``agent`` extra (imports ``langchain.agents.middleware.types``) — only ever
imported from ``middleware.py``, which already gates on that extra.
"""

from __future__ import annotations

from typing import NotRequired

from langchain.agents.middleware.types import AgentState

from openbox_langchain.middleware_turn_state import MiddlewareTurnState

__all__ = ["OpenBoxAgentState"]


class OpenBoxAgentState(AgentState):
    """Agent state extended with the per-run OpenBox turn-identity slot.

    ``ob_turn`` carries the ``MiddlewareTurnState`` for THIS invocation
    (workflow_id/run_id/sync_mode/pre-screen verdict), written once by
    ``before_agent``/``abefore_agent`` and read by every wrap hook via
    ``request.state["ob_turn"]``.

    This REPLACES a ``ContextVar``-based mechanism that does not work through
    the real ``create_agent(...).invoke()``/``.ainvoke()`` path: LangGraph runs
    each node (``model``, ``tools``) as a separate task whose contextvars
    context is copied from the graph-invocation parent, NOT from
    ``before_agent``'s task, so a ContextVar set there is invisible in the wrap
    hooks. Graph state, by contrast, is threaded through every node of a
    SINGLE invocation by LangGraph itself, so it is per-invocation (hence
    concurrency-safe) by construction — no shared mutable instance state
    required.
    """

    ob_turn: NotRequired[MiddlewareTurnState | None]
