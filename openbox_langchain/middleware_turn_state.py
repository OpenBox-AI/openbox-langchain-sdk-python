"""Per-run turn state for ``OpenBoxLangChainMiddleware`` (M17).

Turn identity (workflow_id/run_id/pre-screen verdict/sync-mode) MUST live off
``self`` — the old middleware stored these as instance attributes reset in
``before_agent``, so two concurrent ``agent.ainvoke`` calls on ONE middleware
instance would cross-contaminate turn identity and consume each other's
pre-screen verdict.

It also must NOT live in a ``ContextVar`` bound in ``before_agent`` — LangGraph
runs each node (``model``, ``tools``) as a separate task whose contextvars
context is copied from the GRAPH-INVOCATION parent, not from ``before_agent``'s
task, so a ContextVar set there is invisible to the wrap hooks (confirmed
empirically against a real ``create_agent(...).invoke()``/``.ainvoke()``).

Instead, ``MiddlewareTurnState`` is carried in the LangGraph STATE itself (see
``OpenBoxAgentState.ob_turn`` in ``middleware.py``): ``before_agent`` returns it
as a state update, LangGraph threads it through every node of THAT invocation,
and the wrap hooks read it back off ``request.state["ob_turn"]``. This is
per-invocation by construction (each ``.invoke()``/``.ainvoke()`` gets its own
state), so concurrent invocations never share or overwrite each other's turn
identity — no shared mutable instance or contextvar involved.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from openbox_core.contracts.results import EvaluationResult

__all__ = [
    "MiddlewareTurnState",
    "extract_thread_id",
    "require_turn_state",
    "require_turn_state_from_request",
]


def extract_thread_id(runtime: Any) -> str | None:
    """Best-effort ``thread_id`` extraction from the LangGraph ``Runtime.config``.

    Used only to prefix minted workflow/run ids for readability — never for
    uniqueness (see :meth:`MiddlewareTurnState.new`). Returns None on any
    shape mismatch rather than raising.
    """
    config = getattr(runtime, "config", None) or {}
    if not isinstance(config, dict):
        return None
    configurable = config.get("configurable", {})
    if not isinstance(configurable, dict):
        return None
    thread_id = configurable.get("thread_id")
    return thread_id if isinstance(thread_id, str) else None


@dataclass(frozen=True)
class MiddlewareTurnState:
    """Immutable per-invocation state for a single agent run.

    Carried by value in the graph state (``OpenBoxAgentState.ob_turn``) rather
    than mutated in place — each hook that needs to "consume" the pre-screen
    verdict does so by DERIVING whether it is the first LLM call from
    ``request.messages`` (no prior ``AIMessage``) instead of flipping a mutable
    flag, so this dataclass never needs post-construction mutation and stays
    safe to freeze.

    Args:
        workflow_id: Governance workflow identity for this run.
        run_id: Governance run identity for this run.
        sync_mode: True when bound from a sync hook (invoke/stream), False
            for async (ainvoke/astream) — controls which client verb the
            hooks use for gate calls (sync vs async httpx).
        pre_screen_response: The pre-screen ``EvaluationResult`` evaluated
            against the initial user prompt in ``before_agent`` (or None if no
            user prompt was present). Reused by the FIRST LLM call of the run
            (detected via no prior ``AIMessage`` in the model request's
            messages) instead of re-evaluating; ignored by any later call.
    """

    workflow_id: str
    run_id: str
    sync_mode: bool
    pre_screen_response: EvaluationResult | None = None

    @classmethod
    def new(cls, *, sync_mode: bool, thread_id: str | None = None) -> MiddlewareTurnState:
        """Mint fresh workflow/run identity for a new agent invocation.

        ``thread_id`` (when the LangGraph runtime config carries one) prefixes
        the minted ids for readability/correlation, matching prior behavior —
        but the random suffix is what actually guarantees per-run uniqueness,
        so concurrent invocations on the same thread never collide.
        """
        turn = uuid.uuid4().hex
        prefix = thread_id or "langchain"
        return cls(
            workflow_id=f"{prefix}-{turn[:16]}",
            run_id=f"{prefix}-run-{turn[16:32]}",
            sync_mode=sync_mode,
        )

    def with_pre_screen(self, response: EvaluationResult | None) -> MiddlewareTurnState:
        """Return a copy carrying ``response`` as the pre-screen verdict.

        Used by ``before_agent`` to attach the freshly-evaluated pre-screen
        result before returning the state update (the dataclass is frozen, so
        this replaces rather than mutates).
        """
        from dataclasses import replace

        return replace(self, pre_screen_response=response)


def require_turn_state(state: Any) -> MiddlewareTurnState:
    """Fetch ``ob_turn`` off a ``before_agent``/``after_agent`` ``state`` arg.

    Raises if before_agent never ran for this invocation — a missing turn
    state means a hook fired outside a before_agent/after_agent bracket, a
    middleware wiring bug, not a recoverable runtime condition.
    """
    turn = state.get("ob_turn") if isinstance(state, dict) else getattr(state, "ob_turn", None)
    if turn is None:
        raise RuntimeError(
            "OpenBoxLangChainMiddleware: no turn state bound — "
            "before_agent/abefore_agent must run before wrap hooks fire"
        )
    return turn  # type: ignore[no-any-return]


def require_turn_state_from_request(request: Any) -> MiddlewareTurnState:
    """Fetch ``ob_turn`` off a ``wrap_model_call``/``wrap_tool_call`` request.

    ``request.state`` is the same graph state ``before_agent`` updated —
    LangGraph threads it through every node of THIS invocation.
    """
    return require_turn_state(getattr(request, "state", None) or {})
