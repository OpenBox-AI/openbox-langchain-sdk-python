"""Shared constructor options for the pure LangChain-Core callback handlers.

Split out from ``core_callback.py`` so both
``OpenBoxLangChainCoreAsyncCallbackHandler`` and
``OpenBoxLangChainCoreSyncCallbackHandler`` build from one options dataclass
without a shared base class import cycle.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from openbox_core.context import register_trace as default_register_trace
from openbox_core.context import unregister_trace as default_unregister_trace

if TYPE_CHECKING:
    from openbox_core.contracts.context import ActivityContext
    from openbox_core.contracts.results import EvaluationResult
    from openbox_core.runtime import OpenBoxRuntime

    from openbox_langchain.activity_bridge import ActivityBridge
    from openbox_langchain.core_callback_llm_logic import LLMTraceHandle

__all__ = ["OpenBoxLangChainCoreCallbackOptions"]


@dataclass
class OpenBoxLangChainCoreCallbackOptions:
    """Constructor args shared by both core callback handlers.

    Args:
        runtime: The ``OpenBoxRuntime`` providing ``.gate`` and ``.adapter``.
        bridge: The ``ActivityBridge`` ownership channel (handler/middleware
            owned — never a module-global).
        workflow_id: Governance workflow identity.
        run_id: Governance run identity.
        workflow_type: Governance workflow type label.
        task_queue: Optional task-queue label (wire field passthrough).
        session_id: Optional session identity (wire ``extra`` field).
        agent_name: Optional agent name (wire ``extra`` field).
        send_tool_start_event: Emit ``ActivityStarted`` for tool calls.
        send_tool_end_event: Emit ``ActivityCompleted`` for tool calls.
        send_llm_start_event: Emit ``ActivityStarted`` for LLM calls.
        send_llm_end_event: Emit ``ActivityCompleted`` for LLM calls.
        tool_type_resolver: Optional ``tool_name -> tool_type | None`` mapper
            for ``__openbox`` activity-input enrichment.
        pre_screen_response: An already-evaluated verdict for the FIRST LLM
            call (avoids a duplicate gate call when an upstream layer already
            screened the initial prompt).
        pre_screen_activity_id: The activity_id the pre-screen verdict was
            evaluated against — reused as the LLMStarted activity_id for call 1.
        register_trace: Injected trace-registration callable
            (``(trace_id, ActivityContext) -> None``). Defaults to the base
            SDK's process-wide ``ContextStore``. Phase 5 (LangGraph LLM
            ownership) injects a registry-backed callable instead.
        unregister_trace: Injected trace-unregistration callable
            (``(trace_id) -> None``). Same default/override rule.
        record_less_ok: When True (pure-LangChain default), the callback may
            send a lifecycle event even when no bridge record exists yet
            (record-less send). When False (the LangGraph embedding sets
            this), the callback must NOT send without a record — prevents a
            nested-graph double-send (C8).
    """

    runtime: OpenBoxRuntime
    bridge: ActivityBridge
    workflow_id: str
    run_id: str
    workflow_type: str
    task_queue: str | None = None
    session_id: str | None = None
    agent_name: str | None = None
    send_tool_start_event: bool = True
    send_tool_end_event: bool = True
    send_llm_start_event: bool = True
    send_llm_end_event: bool = True
    tool_type_resolver: Callable[[str], str | None] | None = None
    pre_screen_response: EvaluationResult | None = None
    pre_screen_activity_id: str | None = None
    register_trace: Callable[[int | str, ActivityContext], None] = default_register_trace
    unregister_trace: Callable[[int | str], None] = default_unregister_trace
    record_less_ok: bool = True
    llm_trace_handles: dict[str, LLMTraceHandle] = field(default_factory=dict)
