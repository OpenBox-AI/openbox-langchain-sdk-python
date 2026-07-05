"""LLM-lifecycle logic shared by the async and sync core callback handlers.

Per Phase 0's measured propagation matrix, chat-model raises PROPAGATE under
``raise_error=True, run_inline=True`` — but THIS phase keeps the LLM path
telemetry + redaction + trace-registration only; enforcement wiring for
LLM-start is deferred to the integrating layer (a later phase decides,
grounded in the Phase 0 result). Do not add an enforcing raise to LLM-start
here.

H11 alias: the first LLM call's bridge record may be keyed differently from
the callback's own ``run_id`` (e.g. an upstream pre-screen row keyed
``"{run_id}-pre"``). ``prepare_llm(..., event_run_id=...)`` registers the
alias so completion resolves via ``get_by_event_run_id`` without emitting an
orphan ``-c`` row.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from openbox_core.contracts.context import ActivityContext
from openbox_core.otel.trace_context import raw_trace_id
from opentelemetry import trace as otel_trace

from openbox_langchain.lifecycle_events import (
    apply_redaction_to_messages,
    build_activity_completed,
    build_activity_started,
    extract_human_turn_prompt,
)

if TYPE_CHECKING:
    from openbox_core.contracts.results import EvaluationResult

    from openbox_langchain.core_callback_options import OpenBoxLangChainCoreCallbackOptions

__all__ = [
    "apply_llm_redaction",
    "build_llm_started_envelope",
    "register_llm_trace",
    "send_llm_completed",
    "send_llm_completed_sync",
    "unregister_llm_trace",
]


def build_llm_started_envelope(
    options: OpenBoxLangChainCoreCallbackOptions,
    activity_id: str,
    llm_type: str,
    messages: Any,
) -> Any:
    """Build the LLMStarted (ActivityStarted) envelope from raw callback messages.

    Prompt extraction returns "" when no human turn is found (documented gap
    M14, see ``lifecycle_events.extract_human_turn_prompt``) — an empty
    activity_input entry is sent rather than omitting the field, so downstream
    policies see a consistent shape.
    """
    prompt = extract_human_turn_prompt(messages)
    return build_activity_started(
        workflow_id=options.workflow_id,
        run_id=options.run_id,
        workflow_type=options.workflow_type,
        activity_id=activity_id,
        activity_type=llm_type,
        task_queue=options.task_queue,
        activity_input=[{"prompt": prompt}],
        session_id=options.session_id,
        agent_name=options.agent_name,
    )


def apply_llm_redaction(result: EvaluationResult, messages: Any) -> bool:
    """Apply guardrails redaction to the LLM-bound messages, in place.

    Only mutates when ``guardrails.input_type == "activity_input"`` — output
    redaction (``"activity_output"``) never applies to the pre-call prompt.
    Returns True if a message was mutated.
    """
    guardrails = result.guardrails
    if guardrails is None or not guardrails.redacted_input:
        return False
    if guardrails.input_type and guardrails.input_type != "activity_input":
        return False
    flat_messages = _flatten_messages(messages)
    return apply_redaction_to_messages(flat_messages, guardrails.redacted_input)


def _flatten_messages(messages: Any) -> list[Any]:
    """Flatten the ``on_chat_model_start`` ``list[list[BaseMessage]]`` shape
    (or an already-flat list) into a single mutable list for redaction."""
    if not isinstance(messages, (list, tuple)):
        return []
    flat: list[Any] = []
    for item in messages:
        if isinstance(item, (list, tuple)):
            flat.extend(item)
        else:
            flat.append(item)
    return flat


async def send_llm_completed(
    options: OpenBoxLangChainCoreCallbackOptions,
    activity_id: str,
    llm_type: str,
    *,
    result: Any = None,
    error: str | None = None,
) -> EvaluationResult:
    """Send LLMCompleted (ActivityCompleted, same id), stash verdict, mark sent.

    ``gate.aevaluate`` ONLY (C4) — never adapter-enforcing. Guard callers with
    ``llm_completed_sent`` before calling this from ``on_llm_error`` so a
    ``return_exceptions=True``-captured end that already sent doesn't
    double-send a failed row.
    """
    envelope = build_activity_completed(
        workflow_id=options.workflow_id,
        run_id=options.run_id,
        workflow_type=options.workflow_type,
        activity_id=activity_id,
        activity_type=llm_type,
        task_queue=options.task_queue,
        result=result,
        error=error,
        session_id=options.session_id,
        agent_name=options.agent_name,
    )
    verdict = await options.runtime.gate.aevaluate(envelope)
    options.bridge.stash_completion_result(options.workflow_id, activity_id, verdict)
    options.bridge.mark_sent(options.workflow_id, activity_id, "llm_complete")
    return verdict


def send_llm_completed_sync(
    options: OpenBoxLangChainCoreCallbackOptions,
    activity_id: str,
    llm_type: str,
    *,
    result: Any = None,
    error: str | None = None,
) -> EvaluationResult:
    """Sync counterpart of :func:`send_llm_completed` (uses ``gate.evaluate``)."""
    envelope = build_activity_completed(
        workflow_id=options.workflow_id,
        run_id=options.run_id,
        workflow_type=options.workflow_type,
        activity_id=activity_id,
        activity_type=llm_type,
        task_queue=options.task_queue,
        result=result,
        error=error,
        session_id=options.session_id,
        agent_name=options.agent_name,
    )
    verdict = options.runtime.gate.evaluate(envelope)
    options.bridge.stash_completion_result(options.workflow_id, activity_id, verdict)
    options.bridge.mark_sent(options.workflow_id, activity_id, "llm_complete")
    return verdict


def register_llm_trace(
    options: OpenBoxLangChainCoreCallbackOptions, activity_id: str, llm_type: str
) -> None:
    """Register the current OTel trace id against this LLM's ActivityContext.

    "Before the provider HTTP" (phase-02 step 7) means: whatever span is
    ambient at ``on_chat_model_start`` time — callers that want the provider's
    own HTTP span correlated register a NEW span themselves before invoking
    the model; here we simply bind the context under whatever trace is
    current so Layer 2 HTTP-hook instrumentation resolves it. No trace is
    registered when there is no active OTel span (``trace_id == 0``, the
    no-op default) — nothing to correlate.
    """
    trace_id = raw_trace_id(otel_trace.get_current_span())
    if not trace_id:
        return
    ctx = ActivityContext(
        workflow_id=options.workflow_id,
        run_id=options.run_id,
        workflow_type=options.workflow_type,
        task_queue=options.task_queue,
        activity_id=activity_id,
        activity_type=llm_type,
        agent_name=options.agent_name,
        session_id=options.session_id,
    )
    options.register_trace(trace_id, ctx)


def unregister_llm_trace(options: OpenBoxLangChainCoreCallbackOptions) -> None:
    """Unregister the current OTel trace id (mandatory cleanup on completion)."""
    trace_id = raw_trace_id(otel_trace.get_current_span())
    if not trace_id:
        return
    options.unregister_trace(trace_id)
