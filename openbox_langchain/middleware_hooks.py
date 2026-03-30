"""Hook implementations for OpenBoxLangChainMiddleware.

Each function implements one middleware hook, mapping to governance events:
- handle_before_agent  → SignalReceived + WorkflowStarted + pre-screen LLMStarted
- handle_after_agent   → WorkflowCompleted + cleanup
- handle_wrap_model_call → LLMStarted (PII redaction) → Model → LLMCompleted
- handle_wrap_tool_call  → ToolStarted → Tool (OTel spans) → ToolCompleted
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from openbox_langgraph.errors import (
    ApprovalExpiredError,
    ApprovalRejectedError,
    GovernanceBlockedError,
    GovernanceHaltError,
)
from openbox_langgraph.hitl import HITLPollParams, poll_until_decision
from openbox_langgraph.types import (
    rfc3339_now,
)
from opentelemetry import context as otel_context
from opentelemetry import trace as otel_trace

_tracer = otel_trace.get_tracer("openbox-langchain")
_logger = logging.getLogger("openbox_langchain")

if TYPE_CHECKING:
    from openbox_langchain.middleware import OpenBoxLangChainMiddleware


# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════


def _base_event_fields(mw: OpenBoxLangChainMiddleware) -> dict[str, Any]:
    """Common fields for all governance events."""
    return {
        "source": "workflow-telemetry",
        "workflow_id": mw._workflow_id,
        "run_id": mw._run_id,
        "workflow_type": mw._workflow_type,
        "task_queue": mw._config.task_queue,
        "timestamp": rfc3339_now(),
        "session_id": mw._config.session_id,
    }


async def _evaluate(mw: OpenBoxLangChainMiddleware, event: Any) -> Any:
    """Send governance event — sync httpx in sync mode, async otherwise."""
    if mw._sync_mode:
        return mw._client.evaluate_event_sync(event)
    return await mw._client.evaluate_event(event)


def _extract_governance_blocked(exc: Exception) -> GovernanceBlockedError | None:
    """Unwrap GovernanceBlockedError from LLM SDK exception chains."""
    cause: BaseException | None = exc
    seen: set[int] = set()
    while cause is not None:
        if id(cause) in seen:
            break
        seen.add(id(cause))
        if isinstance(cause, GovernanceBlockedError):
            return cause
        cause = getattr(cause, "__cause__", None) or getattr(cause, "__context__", None)
    return None


async def _poll_approval_or_halt(
    mw: OpenBoxLangChainMiddleware, activity_id: str, activity_type: str,
) -> None:
    """Poll for HITL approval. On rejection/expiry, raises GovernanceHaltError."""
    if mw._span_processor:
        mw._span_processor.clear_activity_abort(mw._workflow_id, activity_id)
    try:
        await poll_until_decision(
            mw._client,
            HITLPollParams(
                workflow_id=mw._workflow_id, run_id=mw._run_id,
                activity_id=activity_id, activity_type=activity_type,
            ),
            mw._config.hitl,
        )
    except (ApprovalRejectedError, ApprovalExpiredError) as e:
        if mw._span_processor:
            mw._span_processor.clear_activity_context(mw._workflow_id, activity_id)
        raise GovernanceHaltError(str(e)) from e


def _extract_last_user_message(messages: list[Any]) -> str | None:
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


def _extract_prompt_from_messages(messages: Any) -> str:
    """Extract human/user message text from a messages list."""
    if not isinstance(messages, (list, tuple)):
        return ""
    parts: list[str] = []
    for msg in messages:
        if isinstance(msg, (list, tuple)):
            for inner in msg:
                _append_human_content(inner, parts)
        else:
            _append_human_content(msg, parts)
    return "\n".join(parts)


def _append_human_content(msg: Any, parts: list[str]) -> None:
    """Append human message content to parts list."""
    role = None
    content = None
    if hasattr(msg, "type"):
        role = msg.type
        content = msg.content
    elif isinstance(msg, dict):
        role = msg.get("role") or msg.get("type", "")
        content = msg.get("content", "")
    if role not in ("human", "user", "generic"):
        return
    if isinstance(content, str):
        parts.append(content)
    elif isinstance(content, list):
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append(part.get("text", ""))


def _apply_pii_redaction(messages: list[Any], redacted_input: Any) -> None:
    """Apply PII redaction to messages in-place from guardrails response."""
    redacted_text = None
    if isinstance(redacted_input, list) and redacted_input:
        first = redacted_input[0]
        if isinstance(first, dict):
            redacted_text = first.get("prompt")
        elif isinstance(first, str):
            redacted_text = first
    elif isinstance(redacted_input, str):
        redacted_text = redacted_input

    if not redacted_text:
        return

    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        if hasattr(msg, "type") and msg.type in ("human", "generic"):
            msg.content = redacted_text
            break
        elif isinstance(msg, dict) and msg.get("role") in ("user", "human"):
            msg["content"] = redacted_text
            break


def _extract_response_metadata(response: Any) -> dict[str, Any]:
    """Extract tokens, model name, completion from model response."""
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
            p.get("text", "") for p in content
            if isinstance(p, dict) and p.get("type") == "text"
        ]
        result["completion"] = " ".join(text_parts) if text_parts else None

    result["has_tool_calls"] = bool(getattr(ai_msg, "tool_calls", None))
    return result


async def _run_with_otel_context(
    mw: OpenBoxLangChainMiddleware, span_name: str, activity_id: str,
    handler: Any, request: Any,
) -> Any:
    """Execute handler inside an OTel span for trace context propagation."""
    parent_ctx = otel_context.get_current()
    span = _tracer.start_span(span_name, context=parent_ctx, kind=otel_trace.SpanKind.INTERNAL)
    token = otel_context.attach(otel_trace.set_span_in_context(span, parent_ctx))

    trace_id = span.get_span_context().trace_id
    if mw._span_processor and trace_id:
        mw._span_processor.register_trace(trace_id, mw._workflow_id, activity_id)

    try:
        return await handler(request)
    finally:
        span.end()
        try:
            otel_context.detach(token)
        except Exception:
            pass
