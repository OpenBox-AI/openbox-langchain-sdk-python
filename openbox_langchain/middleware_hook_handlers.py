"""Governance hook handler functions for OpenBoxLangChainMiddleware.

before_agent / after_agent / wrap_model_call / wrap_tool_call implementations.
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import TYPE_CHECKING, Any

from openbox_langgraph.types import LangChainGovernanceEvent, safe_serialize
from openbox_langgraph.verdict_handler import enforce_verdict

from openbox_langchain.middleware_hooks import (
    _apply_pii_redaction,
    _base_event_fields,
    _evaluate,
    _extract_governance_blocked,
    _extract_last_user_message,
    _extract_prompt_from_messages,
    _extract_response_metadata,
    _poll_approval_or_halt,
    _run_with_otel_context,
)

_logger = logging.getLogger("openbox_langchain")

if TYPE_CHECKING:
    from openbox_langchain.middleware import OpenBoxLangChainMiddleware


# ═══════════════════════════════════════════════════════════════════
# Hook: before_agent → WorkflowStarted + pre-screen
# ═══════════════════════════════════════════════════════════════════


async def handle_before_agent(
    mw: OpenBoxLangChainMiddleware, state: Any, runtime: Any,
) -> dict[str, Any] | None:
    """Session setup: SignalReceived + WorkflowStarted + pre-screen guardrails."""
    # Generate unique session IDs per invocation
    config = getattr(runtime, "config", None) or {}
    configurable = config.get("configurable", {}) if isinstance(config, dict) else {}
    thread_id = configurable.get("thread_id", "langchain")
    _turn = uuid.uuid4().hex
    mw._workflow_id = f"{thread_id}-{_turn[:8]}"
    mw._run_id = f"{thread_id}-run-{_turn[8:16]}"
    mw._first_llm_call = True
    mw._pre_screen_response = None

    base = _base_event_fields(mw)
    messages = (
        state.get("messages", []) if isinstance(state, dict)
        else getattr(state, "messages", [])
    )

    # SignalReceived — user prompt as trigger
    user_prompt = _extract_last_user_message(messages)
    if user_prompt:
        sig_event = LangChainGovernanceEvent(
            **base, event_type="SignalReceived",
            activity_id=f"{mw._run_id}-sig", activity_type="user_prompt",
            signal_name="user_prompt", signal_args=[user_prompt],
        )
        await _evaluate(mw, sig_event)

    # WorkflowStarted
    if mw._config.send_chain_start_event:
        wf_event = LangChainGovernanceEvent(
            **base, event_type="WorkflowStarted",
            activity_id=f"{mw._run_id}-wf",
            activity_type=mw._workflow_type,
            activity_input=[safe_serialize(state)],
        )
        await _evaluate(mw, wf_event)

    # Pre-screen LLMStarted (guardrails on user prompt)
    if mw._config.send_llm_start_event and user_prompt and user_prompt.strip():
        gov = LangChainGovernanceEvent(
            **base, event_type="LLMStarted",
            activity_id=f"{mw._run_id}-pre", activity_type="llm_call",
            activity_input=[{"prompt": user_prompt}], prompt=user_prompt,
        )
        response = await _evaluate(mw, gov)

        if response is not None:
            enforcement_error: Exception | None = None
            try:
                result = enforce_verdict(response, "llm_start")
            except Exception as exc:
                enforcement_error = exc

            if enforcement_error is not None and mw._config.send_chain_end_event:
                wf_end = LangChainGovernanceEvent(
                    **_base_event_fields(mw), event_type="WorkflowCompleted",
                    activity_id=f"{mw._run_id}-wf",
                    activity_type=mw._workflow_type,
                    status="failed", error=str(enforcement_error),
                )
                await _evaluate(mw, wf_end)
                raise enforcement_error

            if result and result.requires_hitl:
                await _poll_approval_or_halt(mw, f"{mw._run_id}-pre", "llm_call")

        mw._pre_screen_response = response
    return None


# ═══════════════════════════════════════════════════════════════════
# Hook: after_agent → WorkflowCompleted
# ═══════════════════════════════════════════════════════════════════


async def handle_after_agent(
    mw: OpenBoxLangChainMiddleware, state: Any, runtime: Any,
) -> dict[str, Any] | None:
    """Session close: WorkflowCompleted + cleanup."""
    if mw._config.send_chain_end_event:
        messages = (
            state.get("messages", []) if isinstance(state, dict)
            else getattr(state, "messages", [])
        )
        last_content = None
        if messages:
            last_msg = messages[-1]
            last_content = (
                getattr(last_msg, "content", None) if hasattr(last_msg, "content")
                else (last_msg.get("content") if isinstance(last_msg, dict) else None)
            )

        wf_event = LangChainGovernanceEvent(
            **_base_event_fields(mw), event_type="WorkflowCompleted",
            activity_id=f"{mw._run_id}-wf",
            activity_type=mw._workflow_type,
            workflow_output=safe_serialize({"result": last_content}),
            status="completed",
        )
        await _evaluate(mw, wf_event)

    if mw._span_processor:
        mw._span_processor.unregister_workflow(mw._workflow_id)
    return None


# ═══════════════════════════════════════════════════════════════════
# Hook: wrap_model_call → LLMStarted/Completed
# ═══════════════════════════════════════════════════════════════════


async def handle_wrap_model_call(
    mw: OpenBoxLangChainMiddleware, request: Any, handler: Any,
) -> Any:
    """LLM governance: LLMStarted → PII redaction → Model → LLMCompleted."""
    prompt_text = _extract_prompt_from_messages(request.messages)
    if not prompt_text.strip():
        return await handler(request)

    base = _base_event_fields(mw)
    activity_id = str(uuid.uuid4())

    # Reuse pre-screen response for first LLM call
    if mw._first_llm_call and mw._pre_screen_response is not None:
        response = mw._pre_screen_response
        mw._pre_screen_response = None
        mw._first_llm_call = False
        activity_id = f"{mw._run_id}-pre"
    else:
        mw._first_llm_call = False
        if mw._config.send_llm_start_event:
            gov = LangChainGovernanceEvent(
                **base, event_type="LLMStarted", activity_id=activity_id,
                activity_type="llm_call", activity_input=[{"prompt": prompt_text}],
                prompt=prompt_text,
            )
            response = await _evaluate(mw, gov)
        else:
            response = None

    # PII redaction
    if response and response.guardrails_result:
        gr = response.guardrails_result
        if gr.input_type == "activity_input" and gr.redacted_input is not None:
            _apply_pii_redaction(request.messages, gr.redacted_input)

    # Register SpanProcessor context
    if mw._span_processor:
        mw._span_processor.set_activity_context(mw._workflow_id, activity_id, {
            **base, "event_type": "ActivityStarted",
            "activity_id": activity_id, "activity_type": "llm_call",
        })

    # Execute model with OTel span + HITL retry loop
    start = time.monotonic()
    while True:
        try:
            model_response = await _run_with_otel_context(
                mw, "llm.call", activity_id, handler, request,
            )
            break
        except Exception as exc:
            hook_err = _extract_governance_blocked(exc)
            if hook_err is not None and hook_err.verdict == "require_approval":
                await _poll_approval_or_halt(mw, activity_id, "llm_call")
            else:
                raise
    duration_ms = (time.monotonic() - start) * 1000

    # LLMCompleted
    if mw._config.send_llm_end_event:
        meta = _extract_response_metadata(model_response)
        completed = LangChainGovernanceEvent(
            **_base_event_fields(mw), event_type="LLMCompleted",
            activity_id=f"{activity_id}-c", activity_type="llm_call",
            status="completed", duration_ms=duration_ms,
            llm_model=meta.get("llm_model"),
            input_tokens=meta.get("input_tokens"),
            output_tokens=meta.get("output_tokens"),
            total_tokens=meta.get("total_tokens"),
            has_tool_calls=meta.get("has_tool_calls"),
            completion=meta.get("completion"),
        )
        resp = await _evaluate(mw, completed)
        if resp is not None:
            enforce_verdict(resp, "llm_end")

    if mw._span_processor:
        mw._span_processor.clear_activity_context(mw._workflow_id, activity_id)
    return model_response
