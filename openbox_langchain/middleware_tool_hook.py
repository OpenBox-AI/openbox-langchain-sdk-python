"""Tool governance hook for OpenBoxLangChainMiddleware.

Separated from middleware_hook_handlers.py to stay under 200 lines per file.
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import TYPE_CHECKING, Any

from openbox_langgraph.errors import GovernanceBlockedError, GovernanceHaltError
from openbox_langgraph.types import LangChainGovernanceEvent, safe_serialize
from openbox_langgraph.verdict_handler import enforce_verdict

from openbox_langchain.middleware_hooks import (
    _base_event_fields,
    _evaluate,
    _extract_governance_blocked,
    _poll_approval_or_halt,
    _run_with_otel_context,
)

_logger = logging.getLogger("openbox_langchain")

if TYPE_CHECKING:
    from openbox_langchain.middleware import OpenBoxLangChainMiddleware


async def handle_wrap_tool_call(
    mw: OpenBoxLangChainMiddleware, request: Any, handler: Any,
) -> Any:
    """Tool governance: ToolStarted → Tool (OTel spans) → ToolCompleted."""
    tool_name = request.tool_call["name"]
    tool_args = request.tool_call.get("args", {})

    # Skip governance for excluded tools
    if tool_name in mw._config.skip_tool_types:
        return await handler(request)

    activity_id = str(uuid.uuid4())
    tool_type = mw._config.tool_type_map.get(tool_name)
    base = _base_event_fields(mw)

    # Register SpanProcessor context for Layer 2 hooks
    if mw._span_processor:
        mw._span_processor.set_activity_context(mw._workflow_id, activity_id, {
            **base, "event_type": "ActivityStarted",
            "activity_id": activity_id, "activity_type": tool_name,
        })

    # ToolStarted + verdict enforcement
    if mw._config.send_tool_start_event:
        gov = LangChainGovernanceEvent(
            **base, event_type="ToolStarted", activity_id=activity_id,
            activity_type=tool_name, activity_input=[safe_serialize(tool_args)],
            tool_name=tool_name, tool_type=tool_type,
        )
        response = await _evaluate(mw, gov)
        if response is not None:
            result = enforce_verdict(response, "tool_start")
            if result.requires_hitl:
                try:
                    await _poll_approval_or_halt(mw, activity_id, tool_name)
                except GovernanceHaltError:
                    if mw._span_processor:
                        mw._span_processor.clear_activity_context(
                            mw._workflow_id, activity_id
                        )
                    raise

    # Execute tool with OTel span + HITL retry loop
    start = time.monotonic()
    while True:
        try:
            tool_result = await _run_with_otel_context(
                mw, f"tool.{tool_name}", activity_id, handler, request,
            )
            break
        except GovernanceBlockedError as hook_err:
            if hook_err.verdict != "require_approval":
                duration_ms = (time.monotonic() - start) * 1000
                await _send_tool_failed(
                    mw, activity_id, tool_name, tool_type, hook_err, duration_ms,
                )
                raise
            await _poll_approval_or_halt(mw, activity_id, tool_name)
        except Exception as exc:
            hook_err = _extract_governance_blocked(exc)
            if hook_err is not None and hook_err.verdict == "require_approval":
                await _poll_approval_or_halt(mw, activity_id, tool_name)
            else:
                duration_ms = (time.monotonic() - start) * 1000
                await _send_tool_failed(mw, activity_id, tool_name, tool_type, exc, duration_ms)
                raise
    duration_ms = (time.monotonic() - start) * 1000

    # Clear SpanProcessor context
    if mw._span_processor:
        mw._span_processor.clear_activity_context(mw._workflow_id, activity_id)

    # ToolCompleted + verdict enforcement
    if mw._config.send_tool_end_event:
        try:
            serialized_output = (
                safe_serialize({"result": tool_result})
                if isinstance(tool_result, str)
                else safe_serialize(tool_result)
            )
        except Exception:
            serialized_output = {"result": str(tool_result)}
        completed = LangChainGovernanceEvent(
            **_base_event_fields(mw), event_type="ToolCompleted",
            activity_id=f"{activity_id}-c", activity_type=tool_name,
            activity_output=serialized_output, tool_name=tool_name,
            tool_type=tool_type, status="completed", duration_ms=duration_ms,
        )
        resp = await _evaluate(mw, completed)
        if resp is not None:
            result = enforce_verdict(resp, "tool_end")
            if result.requires_hitl:
                await _poll_approval_or_halt(mw, f"{activity_id}-c", tool_name)

    return tool_result


async def _send_tool_failed(
    mw: OpenBoxLangChainMiddleware,
    activity_id: str, tool_name: str, tool_type: str | None,
    error: Exception, duration_ms: float,
) -> None:
    """Send ToolCompleted with failed status."""
    if mw._span_processor:
        mw._span_processor.clear_activity_context(mw._workflow_id, activity_id)
    if mw._config.send_tool_end_event:
        failed_event = LangChainGovernanceEvent(
            **_base_event_fields(mw), event_type="ToolCompleted",
            activity_id=f"{activity_id}-c", activity_type=tool_name,
            activity_output=safe_serialize({"error": str(error)}),
            tool_name=tool_name, tool_type=tool_type,
            status="failed", duration_ms=duration_ms,
        )
        await _evaluate(mw, failed_event)
