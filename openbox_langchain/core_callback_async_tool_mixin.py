"""Async tool-lifecycle methods for ``OpenBoxLangChainCoreAsyncCallbackHandler``.

Split out of ``core_callback_async.py`` to stay under 200 lines per file. Mixed
into the handler class — not usable standalone (relies on ``self._options``).
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

from openbox_langchain.core_callback_options import OpenBoxLangChainCoreCallbackOptions
from openbox_langchain.core_callback_tool_complete import send_tool_completed
from openbox_langchain.core_callback_tool_start import (
    build_tool_started_envelope,
    build_tool_started_input,
    enforce_tool_start_async,
)

__all__ = ["AsyncToolLifecycleMixin"]


class AsyncToolLifecycleMixin:
    """Tool lifecycle (start/end/error) for the async core callback handler."""

    _options: OpenBoxLangChainCoreCallbackOptions

    async def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        inputs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        options = self._options
        activity_id = str(run_id)
        tool_name = serialized.get("name") or "unknown_tool"
        record = options.bridge.get(options.workflow_id, activity_id)
        if record is None and not options.record_less_ok:
            return

        if record is not None and record.start_result is not None:
            # Evaluate-once (C2): a cross-dispatched sibling handler already
            # evaluated and stashed the verdict — reuse it, do not re-call the gate.
            result = record.start_result
        else:
            tool_type = (
                options.tool_type_resolver(tool_name) if options.tool_type_resolver else None
            )
            tool_call_id = kwargs.get("tool_call_id")
            options.bridge.prepare_tool(
                options.workflow_id,
                activity_id,
                tool_name=tool_name,
                tool_type=tool_type,
                tool_call_id=tool_call_id,
                langgraph_node=(metadata or {}).get("langgraph_node"),
                langgraph_step=(metadata or {}).get("langgraph_step"),
            )
            # send_tool_start_event=False disables tool-start ENFORCEMENT here,
            # not just telemetry: with no ActivityStarted sent, the start is
            # unowned (sent-flag stays false) and governance of the start falls
            # to the integrating consumer (the LangGraph handler enforces it in
            # that split config). Pure-LangChain callers that set this False opt
            # out of start-gating entirely.
            if not options.send_tool_start_event:
                return
            activity_input = build_tool_started_input(
                inputs if inputs is not None else input_str, tool_type=tool_type
            )
            envelope = build_tool_started_envelope(options, activity_id, tool_name, activity_input)
            result = await options.runtime.gate.aevaluate(envelope)
            options.bridge.mark_sent(options.workflow_id, activity_id, "tool_start")
            options.bridge.stash_start_result(options.workflow_id, activity_id, result)

        await enforce_tool_start_async(options, activity_id, tool_name, result)

    async def on_tool_end(
        self,
        output: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        options = self._options
        activity_id = str(run_id)
        if options.bridge.is_callback_owned(options.workflow_id, activity_id, "tool_complete"):
            return
        record = options.bridge.get(options.workflow_id, activity_id)
        if record is None and not options.record_less_ok:
            return
        tool_name = record.tool_name if record and record.tool_name else "unknown_tool"
        if not options.send_tool_end_event:
            return
        await send_tool_completed(options, activity_id, tool_name, result=output)

    async def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        options = self._options
        activity_id = str(run_id)
        # Guarded by tool_completed_sent (invariant #7 applied to tools too):
        # a `return_exceptions=True`-captured end that already sent must not
        # double-send a failed row.
        if options.bridge.is_callback_owned(options.workflow_id, activity_id, "tool_complete"):
            return
        record = options.bridge.get(options.workflow_id, activity_id)
        if record is None and not options.record_less_ok:
            return
        tool_name = record.tool_name if record and record.tool_name else "unknown_tool"
        if not options.send_tool_end_event:
            return
        await send_tool_completed(options, activity_id, tool_name, error=str(error))
