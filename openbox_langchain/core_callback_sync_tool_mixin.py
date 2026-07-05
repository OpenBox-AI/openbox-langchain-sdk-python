"""Sync tool-lifecycle methods for ``OpenBoxLangChainCoreSyncCallbackHandler``.

This is the handler that makes the sync-only-tool corner fail-closed PRE-body
(Phase 0-measured: the ASYNC handler's raise is SWALLOWED there by
``_run_coros``; only the SYNC handler's raise propagates through
``BaseTool.run``'s sync callback manager). Split out of
``core_callback_sync.py`` to stay under 200 lines per file.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

from openbox_langchain.core_callback_options import OpenBoxLangChainCoreCallbackOptions
from openbox_langchain.core_callback_tool_complete import send_tool_completed_sync
from openbox_langchain.core_callback_tool_start import (
    build_tool_started_envelope,
    build_tool_started_input,
    enforce_tool_start_sync,
)

__all__ = ["SyncToolLifecycleMixin"]


class SyncToolLifecycleMixin:
    """Tool lifecycle (start/end/error) for the sync core callback handler."""

    _options: OpenBoxLangChainCoreCallbackOptions

    def on_tool_start(
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
            # Evaluate-once (C2): reuse the async handler's stashed verdict —
            # this is the corner where the async handler's raise was
            # SWALLOWED (`_run_coros`) but it already stashed BLOCK/HALT; this
            # sync handler re-raises it inline, PRE-body.
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
            # not just telemetry (see the async mixin note): start unowned →
            # the integrating consumer governs the start in that split config.
            if not options.send_tool_start_event:
                return
            activity_input = build_tool_started_input(
                inputs if inputs is not None else input_str, tool_type=tool_type
            )
            envelope = build_tool_started_envelope(options, activity_id, tool_name, activity_input)
            result = options.runtime.gate.evaluate(envelope)
            options.bridge.mark_sent(options.workflow_id, activity_id, "tool_start")
            options.bridge.stash_start_result(options.workflow_id, activity_id, result)

        enforce_tool_start_sync(options, activity_id, tool_name, result)

    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
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
        send_tool_completed_sync(options, activity_id, tool_name, result=output)

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
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
        send_tool_completed_sync(options, activity_id, tool_name, error=str(error))
