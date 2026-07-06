"""Sync LLM-lifecycle methods for ``OpenBoxLangChainCoreSyncCallbackHandler``.

Split out of ``core_callback_sync.py`` to stay under 200 lines per file. Same
telemetry+redaction+trace-registration-only scope as the async LLM mixin —
enforcement wiring for LLM-start is deferred to the integrating layer.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

from openbox_langchain.core_callback_llm_logic import (
    LLM_ACTIVITY_TYPE,
    apply_llm_redaction,
    build_llm_started_envelope,
    finish_llm_trace,
    send_llm_completed_sync,
    start_llm_trace,
)
from openbox_langchain.core_callback_options import OpenBoxLangChainCoreCallbackOptions

__all__ = ["SyncLLMLifecycleMixin"]


class SyncLLMLifecycleMixin:
    """LLM lifecycle (chat-model start/end/error) for the sync core callback handler."""

    _options: OpenBoxLangChainCoreCallbackOptions

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[Any]],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        options = self._options
        event_run_id = str(run_id)
        llm_type = serialized.get("name") or "unknown_llm"

        # Evaluate-once (C2/Phase 0) — see the async mixin's identical
        # comment: resolve the alias FIRST, before touching
        # `options.pre_screen_response`, so the SECOND cross-dispatched
        # handler (this method runs on BOTH the async and sync handler for
        # every event) reuses whatever the FIRST dispatch already aliased —
        # pre-screen row or a real-evaluate row — instead of re-deciding the
        # branch from a `pre_screen_response` the first dispatch may have
        # already cleared.
        existing = options.bridge.get_by_event_run_id(options.workflow_id, event_run_id)
        if existing is not None and existing.start_result is not None:
            activity_id = existing.activity_id
            result = existing.start_result
            if options.pre_screen_activity_id == activity_id:
                options.pre_screen_response = None
                options.pre_screen_activity_id = None
        elif options.pre_screen_response is not None and options.pre_screen_activity_id is not None:
            activity_id = options.pre_screen_activity_id
            result = options.pre_screen_response
            options.bridge.prepare_llm(options.workflow_id, activity_id, event_run_id=event_run_id)
            options.bridge.mark_sent(options.workflow_id, activity_id, "llm_start")
            options.bridge.stash_start_result(options.workflow_id, activity_id, result)
            options.pre_screen_response = None
            options.pre_screen_activity_id = None
        else:
            activity_id = event_run_id
            options.bridge.prepare_llm(options.workflow_id, activity_id, event_run_id=event_run_id)
            if not options.send_llm_start_event:
                return
            envelope = build_llm_started_envelope(options, activity_id, llm_type, messages)
            result = options.runtime.gate.evaluate(envelope)
            options.bridge.mark_sent(options.workflow_id, activity_id, "llm_start")
            options.bridge.stash_start_result(options.workflow_id, activity_id, result)

        apply_llm_redaction(result, messages)
        if event_run_id not in options.llm_trace_handles:
            options.llm_trace_handles[event_run_id] = start_llm_trace(
                options, activity_id, llm_type
            )

    def on_llm_end(
        self,
        response: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self._finish_llm(run_id, response=response, error=None)

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self._finish_llm(run_id, response=None, error=str(error))

    def _finish_llm(self, run_id: UUID, *, response: Any, error: str | None) -> None:
        options = self._options
        event_run_id = str(run_id)
        record = options.bridge.get_by_event_run_id(options.workflow_id, event_run_id)
        if record is None:
            if not options.record_less_ok:
                return
            activity_id = event_run_id
        else:
            activity_id = record.activity_id
        trace_handle = options.llm_trace_handles.pop(event_run_id, None)
        if options.bridge.is_callback_owned(options.workflow_id, activity_id, "llm_complete"):
            finish_llm_trace(options, trace_handle)
            return
        if not options.send_llm_end_event:
            finish_llm_trace(options, trace_handle)
            return
        try:
            send_llm_completed_sync(
                options, activity_id, LLM_ACTIVITY_TYPE, result=response, error=error
            )
        finally:
            finish_llm_trace(options, trace_handle)
