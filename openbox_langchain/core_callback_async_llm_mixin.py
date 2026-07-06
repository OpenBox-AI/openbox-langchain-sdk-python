"""Async LLM-lifecycle methods for ``OpenBoxLangChainCoreAsyncCallbackHandler``.

Split out of ``core_callback_async.py`` to stay under 200 lines per file. Mixed
into the handler class — not usable standalone (relies on ``self._options``).

Per Phase 0's measured propagation matrix, chat-model raises PROPAGATE — but
enforcement wiring for LLM-start stays deferred to the integrating layer this
phase (telemetry + redaction + trace-registration only here).
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

from openbox_langchain.core_callback_llm_logic import (
    LLM_ACTIVITY_TYPE,
    apply_llm_redaction,
    build_llm_started_envelope,
    finish_llm_trace,
    send_llm_completed,
    start_llm_trace,
)
from openbox_langchain.core_callback_options import OpenBoxLangChainCoreCallbackOptions

__all__ = ["AsyncLLMLifecycleMixin"]


class AsyncLLMLifecycleMixin:
    """LLM lifecycle (chat-model start/end/error) for the async core callback handler."""

    _options: OpenBoxLangChainCoreCallbackOptions

    async def on_chat_model_start(
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

        # Evaluate-once (C2/Phase 0): the async AND sync handlers are BOTH
        # installed and BOTH cross-dispatched for every event (Phase 0's
        # measured matrix), so this method runs TWICE for the SAME
        # `run_id`/`event_run_id`. Resolve the alias FIRST — before touching
        # `options.pre_screen_response` at all — so the SECOND dispatch finds
        # the record the FIRST dispatch already aliased (whether that was the
        # pre-screen `-pre` row or a real-evaluate row) and reuses its
        # activity_id + stashed verdict, rather than re-deciding the
        # pre-screen-vs-real-evaluate branch from a `pre_screen_response`
        # the first dispatch may have already cleared.
        existing = options.bridge.get_by_event_run_id(options.workflow_id, event_run_id)
        if existing is not None and existing.start_result is not None:
            activity_id = existing.activity_id
            result = existing.start_result
            # Still consume the pre-screen fields if this run_id is what
            # they were meant for — a later LLM call on this handler must
            # not accidentally reuse a stale pre-screen entry.
            if options.pre_screen_activity_id == activity_id:
                options.pre_screen_response = None
                options.pre_screen_activity_id = None
        elif options.pre_screen_response is not None and options.pre_screen_activity_id is not None:
            # Reuse the upstream pre-screen verdict for call 1 (H11): the
            # activity_id may diverge from the callback's own run_id, so the
            # alias index is what lets completion resolve correctly later.
            activity_id = options.pre_screen_activity_id
            result = options.pre_screen_response
            options.bridge.prepare_llm(options.workflow_id, activity_id, event_run_id=event_run_id)
            options.bridge.mark_sent(options.workflow_id, activity_id, "llm_start")
            options.bridge.stash_start_result(options.workflow_id, activity_id, result)
            # Consumed once — subsequent LLM calls on this handler evaluate for real.
            options.pre_screen_response = None
            options.pre_screen_activity_id = None
        else:
            activity_id = event_run_id
            options.bridge.prepare_llm(options.workflow_id, activity_id, event_run_id=event_run_id)
            if not options.send_llm_start_event:
                return
            envelope = build_llm_started_envelope(options, activity_id, llm_type, messages)
            result = await options.runtime.gate.aevaluate(envelope)
            options.bridge.mark_sent(options.workflow_id, activity_id, "llm_start")
            options.bridge.stash_start_result(options.workflow_id, activity_id, result)

        apply_llm_redaction(result, messages)
        if event_run_id not in options.llm_trace_handles:
            options.llm_trace_handles[event_run_id] = start_llm_trace(
                options, activity_id, llm_type
            )

    async def on_llm_end(
        self,
        response: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        await self._finish_llm(run_id, response=response, error=None)

    async def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        # Guarded by llm_completed_sent via _finish_llm's ownership check: a
        # `return_exceptions=True`-captured end that already sent does not
        # double-send a failed row.
        await self._finish_llm(run_id, response=None, error=str(error))

    async def _finish_llm(self, run_id: UUID, *, response: Any, error: str | None) -> None:
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
            await send_llm_completed(
                options, activity_id, LLM_ACTIVITY_TYPE, result=response, error=error
            )
        finally:
            finish_llm_trace(options, trace_handle)
