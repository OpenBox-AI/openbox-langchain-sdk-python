"""Tool-start lifecycle logic shared by the async and sync core callback handlers.

Implements the evaluate-once / enforce-from-stash contract (C2 sync corner):
the caller decides whether to reuse a stashed ``start_result`` or call the
gate once, then calls the appropriate ``enforce_tool_start_*`` here so
enforcement replays from the stash in every handler that reaches this point.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from openbox_core.context import current_activity_context
from openbox_core.contracts.context import ActivityContext
from openbox_core.errors import ApprovalRejectedError

from openbox_langchain.core_callback_tool_complete import (
    close_orphan_start,
    close_orphan_start_sync,
)
from openbox_langchain.lifecycle_events import build_activity_started, enrich_activity_input

if TYPE_CHECKING:
    from openbox_core.contracts.results import EvaluationResult

    from openbox_langchain.activity_bridge import ActivityRecord
    from openbox_langchain.core_callback_options import OpenBoxLangChainCoreCallbackOptions

__all__ = [
    "build_tool_started_envelope",
    "build_tool_started_input",
    "enforce_tool_start_async",
    "enforce_tool_start_sync",
    "resolve_tool_context",
]


def resolve_tool_context(
    options: OpenBoxLangChainCoreCallbackOptions, record: ActivityRecord | None
) -> ActivityContext | None:
    """Resolve the ActivityContext for a tool event.

    With ``run_inline=True`` the callback runs inside the ToolNode's
    ``activity_scope``, so ``current_activity_context()`` already carries the
    right context in the LangGraph-embedded case (H12 — the bridge does not
    re-carry it). Pure-LangChain callers with no bound context get None; the
    caller builds a minimal context from local fields instead.
    """
    return current_activity_context()


def build_tool_started_input(
    tool_args: Any,
    *,
    tool_type: str | None,
    subagent_name: str | None = None,
) -> list[Any]:
    """Build the ``activity_input`` list for a ToolStarted event, applying the
    ``__openbox`` enrichment sentinel (wire parity)."""
    base_input = [tool_args] if tool_args is not None else []
    enriched = enrich_activity_input(base_input, tool_type=tool_type, subagent_name=subagent_name)
    return enriched if enriched is not None else base_input


def build_tool_started_envelope(
    options: OpenBoxLangChainCoreCallbackOptions,
    activity_id: str,
    tool_name: str,
    activity_input: list[Any],
) -> Any:
    """Build the ActivityStarted envelope for a tool call (does not send)."""
    return build_activity_started(
        workflow_id=options.workflow_id,
        run_id=options.run_id,
        workflow_type=options.workflow_type,
        activity_id=activity_id,
        activity_type=tool_name,
        task_queue=options.task_queue,
        activity_input=activity_input,
        session_id=options.session_id,
        agent_name=options.agent_name,
    )


async def enforce_tool_start_async(
    options: OpenBoxLangChainCoreCallbackOptions,
    activity_id: str,
    tool_name: str,
    result: EvaluationResult,
) -> None:
    """Enforce a (reused or fresh) start verdict on the ASYNC path.

    Stop-shaped (BLOCK/HALT): close the orphan row then raise via
    ``adapter.raise_lifecycle_blocked`` (C6). REQUIRE_APPROVAL: drive
    ``adapter.handle_approval`` — approved returns normally, rejected/expired
    raises the adapter's native error.
    """
    if result.verdict.should_stop():
        await close_orphan_start(options, activity_id, tool_name, result)
        options.runtime.adapter.raise_lifecycle_blocked(result)
        return
    if result.verdict.requires_approval():
        await options.runtime.adapter.handle_approval(result)


def enforce_tool_start_sync(
    options: OpenBoxLangChainCoreCallbackOptions,
    activity_id: str,
    tool_name: str,
    result: EvaluationResult,
) -> None:
    """Enforce a (reused or fresh) start verdict on the SYNC path.

    Stop-shaped: same orphan-close-then-raise as the async path (uses the
    sync gate). REQUIRE_APPROVAL fail-safe (M15): the sync path cannot await
    the adapter's async approval flow, so it calls
    ``adapter.handle_approval_sync`` when the adapter defines one, else raises
    ``ApprovalRejectedError`` — never silently allowed.
    """
    if result.verdict.should_stop():
        close_orphan_start_sync(options, activity_id, tool_name, result)
        options.runtime.adapter.raise_lifecycle_blocked(result)
        return
    if result.verdict.requires_approval():
        handle_approval_sync = getattr(options.runtime.adapter, "handle_approval_sync", None)
        if callable(handle_approval_sync):
            handle_approval_sync(result)
            return
        raise ApprovalRejectedError(
            "REQUIRE_APPROVAL verdict on the sync callback path but the adapter "
            "defines no handle_approval_sync — failing safe (operation not run)"
        )
