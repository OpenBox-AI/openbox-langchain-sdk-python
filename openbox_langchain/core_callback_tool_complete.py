"""Tool-completion lifecycle logic shared by the async and sync core callback handlers.

Covers the C6 orphan-row close (a stop-shaped start verdict must close its own
ActivityStarted with a failed ActivityCompleted before re-raising) and the C4
completion-send rule (``gate.aevaluate``/``gate.evaluate`` ONLY — never
adapter-enforcing, since enforcing here would raise-and-retry the whole graph
and re-run already-committed side effects).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from openbox_langchain.lifecycle_events import build_activity_completed

if TYPE_CHECKING:
    from openbox_core.contracts.results import EvaluationResult

    from openbox_langchain.core_callback_options import OpenBoxLangChainCoreCallbackOptions

__all__ = [
    "close_orphan_start",
    "close_orphan_start_sync",
    "send_tool_completed",
    "send_tool_completed_sync",
]


async def close_orphan_start(
    options: OpenBoxLangChainCoreCallbackOptions,
    activity_id: str,
    tool_name: str,
    result: EvaluationResult,
) -> None:
    """Send a failed ActivityCompleted (SAME id) for a stop-shaped start verdict.

    ``BaseTool.arun`` awaits ``on_tool_start`` outside its body try-block, so a
    start-raise never triggers ``on_tool_error`` and nothing else closes the
    row the callback already sent (C6). Guarded by ``tool_completed_sent`` so
    a cross-dispatched sibling handler does not double-send.
    """
    if options.bridge.is_callback_owned(options.workflow_id, activity_id, "tool_complete"):
        return
    envelope = build_activity_completed(
        workflow_id=options.workflow_id,
        run_id=options.run_id,
        workflow_type=options.workflow_type,
        activity_id=activity_id,
        activity_type=tool_name,
        task_queue=options.task_queue,
        error=result.reason or f"Governance {result.verdict.value}",
        session_id=options.session_id,
        agent_name=options.agent_name,
    )
    await options.runtime.gate.aevaluate(envelope)
    options.bridge.mark_sent(options.workflow_id, activity_id, "tool_complete")


def close_orphan_start_sync(
    options: OpenBoxLangChainCoreCallbackOptions,
    activity_id: str,
    tool_name: str,
    result: EvaluationResult,
) -> None:
    """Sync counterpart of :func:`close_orphan_start` (uses ``gate.evaluate``)."""
    if options.bridge.is_callback_owned(options.workflow_id, activity_id, "tool_complete"):
        return
    envelope = build_activity_completed(
        workflow_id=options.workflow_id,
        run_id=options.run_id,
        workflow_type=options.workflow_type,
        activity_id=activity_id,
        activity_type=tool_name,
        task_queue=options.task_queue,
        error=result.reason or f"Governance {result.verdict.value}",
        session_id=options.session_id,
        agent_name=options.agent_name,
    )
    options.runtime.gate.evaluate(envelope)
    options.bridge.mark_sent(options.workflow_id, activity_id, "tool_complete")


async def send_tool_completed(
    options: OpenBoxLangChainCoreCallbackOptions,
    activity_id: str,
    tool_name: str,
    *,
    result: Any = None,
    error: str | None = None,
) -> EvaluationResult:
    """Send ActivityCompleted (same id), stash the verdict, mark sent.

    Completion sends use ``gate.aevaluate`` ONLY (C4) — never
    ``adapter.raise_lifecycle_blocked`` / ``aevaluate_lifecycle`` — because
    enforcing here would raise-and-retry the whole graph, re-running side
    effects already committed. The integrating layer reads
    ``record.completion_result`` to decide poll-and-continue.
    """
    envelope = build_activity_completed(
        workflow_id=options.workflow_id,
        run_id=options.run_id,
        workflow_type=options.workflow_type,
        activity_id=activity_id,
        activity_type=tool_name,
        task_queue=options.task_queue,
        result=result,
        error=error,
        session_id=options.session_id,
        agent_name=options.agent_name,
    )
    verdict = await options.runtime.gate.aevaluate(envelope)
    options.bridge.stash_completion_result(options.workflow_id, activity_id, verdict)
    options.bridge.mark_sent(options.workflow_id, activity_id, "tool_complete")
    return verdict


def send_tool_completed_sync(
    options: OpenBoxLangChainCoreCallbackOptions,
    activity_id: str,
    tool_name: str,
    *,
    result: Any = None,
    error: str | None = None,
) -> EvaluationResult:
    """Sync counterpart of :func:`send_tool_completed` (uses ``gate.evaluate``).

    Same C4 rule: telemetry-only, verdict stashed for the integrating layer,
    never enforced here.
    """
    envelope = build_activity_completed(
        workflow_id=options.workflow_id,
        run_id=options.run_id,
        workflow_type=options.workflow_type,
        activity_id=activity_id,
        activity_type=tool_name,
        task_queue=options.task_queue,
        result=result,
        error=error,
        session_id=options.session_id,
        agent_name=options.agent_name,
    )
    verdict = options.runtime.gate.evaluate(envelope)
    options.bridge.stash_completion_result(options.workflow_id, activity_id, verdict)
    options.bridge.mark_sent(options.workflow_id, activity_id, "tool_complete")
    return verdict
