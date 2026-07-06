"""ActivityStarted/ActivityCompleted envelope builders — split out of
``lifecycle_events.py`` to stay under 200 lines. Re-exported there; not a
separate public surface.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from openbox_core.contracts.events import EventEnvelope, activity_completed, activity_started

__all__ = ["build_activity_completed", "build_activity_started"]


def build_activity_started(
    *,
    workflow_id: str,
    run_id: str,
    workflow_type: str,
    activity_id: str,
    activity_type: str,
    task_queue: str | None = None,
    activity_input: Any = None,
    attempt: int | None = None,
    multi_agent_session_id: str | None = None,
    session_id: str | None = None,
    agent_name: str | None = None,
    extra: Mapping[str, Any] | None = None,
) -> EventEnvelope:
    """Build an ``ActivityStarted`` envelope for a tool/LLM lifecycle event.

    ``session_id``/``agent_name`` are not first-class fields on the base
    factory — they travel via ``extra`` (merged into the payload) so the wire
    body carries them without widening the base SDK's contract surface.
    """
    merged_extra: dict[str, Any] = dict(extra) if extra else {}
    if session_id is not None:
        merged_extra.setdefault("session_id", session_id)
    if agent_name is not None:
        merged_extra.setdefault("agent_name", agent_name)
    return activity_started(
        workflow_id=workflow_id,
        run_id=run_id,
        workflow_type=workflow_type,
        activity_id=activity_id,
        activity_type=activity_type,
        task_queue=task_queue,
        activity_input=activity_input,
        attempt=attempt,
        multi_agent_session_id=multi_agent_session_id,
        extra=merged_extra or None,
    )


def build_activity_completed(
    *,
    workflow_id: str,
    run_id: str,
    workflow_type: str,
    activity_id: str,
    activity_type: str,
    task_queue: str | None = None,
    result: Any = None,
    error: str | None = None,
    attempt: int | None = None,
    multi_agent_session_id: str | None = None,
    session_id: str | None = None,
    agent_name: str | None = None,
    extra: Mapping[str, Any] | None = None,
) -> EventEnvelope:
    """Build an ``ActivityCompleted`` envelope, same ``session_id``/``agent_name``
    passthrough rule as :func:`build_activity_started`."""
    merged_extra: dict[str, Any] = dict(extra) if extra else {}
    if session_id is not None:
        merged_extra.setdefault("session_id", session_id)
    if agent_name is not None:
        merged_extra.setdefault("agent_name", agent_name)
    return activity_completed(
        workflow_id=workflow_id,
        run_id=run_id,
        workflow_type=workflow_type,
        activity_id=activity_id,
        activity_type=activity_type,
        task_queue=task_queue,
        result=result,
        error=error,
        attempt=attempt,
        multi_agent_session_id=multi_agent_session_id,
        extra=merged_extra or None,
    )
