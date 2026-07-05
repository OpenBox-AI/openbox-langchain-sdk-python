"""Mutation methods for ``ActivityBridge`` — split out to keep
``activity_bridge.py`` under 200 lines. Mixed into ``ActivityBridge``, not
usable standalone (relies on ``self._lock``/``self._workflows``).
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

from openbox_langchain.activity_bridge_records import ActivityRecord, WorkflowRecords

if TYPE_CHECKING:
    from openbox_core.contracts.results import EvaluationResult

    from openbox_langchain.activity_bridge import EventType

__all__ = ["ActivityBridgeMutationsMixin"]


class ActivityBridgeMutationsMixin:
    """Sent-flag marking and verdict-stashing mutations, lock-guarded."""

    _lock: threading.Lock
    _workflows: dict[str, WorkflowRecords]

    def mark_sent(self, workflow_id: str, activity_id: str, event_type: EventType) -> None:
        """Mark the sent-flag for ``event_type`` on this activity.

        Called IMMEDIATELY after the envelope is sent, BEFORE any enforcement
        raise (C1) — a blocked start still counts as "started sent".
        """
        with self._lock:
            wf = self._workflows.get(workflow_id)
            if wf is None:
                return
            record = wf.by_activity_id.get(activity_id)
            if record is None:
                return
            self._set_sent_flag(record, event_type)

    @staticmethod
    def _set_sent_flag(record: ActivityRecord, event_type: EventType) -> None:
        if event_type == "tool_start":
            record.tool_started_sent = True
        elif event_type == "tool_complete":
            record.tool_completed_sent = True
        elif event_type == "llm_start":
            record.llm_started_sent = True
        elif event_type == "llm_complete":
            record.llm_completed_sent = True

    def stash_start_result(
        self, workflow_id: str, activity_id: str, result: EvaluationResult
    ) -> None:
        """Stash the start verdict (enables evaluate-once, C2)."""
        with self._lock:
            wf = self._workflows.get(workflow_id)
            if wf is None:
                return
            record = wf.by_activity_id.get(activity_id)
            if record is None:
                return
            record.start_result = result

    def stash_completion_result(
        self, workflow_id: str, activity_id: str, result: EvaluationResult
    ) -> None:
        """Stash the completion verdict for the integrating layer to read (C4)."""
        with self._lock:
            wf = self._workflows.get(workflow_id)
            if wf is None:
                return
            record = wf.by_activity_id.get(activity_id)
            if record is None:
                return
            record.completion_result = result
