"""ActivityBridge — a minimal, framework-neutral ownership channel.

NOT a context store (that is ``openbox_core.context.ContextStore``) and NOT a
third source of truth for activity context. Its sole job is answering "has the
callback already sent event X for this activity?" (ownership, C1) and stashing
the verdicts the callback evaluates so cross-dispatched handlers (C2/C4/H12 —
see phase-02-activitybridge-and-core-callback.md in the langgraph SDK repo)
enforce from a single gate call instead of re-evaluating.

Deliberately excludes (H12 — trimmed from an earlier draft): a frozen
``ActivityBridgeKey`` triple (only needed for a forbidden global fallback),
an ``owner`` field (one possible value in this phase), ``activity_kind``, and
``parent_ids`` (no consumer reads them yet).

Thread-safety: tool bodies run in executor threads (sync-only tools under the
async graph path), so all mutating operations take a ``threading.Lock``.
Instances are handler/middleware-owned — never a module-level singleton.

Split across three files to stay under 200 lines each:
``activity_bridge_records.py`` (the ``ActivityRecord``/``WorkflowRecords``
dataclasses), ``activity_bridge_mutations.py`` (sent-flag + verdict-stash
mutations, mixed in below), and this module (record creation, lookup,
ownership query, cleanup — the primary surface).
"""

from __future__ import annotations

import threading
from typing import Literal

from openbox_langchain.activity_bridge_mutations import ActivityBridgeMutationsMixin
from openbox_langchain.activity_bridge_records import ActivityRecord, WorkflowRecords

__all__ = ["ActivityBridge", "ActivityRecord", "EventType"]

# Per-event-type sent-flag keys. Ownership is keyed on these, NEVER on record
# existence — a record may be prepared without the callback ever running
# (e.g. a subagent gate seam in a later phase), and that path must fall
# through to the consumer rather than report "owned".
EventType = Literal["tool_start", "tool_complete", "llm_start", "llm_complete"]


class ActivityBridge(ActivityBridgeMutationsMixin):
    """Ownership channel shared by the async and sync core callback handlers.

    One instance per handler/middleware installation — never a module-global
    (multiple concurrent agent runs must not share ownership state).
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._workflows: dict[str, WorkflowRecords] = {}

    # ── Record creation ─────────────────────────────────────────────────

    def prepare_tool(
        self,
        workflow_id: str,
        activity_id: str,
        *,
        tool_name: str | None = None,
        tool_type: str | None = None,
        tool_call_id: str | None = None,
        langgraph_node: str | None = None,
        langgraph_step: int | None = None,
    ) -> ActivityRecord:
        """Get-or-create the record for a tool activity.

        Idempotent: calling twice for the same ``(workflow_id, activity_id)``
        returns the SAME record (does not reset sent-flags/stashed verdicts) —
        required for the evaluate-once contract when both handlers fire.
        """
        with self._lock:
            wf = self._workflows.setdefault(workflow_id, WorkflowRecords())
            record = wf.by_activity_id.get(activity_id)
            if record is None:
                record = ActivityRecord(
                    activity_id=activity_id,
                    tool_name=tool_name,
                    tool_type=tool_type,
                    tool_call_id=tool_call_id,
                    langgraph_node=langgraph_node,
                    langgraph_step=langgraph_step,
                )
                wf.by_activity_id[activity_id] = record
            return record

    def prepare_llm(
        self,
        workflow_id: str,
        activity_id: str,
        *,
        event_run_id: str | None = None,
    ) -> ActivityRecord:
        """Get-or-create the record for an LLM activity.

        ``event_run_id`` registers the alias (H11) from the callback's own
        run identifier to ``activity_id`` — needed when the first LLM call's
        activity_id diverges from the callback run_id (e.g. a pre-screen row
        keyed ``"{run_id}-pre"``).
        """
        with self._lock:
            wf = self._workflows.setdefault(workflow_id, WorkflowRecords())
            record = wf.by_activity_id.get(activity_id)
            if record is None:
                record = ActivityRecord(activity_id=activity_id)
                wf.by_activity_id[activity_id] = record
            if event_run_id is not None:
                wf.event_run_id_alias[event_run_id] = activity_id
            return record

    # ── Lookup ───────────────────────────────────────────────────────────

    def get(self, workflow_id: str, activity_id: str) -> ActivityRecord | None:
        with self._lock:
            wf = self._workflows.get(workflow_id)
            if wf is None:
                return None
            return wf.by_activity_id.get(activity_id)

    def get_by_event_run_id(self, workflow_id: str, event_run_id: str) -> ActivityRecord | None:
        """Resolve a record via the H11 alias index (falls back to a direct
        activity_id lookup so callers can pass either key uniformly)."""
        with self._lock:
            wf = self._workflows.get(workflow_id)
            if wf is None:
                return None
            activity_id = wf.event_run_id_alias.get(event_run_id, event_run_id)
            return wf.by_activity_id.get(activity_id)

    # ── Ownership query (C1) ────────────────────────────────────────────

    def is_callback_owned(
        self, workflow_id: str, activity_id: str, event_type: EventType
    ) -> bool:
        """True iff the callback has SENT this specific event type.

        Record-exists is NEVER ownership — a record may be prepared by an
        integrating-layer seam (e.g. a subagent gate) while the callback never
        runs; that path must report False here so the consumer does not skip
        an event nobody sent.
        """
        record = self.get(workflow_id, activity_id)
        if record is None:
            return False
        if event_type == "tool_start":
            return record.tool_started_sent
        if event_type == "tool_complete":
            return record.tool_completed_sent
        if event_type == "llm_start":
            return record.llm_started_sent
        return record.llm_completed_sent

    # ── Cleanup ──────────────────────────────────────────────────────────

    def sweep_workflow(self, workflow_id: str) -> list[ActivityRecord]:
        """Drop and return all records for a workflow (turn/run cleanup).

        The integrating layer uses the returned records to sweep-close any
        prepared-but-not-completed activities (C6 sibling-cancellation case).
        """
        with self._lock:
            wf = self._workflows.pop(workflow_id, None)
            if wf is None:
                return []
            return list(wf.by_activity_id.values())
