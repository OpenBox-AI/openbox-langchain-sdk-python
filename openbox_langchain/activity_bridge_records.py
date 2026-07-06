"""Record dataclasses for ``ActivityBridge`` — split out to keep
``activity_bridge.py`` under 200 lines. Internal to the bridge; not part of
the public API surface beyond ``ActivityRecord`` (re-exported).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from openbox_core.contracts.results import EvaluationResult

__all__ = ["ActivityRecord", "WorkflowRecords"]


@dataclass
class ActivityRecord:
    """Bridge-side bookkeeping for one activity (tool call or LLM call).

    Fields intentionally scoped to what Phase 2 handlers and the Phase 4/5
    integrating layers need to read — see ``activity_bridge.py``'s module
    docstring for what was cut (H12) and why.
    """

    activity_id: str

    # Per-event-type sent-flags (C1 — the ownership pivot).
    tool_started_sent: bool = False
    tool_completed_sent: bool = False
    llm_started_sent: bool = False
    llm_completed_sent: bool = False

    # Evaluate-once / enforce-from-stash (C2 sync corner).
    start_result: EvaluationResult | None = None
    # Completion verdict — the integrating layer (Phase 4/5) reads this to
    # enforce/poll; completion sends are gate.aevaluate-only (C4), never
    # adapter-enforcing, so this is the sole channel for that decision.
    completion_result: EvaluationResult | None = None

    # Tool metadata not carried on `current_activity_context()`.
    tool_name: str | None = None
    tool_type: str | None = None
    tool_call_id: str | None = None
    langgraph_node: str | None = None
    langgraph_step: int | None = None

    # Phase 4/M19 — set by the integrating layer when it force-aborts an
    # in-flight activity (e.g. sibling cancellation sweep).
    abort_marked: bool = False


@dataclass
class WorkflowRecords:
    """Per-workflow record storage, including the LLM event_run_id alias index."""

    by_activity_id: dict[str, ActivityRecord] = field(default_factory=dict)
    # H11: the first LLM call's record may be keyed by a different id than the
    # callback's own run_id (e.g. a "{run_id}-pre" pre-screen row). This maps
    # the callback's `event_run_id` to the `activity_id` actually used, so
    # completion resolves without emitting an orphan `-c` row.
    event_run_id_alias: dict[str, str] = field(default_factory=dict)
