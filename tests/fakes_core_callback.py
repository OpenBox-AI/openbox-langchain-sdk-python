"""Fakes for core-callback tests — a scripted gate + adapter + runtime.

No network. The fake gate records every envelope sent (for assertion) and
returns pre-scripted ``EvaluationResult`` verdicts per activity_id so tests
control BLOCK/HALT/REQUIRE_APPROVAL/ALLOW exactly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from openbox_core.context import ContextStore
from openbox_core.contracts.events import EventEnvelope
from openbox_core.contracts.results import EvaluationResult, Verdict
from openbox_core.errors import ApprovalRejectedError, GovernanceBlockedError, GovernanceHaltError


@dataclass
class RecordedEnvelope:
    """One envelope the fake gate observed, plus which gate method sent it."""

    method: str  # "evaluate" | "aevaluate"
    activity_id: str | None
    payload: dict[str, Any]


class FakeGate:
    """Scripted ``GovernanceGate`` stand-in — sync + async evaluate.

    ``verdicts`` maps ``activity_id -> EvaluationResult``; activities not in
    the map default to ALLOW. ``call_count`` tracks total evaluate calls per
    activity_id so tests assert "gate called at most once" (C2).
    """

    def __init__(self, verdicts: dict[str, EvaluationResult] | None = None) -> None:
        self.verdicts = verdicts or {}
        self.sent: list[RecordedEnvelope] = []
        self.call_count: dict[str, int] = {}

    def _resolve(self, event: EventEnvelope) -> EvaluationResult:
        activity_id = event.activity_id
        if activity_id is not None:
            self.call_count[activity_id] = self.call_count.get(activity_id, 0) + 1
        self.sent.append(
            RecordedEnvelope(
                method="evaluate", activity_id=activity_id, payload=event.to_payload_dict()
            )
        )
        if activity_id and activity_id in self.verdicts:
            return self.verdicts[activity_id]
        return EvaluationResult(verdict=Verdict.ALLOW)

    def evaluate(self, event: EventEnvelope) -> EvaluationResult:
        return self._resolve(event)

    async def aevaluate(self, event: EventEnvelope) -> EvaluationResult:
        return self._resolve(event)


class FakeAdapter:
    """Fail-closed adapter matching ``CoreAdapter``'s default (no approval poller)."""

    name = "fake"

    def __init__(self) -> None:
        self.approval_calls: list[EvaluationResult] = []
        self.approval_sync_calls: list[EvaluationResult] = []
        self.blocked_calls: list[EvaluationResult] = []
        # When True, handle_approval/handle_approval_sync return normally
        # (simulating an already-approved decision) instead of raising.
        self.auto_approve = False

    async def handle_approval(self, result: EvaluationResult) -> None:
        self.approval_calls.append(result)
        if self.auto_approve:
            return
        raise ApprovalRejectedError("fake adapter: no approval flow configured")

    def handle_approval_sync(self, result: EvaluationResult) -> None:
        self.approval_sync_calls.append(result)
        if self.auto_approve:
            return
        raise ApprovalRejectedError("fake adapter: no sync approval flow configured")

    def raise_lifecycle_blocked(self, result: EvaluationResult) -> None:
        self.blocked_calls.append(result)
        if result.verdict is Verdict.HALT:
            raise GovernanceHaltError(result.reason or "halted")
        raise GovernanceBlockedError(result.verdict, result.reason or "blocked")

    def raise_hook_blocked(self, result: EvaluationResult) -> None:
        self.raise_lifecycle_blocked(result)

    def on_completed_hook_result(self, result: EvaluationResult, context: Any = None) -> None:
        return None


@dataclass
class FakeRuntime:
    """Minimal ``OpenBoxRuntime`` stand-in exposing ``.gate``/``.adapter``/``.context_store``.

    ``context_store`` is a REAL ``ContextStore`` (not faked) — the middleware
    tool-hook tests bind it via ``activity_scope`` and the real bind/reset
    semantics matter (leak-safety, trace correlation), so faking it would
    hide bugs rather than isolate them.
    """

    gate: FakeGate = field(default_factory=FakeGate)
    adapter: FakeAdapter = field(default_factory=FakeAdapter)
    context_store: ContextStore = field(default_factory=ContextStore)


def block_result(reason: str = "blocked by policy") -> EvaluationResult:
    return EvaluationResult(verdict=Verdict.BLOCK, reason=reason)


def halt_result(reason: str = "halted by policy") -> EvaluationResult:
    return EvaluationResult(verdict=Verdict.HALT, reason=reason)


def require_approval_result(reason: str = "needs approval") -> EvaluationResult:
    return EvaluationResult(verdict=Verdict.REQUIRE_APPROVAL, reason=reason)


def allow_result() -> EvaluationResult:
    return EvaluationResult(verdict=Verdict.ALLOW)
