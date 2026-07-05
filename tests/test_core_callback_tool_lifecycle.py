"""Tool-lifecycle tests for the pure LangChain-Core callback handlers.

Covers phase-02 step 9 + Success Criteria: start-before-body ordering,
BLOCK/HALT on both async and sync (real ``BaseTool.run``) paths, async
REQUIRE_APPROVAL pre-body raise, sync REQUIRE_APPROVAL fail-safe, orphan-row
close on start-raise (C6), sent-flag idempotency under duplicate dispatch,
completion reusing the start id, ``record_less_ok=False`` suppression, and
ownership asserted in both directions (record-exists != owned).

No network — ``FakeGate``/``FakeAdapter``/``FakeRuntime`` from
``tests/fakes_core_callback.py`` record envelopes and return scripted verdicts.
"""

from __future__ import annotations

import pytest
from langchain_core.tools import StructuredTool
from openbox_core.errors import ApprovalRejectedError, GovernanceBlockedError, GovernanceHaltError

from openbox_langchain.activity_bridge import ActivityBridge
from openbox_langchain.core_callback_async import OpenBoxLangChainCoreAsyncCallbackHandler
from openbox_langchain.core_callback_options import OpenBoxLangChainCoreCallbackOptions
from openbox_langchain.core_callback_sync import OpenBoxLangChainCoreSyncCallbackHandler
from tests.fakes_core_callback import (
    FakeAdapter,
    FakeGate,
    FakeRuntime,
    block_result,
    halt_result,
    require_approval_result,
)

WORKFLOW_ID = "wf-1"
RUN_ID = "run-1"
WORKFLOW_TYPE = "test-workflow"


def make_options(
    *, gate: FakeGate | None = None, adapter: FakeAdapter | None = None, **overrides
) -> tuple[OpenBoxLangChainCoreCallbackOptions, FakeRuntime]:
    runtime = FakeRuntime(gate=gate or FakeGate(), adapter=adapter or FakeAdapter())
    bridge = overrides.pop("bridge", ActivityBridge())
    options = OpenBoxLangChainCoreCallbackOptions(
        runtime=runtime,  # type: ignore[arg-type]
        bridge=bridge,
        workflow_id=WORKFLOW_ID,
        run_id=RUN_ID,
        workflow_type=WORKFLOW_TYPE,
        **overrides,
    )
    return options, runtime


def make_async_tool(name: str = "async_tool") -> StructuredTool:
    body_ran: dict[str, bool] = {"ran": False}

    async def _coro(x: int = 1) -> str:
        body_ran["ran"] = True
        return "async-result"

    def _sync(x: int = 1) -> str:
        body_ran["ran"] = True
        return "sync-result"

    tool = StructuredTool.from_function(func=_sync, coroutine=_coro, name=name, description="d")
    tool.__dict__["_body_ran"] = body_ran
    return tool


def make_sync_only_tool(name: str = "sync_tool") -> StructuredTool:
    body_ran: dict[str, bool] = {"ran": False}

    def _sync(x: int = 1) -> str:
        body_ran["ran"] = True
        return "sync-result"

    tool = StructuredTool.from_function(func=_sync, name=name, description="d")
    tool.__dict__["_body_ran"] = body_ran
    return tool


# ── Async path: start-before-body, BLOCK/HALT, REQUIRE_APPROVAL ────────────


async def test_async_tool_start_sent_before_body_runs() -> None:
    """ToolStarted is sent (and the gate called) before the tool body executes."""
    activity_id_holder: dict[str, str] = {}
    order: list[str] = []

    class OrderedGate(FakeGate):
        async def aevaluate(self, event):  # type: ignore[override]
            order.append("gate_evaluate")
            if event.activity_id:
                activity_id_holder["id"] = event.activity_id
            return await super().aevaluate(event)

    gate = OrderedGate()
    options, _ = make_options(gate=gate)
    handler = OpenBoxLangChainCoreAsyncCallbackHandler(options)

    tool = make_async_tool()

    async def _tracking_coro(x: int = 1) -> str:
        order.append("body_ran")
        return "ok"

    tool.coroutine = _tracking_coro

    result = await tool.ainvoke({"x": 1}, config={"callbacks": [handler]})
    assert result == "async-result" or result == "ok"
    # First evaluate (ToolStarted) precedes the body; a second evaluate call
    # (ToolCompleted) may follow the body — assert on the PREFIX only.
    assert order[:2] == ["gate_evaluate", "body_ran"]
    activity_id = activity_id_holder["id"]
    assert options.bridge.is_callback_owned(WORKFLOW_ID, activity_id, "tool_start")


async def test_async_block_prevents_body_and_raises() -> None:
    """A BLOCK verdict on tool start raises before the body runs (async path)."""
    gate = FakeGate()
    options, _ = make_options(gate=gate)
    handler = OpenBoxLangChainCoreAsyncCallbackHandler(options)
    tool = make_async_tool()

    # Script BLOCK once we learn the run_id-derived activity_id — use a gate
    # that blocks the FIRST evaluate call unconditionally instead.
    class BlockFirstGate(FakeGate):
        async def aevaluate(self, event):  # type: ignore[override]
            if event.activity_id and event.event_type.value == "ActivityStarted":
                self.verdicts[event.activity_id] = block_result()
            return await super().aevaluate(event)

    options.runtime.gate = BlockFirstGate()  # type: ignore[attr-defined]

    with pytest.raises(GovernanceBlockedError):
        await tool.ainvoke({"x": 1}, config={"callbacks": [handler]})

    assert tool.__dict__["_body_ran"]["ran"] is False


async def test_async_halt_prevents_body_and_raises() -> None:
    class HaltFirstGate(FakeGate):
        async def aevaluate(self, event):  # type: ignore[override]
            if event.activity_id and event.event_type.value == "ActivityStarted":
                self.verdicts[event.activity_id] = halt_result()
            return await super().aevaluate(event)

    options, _ = make_options(gate=HaltFirstGate())
    handler = OpenBoxLangChainCoreAsyncCallbackHandler(options)
    tool = make_async_tool()

    with pytest.raises(GovernanceHaltError):
        await tool.ainvoke({"x": 1}, config={"callbacks": [handler]})
    assert tool.__dict__["_body_ran"]["ran"] is False


async def test_async_require_approval_raises_pre_body_when_no_approval_flow() -> None:
    """Fail-safe: REQUIRE_APPROVAL with no approval flow configured rejects (M15)."""

    class ApprovalFirstGate(FakeGate):
        async def aevaluate(self, event):  # type: ignore[override]
            if event.activity_id and event.event_type.value == "ActivityStarted":
                self.verdicts[event.activity_id] = require_approval_result()
            return await super().aevaluate(event)

    options, _ = make_options(gate=ApprovalFirstGate())
    handler = OpenBoxLangChainCoreAsyncCallbackHandler(options)
    tool = make_async_tool()

    with pytest.raises(ApprovalRejectedError):
        await tool.ainvoke({"x": 1}, config={"callbacks": [handler]})
    assert tool.__dict__["_body_ran"]["ran"] is False


async def test_start_raise_sends_failed_completion_same_id_no_orphan() -> None:
    """C6: a stop-shaped start verdict closes its own row (same activity_id) before raising."""

    class BlockFirstGate(FakeGate):
        async def aevaluate(self, event):  # type: ignore[override]
            if event.activity_id and event.event_type.value == "ActivityStarted":
                self.verdicts[event.activity_id] = block_result("no tools allowed")
            return await super().aevaluate(event)

    gate = BlockFirstGate()
    options, _ = make_options(gate=gate)
    handler = OpenBoxLangChainCoreAsyncCallbackHandler(options)
    tool = make_async_tool()

    with pytest.raises(GovernanceBlockedError):
        await tool.ainvoke({"x": 1}, config={"callbacks": [handler]})

    # Exactly one ActivityStarted and one ActivityCompleted, same activity_id.
    started = [e for e in gate.sent if e.payload.get("event_type") == "ActivityStarted"]
    completed = [e for e in gate.sent if e.payload.get("event_type") == "ActivityCompleted"]
    assert len(started) == 1
    assert len(completed) == 1
    assert started[0].activity_id == completed[0].activity_id
    assert completed[0].payload.get("error")


# ── Sync-only-tool corner (C2) — the async-handler-swallowed hazard + the fix ──


def test_sync_block_via_base_tool_run_and_sync_handler_stops_body() -> None:
    """Sync BLOCK via BaseTool.run + the SYNC handler stops the body (the fix)."""

    class BlockFirstGate(FakeGate):
        def evaluate(self, event):  # type: ignore[override]
            if event.activity_id and event.event_type.value == "ActivityStarted":
                self.verdicts[event.activity_id] = block_result()
            return super().evaluate(event)

    options, _ = make_options(gate=BlockFirstGate())
    handler = OpenBoxLangChainCoreSyncCallbackHandler(options)
    tool = make_sync_only_tool()

    with pytest.raises(GovernanceBlockedError):
        tool.run({"x": 1}, callbacks=[handler])

    assert tool.__dict__["_body_ran"]["ran"] is False


async def test_async_handler_only_path_is_swallowed_for_sync_only_tool() -> None:
    """Proves the async-handler-alone hazard: BLOCK is NOT relied upon via the
    async handler for a sync-only tool invoked through BaseTool.run's sync
    callback manager (cross-dispatch swallows the raise, per Phase 0)."""

    class BlockFirstGate(FakeGate):
        async def aevaluate(self, event):  # type: ignore[override]
            if event.activity_id and event.event_type.value == "ActivityStarted":
                self.verdicts[event.activity_id] = block_result()
            return await super().aevaluate(event)

    options, _ = make_options(gate=BlockFirstGate())
    handler = OpenBoxLangChainCoreAsyncCallbackHandler(options)
    tool = make_sync_only_tool()

    # Direct BaseTool.run (sync callback manager) with ONLY the async handler
    # installed: the manager drives the coroutine via _run_coros, which logs
    # the raise instead of propagating it — body runs regardless.
    result = tool.run({"x": 1}, callbacks=[handler])
    assert result == "sync-result"
    assert tool.__dict__["_body_ran"]["ran"] is True


def test_sync_require_approval_fail_safe_without_handle_approval_sync() -> None:
    """Sync REQUIRE_APPROVAL fail-safe: no handle_approval_sync -> ApprovalRejectedError."""

    class ApprovalFirstGate(FakeGate):
        def evaluate(self, event):  # type: ignore[override]
            if event.activity_id and event.event_type.value == "ActivityStarted":
                self.verdicts[event.activity_id] = require_approval_result()
            return super().evaluate(event)

    adapter = FakeAdapter()
    options, _ = make_options(gate=ApprovalFirstGate(), adapter=adapter)
    # Shadow handle_approval_sync with None to exercise the "adapter defines
    # none" branch (enforce_tool_start_sync checks `callable(...)`).
    adapter.handle_approval_sync = None  # type: ignore[assignment]
    handler = OpenBoxLangChainCoreSyncCallbackHandler(options)
    tool = make_sync_only_tool()

    with pytest.raises(ApprovalRejectedError):
        tool.run({"x": 1}, callbacks=[handler])
    assert tool.__dict__["_body_ran"]["ran"] is False


def test_sync_corner_both_handlers_block_is_failclosed_pre_body() -> None:
    """THE production sync corner: both handlers installed on a sync-only tool
    invoked via BaseTool.run (sync callback manager), BLOCK start verdict.

    The async handler's raise is swallowed by the sync manager's ``_run_coros``,
    but the SYNC handler re-raises the (fresh or stashed) BLOCK INLINE pre-body
    (fail-closed). Also locks in evaluate-once (exactly one ActivityStarted) and
    the guarded orphan-close (exactly one failed ActivityCompleted, not doubled
    by the cross-dispatched sibling) so a future edit to the reuse-from-stash
    branch cannot silently open a fail-OPEN hole.
    """

    class BlockFirstGate(FakeGate):
        def evaluate(self, event):  # type: ignore[override]
            if event.activity_id and event.event_type.value == "ActivityStarted":
                self.verdicts[event.activity_id] = block_result()
            return super().evaluate(event)

        async def aevaluate(self, event):  # type: ignore[override]
            if event.activity_id and event.event_type.value == "ActivityStarted":
                self.verdicts[event.activity_id] = block_result()
            return await super().aevaluate(event)

    gate = BlockFirstGate()
    # A single options object => both handlers share ONE bridge (and its stash).
    options, _ = make_options(gate=gate)
    async_handler = OpenBoxLangChainCoreAsyncCallbackHandler(options)
    sync_handler = OpenBoxLangChainCoreSyncCallbackHandler(options)
    tool = make_sync_only_tool()

    with pytest.raises(GovernanceBlockedError):
        tool.run({"x": 1}, callbacks=[async_handler, sync_handler])

    # Fail-closed: the tool body never ran despite the swallowed async raise.
    assert tool.__dict__["_body_ran"]["ran"] is False
    started = [e for e in gate.sent if e.payload.get("event_type") == "ActivityStarted"]
    completed = [e for e in gate.sent if e.payload.get("event_type") == "ActivityCompleted"]
    # Evaluate-once: exactly one ActivityStarted regardless of cross-dispatch.
    assert len(started) == 1, f"expected 1 ActivityStarted (evaluate-once), got {len(started)}"
    # Orphan close guarded: exactly one failed ActivityCompleted, same id, no -c.
    assert len(completed) == 1, f"expected 1 orphan-close ActivityCompleted, got {len(completed)}"
    assert started[0].activity_id == completed[0].activity_id
    assert completed[0].payload.get("error")


# ── Duplicate dispatch idempotency (sent-flags) ─────────────────────────────


async def test_duplicate_tool_start_and_end_idempotent_via_sent_flags() -> None:
    """Calling on_tool_start/on_tool_end twice for the same run_id sends once."""
    gate = FakeGate()
    options, _ = make_options(gate=gate)
    handler = OpenBoxLangChainCoreAsyncCallbackHandler(options)

    import uuid

    run_id = uuid.uuid4()
    serialized = {"name": "dup_tool"}

    await handler.on_tool_start(serialized, "{}", run_id=run_id, inputs={"x": 1})
    await handler.on_tool_start(serialized, "{}", run_id=run_id, inputs={"x": 1})
    await handler.on_tool_end("result", run_id=run_id)
    await handler.on_tool_end("result", run_id=run_id)

    started = [e for e in gate.sent if e.payload.get("event_type") == "ActivityStarted"]
    completed = [e for e in gate.sent if e.payload.get("event_type") == "ActivityCompleted"]
    assert len(started) == 1
    assert len(completed) == 1
    assert started[0].activity_id == completed[0].activity_id == str(run_id)


# ── record_less_ok=False suppression (C8) ───────────────────────────────────


async def test_record_less_ok_false_suppresses_send_without_record() -> None:
    """When record_less_ok=False, the callback does not send absent a bridge record."""
    gate = FakeGate()
    options, _ = make_options(gate=gate, record_less_ok=False)
    handler = OpenBoxLangChainCoreAsyncCallbackHandler(options)

    import uuid

    run_id = uuid.uuid4()
    # on_tool_end with NO prior prepare_tool/on_tool_start call: no record exists.
    await handler.on_tool_end("result", run_id=run_id)

    assert gate.sent == []


async def test_record_less_ok_true_allows_send_without_prior_prepare() -> None:
    """Pure-LangChain default (record_less_ok=True): on_tool_start itself
    creates the record, so this documents that a record IS created on start —
    the record-less allowance applies to callback methods invoked without ANY
    record existing yet (e.g. on_tool_end with no matching on_tool_start,
    which still must not crash when record_less_ok=True)."""
    gate = FakeGate()
    options, _ = make_options(gate=gate, record_less_ok=True)
    handler = OpenBoxLangChainCoreAsyncCallbackHandler(options)

    import uuid

    run_id = uuid.uuid4()
    # No prior on_tool_start -> no record. record_less_ok=True must not crash;
    # tool_name falls back to "unknown_tool" and the completion still sends.
    await handler.on_tool_end("result", run_id=run_id)

    completed = [e for e in gate.sent if e.payload.get("event_type") == "ActivityCompleted"]
    assert len(completed) == 1
    assert completed[0].activity_id == str(run_id)


# ── Ownership asserted BOTH directions (record-exists != owned) ────────────


def test_ownership_false_when_no_record_exists() -> None:
    bridge = ActivityBridge()
    assert bridge.is_callback_owned(WORKFLOW_ID, "nonexistent", "tool_start") is False


def test_ownership_false_when_record_exists_but_not_sent() -> None:
    """A record may be prepared (e.g. an integrating-layer seam) without the
    callback ever sending — record-exists must NOT imply ownership."""
    bridge = ActivityBridge()
    bridge.prepare_tool(WORKFLOW_ID, "act-1", tool_name="t")
    assert bridge.get(WORKFLOW_ID, "act-1") is not None
    assert bridge.is_callback_owned(WORKFLOW_ID, "act-1", "tool_start") is False
    assert bridge.is_callback_owned(WORKFLOW_ID, "act-1", "tool_complete") is False


def test_ownership_true_only_for_the_specific_event_type_sent() -> None:
    bridge = ActivityBridge()
    bridge.prepare_tool(WORKFLOW_ID, "act-2", tool_name="t")
    bridge.mark_sent(WORKFLOW_ID, "act-2", "tool_start")
    assert bridge.is_callback_owned(WORKFLOW_ID, "act-2", "tool_start") is True
    assert bridge.is_callback_owned(WORKFLOW_ID, "act-2", "tool_complete") is False


# ── Completion reuses the start id ──────────────────────────────────────────


async def test_completion_reuses_start_activity_id_no_suffix() -> None:
    gate = FakeGate()
    options, _ = make_options(gate=gate)
    handler = OpenBoxLangChainCoreAsyncCallbackHandler(options)
    tool = make_async_tool()

    await tool.ainvoke({"x": 1}, config={"callbacks": [handler]})

    started = [e for e in gate.sent if e.payload.get("event_type") == "ActivityStarted"]
    completed = [e for e in gate.sent if e.payload.get("event_type") == "ActivityCompleted"]
    assert len(started) == 1
    assert len(completed) == 1
    assert started[0].activity_id == completed[0].activity_id
    # Forbidden legacy suffix (middleware_tool_hook.py's `f"{activity_id}-c"`
    # pattern) — assert the exact suffix is absent, not a raw substring check
    # (a UUID may coincidentally contain "-c" elsewhere).
    assert not (completed[0].activity_id or "").endswith("-c")


# ── Evaluate-once across cross-dispatched handlers (C2, both directions) ────


async def test_evaluate_once_when_both_handlers_installed_on_async_graph() -> None:
    """Both async+sync handlers installed on the async graph path: exactly one
    gate call per activity even though cross-dispatch fires both (Phase 0)."""
    gate = FakeGate()
    bridge = ActivityBridge()
    options, _ = make_options(gate=gate, bridge=bridge)
    async_handler = OpenBoxLangChainCoreAsyncCallbackHandler(options)
    sync_handler = OpenBoxLangChainCoreSyncCallbackHandler(options)
    tool = make_async_tool()

    await tool.ainvoke({"x": 1}, config={"callbacks": [async_handler, sync_handler]})

    started = [e for e in gate.sent if e.payload.get("event_type") == "ActivityStarted"]
    assert len(started) == 1
