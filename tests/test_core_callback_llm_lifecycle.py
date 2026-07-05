"""LLM-lifecycle tests for the pure LangChain-Core callback handlers.

Covers phase-02 step 9: LLMStarted sent before the fake provider call,
redaction mutates pre-call, pre-screen verdict reuse (one row, no duplicate
gate call), event_run_id alias resolves completion (H11, no orphan '-c' row),
trace register/unregister via injected callables, and completion via
gate.aevaluate/evaluate only (C4 — no enforcement raise on LLM completion).

No network — ``FakeGate``/``FakeRuntime`` record envelopes; a scripted fake
chat model stands in for the provider HTTP call.
"""

from __future__ import annotations

import uuid

from langchain_core.messages import AIMessage, HumanMessage
from openbox_core.contracts.results import EvaluationResult, GuardrailsResult, Verdict

from openbox_langchain.activity_bridge import ActivityBridge
from openbox_langchain.core_callback_async import OpenBoxLangChainCoreAsyncCallbackHandler
from openbox_langchain.core_callback_options import OpenBoxLangChainCoreCallbackOptions
from openbox_langchain.core_callback_sync import OpenBoxLangChainCoreSyncCallbackHandler
from tests.fakes_core_callback import FakeAdapter, FakeGate, FakeRuntime

WORKFLOW_ID = "wf-llm-1"
RUN_ID = "run-llm-1"
WORKFLOW_TYPE = "test-workflow"


def make_options(
    *, gate: FakeGate | None = None, **overrides
) -> OpenBoxLangChainCoreCallbackOptions:
    runtime = FakeRuntime(gate=gate or FakeGate(), adapter=FakeAdapter())
    bridge = overrides.pop("bridge", ActivityBridge())
    return OpenBoxLangChainCoreCallbackOptions(
        runtime=runtime,  # type: ignore[arg-type]
        bridge=bridge,
        workflow_id=WORKFLOW_ID,
        run_id=RUN_ID,
        workflow_type=WORKFLOW_TYPE,
        **overrides,
    )


def make_messages(text: str = "hello there") -> list[list[HumanMessage]]:
    return [[HumanMessage(content=text)]]


async def fake_provider_call(order: list[str]) -> AIMessage:
    """Stand-in for the model's HTTP call — records that it ran."""
    order.append("provider_call")
    return AIMessage(content="a response")


# ── LLMStarted before the fake provider call ────────────────────────────────


async def test_llm_started_sent_before_provider_call() -> None:
    order: list[str] = []
    gate = FakeGate()
    options = make_options(gate=gate)
    handler = OpenBoxLangChainCoreAsyncCallbackHandler(options)

    class TrackingGate(FakeGate):
        async def aevaluate(self, event):  # type: ignore[override]
            order.append("gate_evaluate")
            return await super().aevaluate(event)

    options.runtime.gate = TrackingGate()  # type: ignore[attr-defined]

    run_id = uuid.uuid4()
    await handler.on_chat_model_start({"name": "fake-model"}, make_messages(), run_id=run_id)
    await fake_provider_call(order)
    await handler.on_llm_end(AIMessage(content="a response"), run_id=run_id)

    assert order[0] == "gate_evaluate"
    assert "provider_call" in order
    assert order.index("gate_evaluate") < order.index("provider_call")


# ── Redaction mutates pre-call ──────────────────────────────────────────────


async def test_redaction_mutates_messages_before_provider_call() -> None:
    class RedactingGate(FakeGate):
        async def aevaluate(self, event):  # type: ignore[override]
            result = await super().aevaluate(event)
            if event.event_type.value == "ActivityStarted":
                return EvaluationResult(
                    verdict=Verdict.ALLOW,
                    guardrails=GuardrailsResult(
                        redacted_input="[REDACTED]", input_type="activity_input"
                    ),
                )
            return result

    options = make_options(gate=RedactingGate())
    handler = OpenBoxLangChainCoreAsyncCallbackHandler(options)

    messages = make_messages("my secret ssn is 123-45-6789")
    run_id = uuid.uuid4()
    await handler.on_chat_model_start({"name": "fake-model"}, messages, run_id=run_id)

    assert messages[0][0].content == "[REDACTED]"


async def test_redaction_skipped_for_activity_output_input_type() -> None:
    """Redaction targeting the OUTPUT never mutates the pre-call prompt."""

    class OutputRedactingGate(FakeGate):
        async def aevaluate(self, event):  # type: ignore[override]
            if event.event_type.value == "ActivityStarted":
                return EvaluationResult(
                    verdict=Verdict.ALLOW,
                    guardrails=GuardrailsResult(
                        redacted_input="[REDACTED]", input_type="activity_output"
                    ),
                )
            return await super().aevaluate(event)

    options = make_options(gate=OutputRedactingGate())
    handler = OpenBoxLangChainCoreAsyncCallbackHandler(options)

    messages = make_messages("keep me")
    run_id = uuid.uuid4()
    await handler.on_chat_model_start({"name": "fake-model"}, messages, run_id=run_id)

    assert messages[0][0].content == "keep me"


# ── Pre-screen reuse (one row, no duplicate gate call) ──────────────────────


async def test_pre_screen_reuse_sends_no_duplicate_started_event() -> None:
    pre_screen = EvaluationResult(verdict=Verdict.ALLOW)
    gate = FakeGate()
    options = make_options(
        gate=gate, pre_screen_response=pre_screen, pre_screen_activity_id="pre-1"
    )
    handler = OpenBoxLangChainCoreAsyncCallbackHandler(options)

    run_id = uuid.uuid4()
    await handler.on_chat_model_start({"name": "fake-model"}, make_messages(), run_id=run_id)

    # The pre-screened verdict is consumed WITHOUT a fresh gate call.
    started = [e for e in gate.sent if e.payload.get("event_type") == "ActivityStarted"]
    assert started == []
    assert options.bridge.is_callback_owned(WORKFLOW_ID, "pre-1", "llm_start")
    # Consumed once: a second chat-model call must evaluate for real.
    assert options.pre_screen_response is None
    assert options.pre_screen_activity_id is None

    run_id_2 = uuid.uuid4()
    await handler.on_chat_model_start({"name": "fake-model"}, make_messages(), run_id=run_id_2)
    started_after = [e for e in gate.sent if e.payload.get("event_type") == "ActivityStarted"]
    assert len(started_after) == 1


# ── event_run_id alias resolves completion (H11, no orphan -c row) ─────────


async def test_event_run_id_alias_resolves_completion_without_orphan() -> None:
    """The pre-screen activity_id diverges from the callback run_id; completion
    must resolve via the alias, not emit a second '-c'-suffixed row."""
    pre_screen = EvaluationResult(verdict=Verdict.ALLOW)
    gate = FakeGate()
    options = make_options(
        gate=gate, pre_screen_response=pre_screen, pre_screen_activity_id="run-1-pre"
    )
    handler = OpenBoxLangChainCoreAsyncCallbackHandler(options)

    run_id = uuid.uuid4()
    await handler.on_chat_model_start({"name": "fake-model"}, make_messages(), run_id=run_id)
    await handler.on_llm_end(AIMessage(content="done"), run_id=run_id)

    completed = [e for e in gate.sent if e.payload.get("event_type") == "ActivityCompleted"]
    assert len(completed) == 1
    assert completed[0].activity_id == "run-1-pre"
    assert not (completed[0].activity_id or "").endswith("-c")


# ── Trace register/unregister via injected callables ────────────────────────


def test_trace_register_unregister_use_injected_callables() -> None:
    """register_trace/unregister_trace default to the base store; overriding
    them at construction time is honored (Phase 5 injects registry-backed ones)."""
    calls: list[str] = []

    def custom_register(trace_id, ctx) -> None:
        calls.append("register")

    def custom_unregister(trace_id) -> None:
        calls.append("unregister")

    options = make_options(register_trace=custom_register, unregister_trace=custom_unregister)
    assert options.register_trace is custom_register
    assert options.unregister_trace is custom_unregister
    # Directly exercise the injected callables the way register_llm_trace would.
    options.register_trace(123, None)
    options.unregister_trace(123)
    assert calls == ["register", "unregister"]


# ── Completion via gate.aevaluate only (C4 — no enforcement raise) ──────────


async def test_llm_completion_uses_aevaluate_and_never_enforces() -> None:
    """Even a BLOCK-shaped completion verdict must NOT raise — it is stashed
    for the integrating layer, never adapter-enforced (C4)."""

    class BlockingCompletionGate(FakeGate):
        async def aevaluate(self, event):  # type: ignore[override]
            if event.event_type.value == "ActivityCompleted":
                return EvaluationResult(verdict=Verdict.BLOCK, reason="policy changed mid-call")
            return await super().aevaluate(event)

    options = make_options(gate=BlockingCompletionGate())
    handler = OpenBoxLangChainCoreAsyncCallbackHandler(options)

    run_id = uuid.uuid4()
    await handler.on_chat_model_start({"name": "fake-model"}, make_messages(), run_id=run_id)
    # Must not raise despite the BLOCK-shaped completion verdict.
    await handler.on_llm_end(AIMessage(content="done"), run_id=run_id)

    record = options.bridge.get_by_event_run_id(WORKFLOW_ID, str(run_id))
    assert record is not None
    assert record.completion_result is not None
    assert record.completion_result.verdict == Verdict.BLOCK


async def test_on_llm_error_guarded_by_completed_sent_no_double_send() -> None:
    """A return_exceptions=True-captured end that already sent must not
    double-send a failed row when on_llm_error also fires."""
    gate = FakeGate()
    options = make_options(gate=gate)
    handler = OpenBoxLangChainCoreAsyncCallbackHandler(options)

    run_id = uuid.uuid4()
    await handler.on_chat_model_start({"name": "fake-model"}, make_messages(), run_id=run_id)
    await handler.on_llm_end(AIMessage(content="done"), run_id=run_id)
    # Simulate a duplicate error callback firing after the end already sent.
    await handler.on_llm_error(RuntimeError("boom"), run_id=run_id)

    completed = [e for e in gate.sent if e.payload.get("event_type") == "ActivityCompleted"]
    assert len(completed) == 1


# ── Sync handler parity ──────────────────────────────────────────────────────


def test_sync_llm_started_and_completed_use_sync_gate() -> None:
    gate = FakeGate()
    options = make_options(gate=gate)
    handler = OpenBoxLangChainCoreSyncCallbackHandler(options)

    run_id = uuid.uuid4()
    handler.on_chat_model_start({"name": "fake-model"}, make_messages(), run_id=run_id)
    handler.on_llm_end(AIMessage(content="done"), run_id=run_id)

    started = [e for e in gate.sent if e.payload.get("event_type") == "ActivityStarted"]
    completed = [e for e in gate.sent if e.payload.get("event_type") == "ActivityCompleted"]
    assert len(started) == 1
    assert len(completed) == 1
    assert started[0].activity_id == completed[0].activity_id == str(run_id)
