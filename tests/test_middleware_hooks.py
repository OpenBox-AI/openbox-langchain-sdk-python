"""Tests for middleware_hooks.py + middleware_message_extraction.py helpers."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from openbox_core.contracts.results import EvaluationResult, GuardrailsResult, Verdict
from openbox_core.errors import (
    ApprovalRejectedError,
    GovernanceBlockedError,
    GovernanceHaltError,
)

from openbox_langchain.middleware_hooks import (
    build_signal_received_envelope,
    build_workflow_completed_envelope,
    build_workflow_started_envelope,
    enforce_start_verdict_async,
    enforce_start_verdict_sync,
    evaluate_lifecycle_async,
    evaluate_lifecycle_sync,
)
from openbox_langchain.middleware_message_extraction import (
    extract_last_user_message,
    extract_response_metadata,
)
from openbox_langchain.middleware_turn_state import MiddlewareTurnState

WORKFLOW_TYPE = "TestAgent"


def make_turn(**overrides) -> MiddlewareTurnState:
    turn = MiddlewareTurnState.new(sync_mode=overrides.pop("sync_mode", False))
    for k, v in overrides.items():
        setattr(turn, k, v)
    return turn


def make_mw(*, gate=None, adapter=None) -> MagicMock:
    mw = MagicMock()
    mw._workflow_type = WORKFLOW_TYPE
    mw._options = MagicMock()
    mw._options.task_queue = "langchain"
    mw._options.session_id = None
    mw._options.agent_name = None
    mw._runtime = MagicMock()
    mw._runtime.gate = gate or MagicMock()
    mw._runtime.adapter = adapter or MagicMock()
    return mw


# ─── extract_last_user_message ──────────────────────────────────────────


class TestExtractLastUserMessage:
    def test_dict_message_with_user_role(self):
        messages = [{"role": "user", "content": "Hello world"}]
        assert extract_last_user_message(messages) == "Hello world"

    def test_dict_message_with_human_role(self):
        messages = [{"role": "human", "content": "Hello"}]
        assert extract_last_user_message(messages) == "Hello"

    def test_baseobject_with_human_type(self):
        msg = MagicMock()
        msg.type = "human"
        msg.content = "Test message"
        assert extract_last_user_message([msg]) == "Test message"

    def test_empty_message_list(self):
        assert extract_last_user_message([]) is None

    def test_no_user_messages(self):
        messages = [
            {"role": "assistant", "content": "Response"},
            {"role": "system", "content": "System prompt"},
        ]
        assert extract_last_user_message(messages) is None

    def test_last_user_message_wins(self):
        messages = [
            {"role": "user", "content": "First"},
            {"role": "assistant", "content": "Response"},
            {"role": "user", "content": "Second"},
        ]
        assert extract_last_user_message(messages) == "Second"

    def test_non_string_content_ignored(self):
        messages = [
            {"role": "user", "content": ["not", "a", "string"]},
            {"role": "user", "content": "Valid string"},
        ]
        assert extract_last_user_message(messages) == "Valid string"


# ─── extract_response_metadata ──────────────────────────────────────────


class TestExtractResponseMetadata:
    def test_extract_tokens(self):
        ai_msg = MagicMock()
        ai_msg.usage_metadata = {"input_tokens": 10, "output_tokens": 20}
        ai_msg.response_metadata = {"model_name": "gpt-4"}
        ai_msg.content = "Response"
        ai_msg.tool_calls = None
        response = MagicMock()
        response.message = ai_msg

        meta = extract_response_metadata(response)
        assert meta["input_tokens"] == 10
        assert meta["output_tokens"] == 20
        assert meta["total_tokens"] == 30
        assert meta["llm_model"] == "gpt-4"

    def test_extract_tokens_from_prompt_tokens(self):
        ai_msg = MagicMock()
        ai_msg.usage_metadata = {"prompt_tokens": 5, "completion_tokens": 15}
        ai_msg.response_metadata = {}
        ai_msg.content = "Response"
        ai_msg.tool_calls = None
        response = MagicMock()
        response.message = ai_msg

        meta = extract_response_metadata(response)
        assert meta["input_tokens"] == 5
        assert meta["output_tokens"] == 15
        assert meta["total_tokens"] == 20

    def test_extract_completion_multimodal(self):
        ai_msg = MagicMock()
        ai_msg.usage_metadata = {}
        ai_msg.response_metadata = {}
        ai_msg.content = [
            {"type": "text", "text": "Hello"},
            {"type": "image", "data": "..."},
            {"type": "text", "text": "World"},
        ]
        ai_msg.tool_calls = None
        response = MagicMock()
        response.message = ai_msg

        meta = extract_response_metadata(response)
        assert meta["completion"] == "Hello World"

    def test_extract_tool_calls(self):
        ai_msg = MagicMock()
        ai_msg.usage_metadata = {}
        ai_msg.response_metadata = {}
        ai_msg.content = "Call tool"
        ai_msg.tool_calls = [{"name": "search", "args": {}}]
        response = MagicMock()
        response.message = ai_msg

        meta = extract_response_metadata(response)
        assert meta["has_tool_calls"] is True

    def test_response_without_message_attribute(self):
        ai_msg = MagicMock()
        ai_msg.usage_metadata = {"input_tokens": 10}
        ai_msg.response_metadata = {"model_name": "gpt-4"}
        ai_msg.content = "Response"
        ai_msg.tool_calls = None
        del ai_msg.message

        meta = extract_response_metadata(ai_msg)
        assert meta.get("input_tokens") == 10
        assert meta.get("completion") == "Response"


# ─── Envelope builders ───────────────────────────────────────────────────


class TestEnvelopeBuilders:
    def test_workflow_started_envelope_shape(self):
        mw = make_mw()
        turn = make_turn()
        envelope = build_workflow_started_envelope(mw, turn, {"messages": []})
        assert envelope.event_type.value == "WorkflowStarted"
        assert envelope.payload["workflow_id"] == turn.workflow_id
        assert envelope.payload["run_id"] == turn.run_id
        assert envelope.payload["workflow_type"] == WORKFLOW_TYPE

    def test_workflow_completed_envelope_success(self):
        mw = make_mw()
        turn = make_turn()
        envelope = build_workflow_completed_envelope(mw, turn, workflow_output={"result": "ok"})
        assert envelope.event_type.value == "WorkflowCompleted"
        assert envelope.payload["status"] == "completed"
        assert envelope.payload["workflow_output"] == {"result": "ok"}

    def test_workflow_completed_envelope_error(self):
        mw = make_mw()
        turn = make_turn()
        envelope = build_workflow_completed_envelope(mw, turn, error="blocked")
        assert envelope.payload["status"] == "failed"
        assert envelope.payload["error"] == "blocked"

    def test_signal_received_envelope_shape(self):
        mw = make_mw()
        turn = make_turn()
        envelope = build_signal_received_envelope(mw, turn, "hello")
        assert envelope.event_type.value == "SignalReceived"
        assert envelope.payload["signal_name"] == "user_prompt"
        assert envelope.payload["signal_args"] == ["hello"]

    def test_session_id_and_agent_name_passthrough(self):
        mw = make_mw()
        mw._options.session_id = "sess-1"
        mw._options.agent_name = "Agent1"
        turn = make_turn()
        envelope = build_workflow_started_envelope(mw, turn, {})
        assert envelope.payload["session_id"] == "sess-1"
        assert envelope.payload["agent_name"] == "Agent1"


# ─── evaluate_lifecycle_async / sync ─────────────────────────────────────


async def test_evaluate_lifecycle_async_calls_gate_aevaluate():
    mw = make_mw()
    mw._runtime.gate.aevaluate = AsyncMock(return_value="verdict")
    event = MagicMock()
    result = await evaluate_lifecycle_async(mw, event)
    assert result == "verdict"
    mw._runtime.gate.aevaluate.assert_called_once_with(event)


def test_evaluate_lifecycle_sync_calls_gate_evaluate():
    mw = make_mw()
    mw._runtime.gate.evaluate = MagicMock(return_value="verdict")
    event = MagicMock()
    result = evaluate_lifecycle_sync(mw, event)
    assert result == "verdict"
    mw._runtime.gate.evaluate.assert_called_once_with(event)


# ─── enforce_start_verdict_async ──────────────────────────────────────────


async def test_enforce_start_verdict_async_allow_is_noop():
    mw = make_mw()
    result = EvaluationResult(verdict=Verdict.ALLOW)
    await enforce_start_verdict_async(mw, result)
    mw._runtime.adapter.raise_lifecycle_blocked.assert_not_called()
    mw._runtime.adapter.handle_approval.assert_not_called()


async def test_enforce_start_verdict_async_block_raises_via_adapter():
    mw = make_mw()
    mw._runtime.adapter.raise_lifecycle_blocked = MagicMock(
        side_effect=GovernanceBlockedError(Verdict.BLOCK, "no")
    )
    result = EvaluationResult(verdict=Verdict.BLOCK, reason="no")
    with pytest.raises(GovernanceBlockedError):
        await enforce_start_verdict_async(mw, result)
    mw._runtime.adapter.raise_lifecycle_blocked.assert_called_once_with(result)


async def test_enforce_start_verdict_async_halt_raises_via_adapter():
    mw = make_mw()
    mw._runtime.adapter.raise_lifecycle_blocked = MagicMock(
        side_effect=GovernanceHaltError("halted")
    )
    result = EvaluationResult(verdict=Verdict.HALT, reason="halted")
    with pytest.raises(GovernanceHaltError):
        await enforce_start_verdict_async(mw, result)


async def test_enforce_start_verdict_async_require_approval_awaits_adapter():
    mw = make_mw()
    mw._runtime.adapter.handle_approval = AsyncMock()
    result = EvaluationResult(verdict=Verdict.REQUIRE_APPROVAL, reason="ask")
    await enforce_start_verdict_async(mw, result)
    mw._runtime.adapter.handle_approval.assert_awaited_once_with(result)


async def test_enforce_start_verdict_async_require_approval_propagates_rejection():
    mw = make_mw()
    mw._runtime.adapter.handle_approval = AsyncMock(
        side_effect=ApprovalRejectedError("rejected")
    )
    result = EvaluationResult(verdict=Verdict.REQUIRE_APPROVAL)
    with pytest.raises(ApprovalRejectedError):
        await enforce_start_verdict_async(mw, result)


# ─── enforce_start_verdict_sync (M15 fail-shut) ──────────────────────────


def test_enforce_start_verdict_sync_allow_is_noop():
    mw = make_mw()
    result = EvaluationResult(verdict=Verdict.ALLOW)
    enforce_start_verdict_sync(mw, result)
    mw._runtime.adapter.raise_lifecycle_blocked.assert_not_called()


def test_enforce_start_verdict_sync_block_raises_via_adapter():
    mw = make_mw()
    mw._runtime.adapter.raise_lifecycle_blocked = MagicMock(
        side_effect=GovernanceBlockedError(Verdict.BLOCK, "no")
    )
    result = EvaluationResult(verdict=Verdict.BLOCK, reason="no")
    with pytest.raises(GovernanceBlockedError):
        enforce_start_verdict_sync(mw, result)


def test_enforce_start_verdict_sync_require_approval_fails_shut_without_handle_approval_sync():
    """M15: no handle_approval_sync on the adapter -> ApprovalRejectedError, not a real wait."""
    mw = make_mw()
    # spec= constrains the mock's attrs so getattr(..., "handle_approval_sync", None)
    # returns None exactly like the real CoreAdapter (no such method defined).
    mw._runtime.adapter = MagicMock(spec=["raise_lifecycle_blocked", "handle_approval"])
    result = EvaluationResult(verdict=Verdict.REQUIRE_APPROVAL, reason="ask")
    with pytest.raises(ApprovalRejectedError):
        enforce_start_verdict_sync(mw, result)


def test_enforce_start_verdict_sync_require_approval_uses_handle_approval_sync_when_present():
    """When the adapter DOES define handle_approval_sync, the sync path drives it."""
    mw = make_mw()
    mw._runtime.adapter.handle_approval_sync = MagicMock()
    result = EvaluationResult(verdict=Verdict.REQUIRE_APPROVAL, reason="ask")
    enforce_start_verdict_sync(mw, result)
    mw._runtime.adapter.handle_approval_sync.assert_called_once_with(result)


def test_enforce_start_verdict_sync_guardrails_redacted_input_not_enforced_here():
    """enforce_start_verdict_sync only inspects .verdict — guardrails redaction
    is applied by the caller (wrap_model_call), not enforced/raised here."""
    mw = make_mw()
    guardrails = GuardrailsResult(redacted_input="[REDACTED]", input_type="activity_input")
    result = EvaluationResult(verdict=Verdict.ALLOW, guardrails=guardrails)
    enforce_start_verdict_sync(mw, result)  # must not raise
    mw._runtime.adapter.raise_lifecycle_blocked.assert_not_called()
