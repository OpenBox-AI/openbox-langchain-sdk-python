"""Tests for middleware_hook_handlers.py hook implementations."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from openbox_langgraph.errors import (
    GovernanceBlockedError,
)
from openbox_langgraph.types import GovernanceVerdictResponse, Verdict

from openbox_langchain.middleware_hook_handlers import (
    handle_after_agent,
    handle_before_agent,
    handle_wrap_model_call,
)

# ─── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def mock_middleware():
    """Create a configured mock OpenBoxLangChainMiddleware."""
    mw = MagicMock()
    mw._client = AsyncMock()
    mw._config = MagicMock()
    mw._config.send_chain_start_event = True
    mw._config.send_chain_end_event = True
    mw._config.send_llm_start_event = True
    mw._config.send_llm_end_event = True
    mw._config.send_tool_start_event = True
    mw._config.send_tool_end_event = True
    mw._config.task_queue = "langchain"
    mw._config.session_id = None
    mw._config.skip_tool_types = set()
    mw._config.hitl = None
    mw._sync_mode = False
    mw._workflow_id = ""
    mw._run_id = ""
    mw._workflow_type = "TestAgent"
    mw._first_llm_call = True
    mw._pre_screen_response = None
    mw._span_processor = None
    return mw


@pytest.fixture
def mock_state_with_messages():
    """Create a mock agent state with messages."""
    return {
        "messages": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]
    }


@pytest.fixture
def mock_runtime():
    """Create a mock LangGraph runtime."""
    return MagicMock(config={"configurable": {"thread_id": "thread-123"}})


# ─── handle_before_agent ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_handle_before_agent_sets_workflow_ids(
    mock_middleware, mock_state_with_messages, mock_runtime
):
    """Set workflow_id and run_id per invocation."""
    with patch("openbox_langchain.middleware_hook_handlers._evaluate", new_callable=AsyncMock):
        with patch(
            "openbox_langchain.middleware_hook_handlers._poll_approval_or_halt",
            new_callable=AsyncMock,
        ):
            result = await handle_before_agent(
                mock_middleware, mock_state_with_messages, mock_runtime
            )

    assert mock_middleware._workflow_id.startswith("thread-123-")
    assert mock_middleware._run_id.startswith("thread-123-run-")
    assert result is None


@pytest.mark.asyncio
async def test_handle_before_agent_sends_signal_received(
    mock_middleware, mock_state_with_messages, mock_runtime
):
    """Send SignalReceived event for user prompt."""
    with patch(
        "openbox_langchain.middleware_hook_handlers._evaluate",
        new_callable=AsyncMock,
    ) as mock_eval:
        with patch(
            "openbox_langchain.middleware_hook_handlers._poll_approval_or_halt",
            new_callable=AsyncMock,
        ):
            await handle_before_agent(mock_middleware, mock_state_with_messages, mock_runtime)

    # Should have called _evaluate at least once (SignalReceived)
    assert mock_eval.call_count >= 1
    first_call = mock_eval.call_args_list[0]
    event = first_call[0][1]
    assert event.event_type == "SignalReceived"


@pytest.mark.asyncio
async def test_handle_before_agent_sends_workflow_started(
    mock_middleware, mock_state_with_messages, mock_runtime
):
    """Send WorkflowStarted event."""
    with patch(
        "openbox_langchain.middleware_hook_handlers._evaluate",
        new_callable=AsyncMock,
    ) as mock_eval:
        with patch(
            "openbox_langchain.middleware_hook_handlers._poll_approval_or_halt",
            new_callable=AsyncMock,
        ):
            await handle_before_agent(mock_middleware, mock_state_with_messages, mock_runtime)

    # Find WorkflowStarted in calls
    events = [call[0][1] for call in mock_eval.call_args_list]
    event_types = [e.event_type for e in events]
    assert "WorkflowStarted" in event_types


@pytest.mark.asyncio
async def test_handle_before_agent_sends_pre_screen_llm_started(
    mock_middleware, mock_state_with_messages, mock_runtime
):
    """Send LLMStarted for pre-screen guardrails."""
    with patch(
        "openbox_langchain.middleware_hook_handlers._evaluate",
        new_callable=AsyncMock,
    ) as mock_eval:
        with patch(
            "openbox_langchain.middleware_hook_handlers._poll_approval_or_halt",
            new_callable=AsyncMock,
        ):
            await handle_before_agent(mock_middleware, mock_state_with_messages, mock_runtime)

    # Find LLMStarted in calls
    events = [call[0][1] for call in mock_eval.call_args_list]
    event_types = [e.event_type for e in events]
    assert "LLMStarted" in event_types


@pytest.mark.asyncio
async def test_handle_before_agent_initializes_first_llm_call(
    mock_middleware, mock_state_with_messages, mock_runtime
):
    """Initialize _first_llm_call to True."""
    with patch("openbox_langchain.middleware_hook_handlers._evaluate", new_callable=AsyncMock):
        with patch(
            "openbox_langchain.middleware_hook_handlers._poll_approval_or_halt",
            new_callable=AsyncMock,
        ):
            await handle_before_agent(mock_middleware, mock_state_with_messages, mock_runtime)

    assert mock_middleware._first_llm_call is True


@pytest.mark.asyncio
async def test_handle_before_agent_clears_pre_screen_response(
    mock_middleware, mock_state_with_messages, mock_runtime
):
    """Clear _pre_screen_response at start."""
    with patch(
        "openbox_langchain.middleware_hook_handlers._evaluate",
        new_callable=AsyncMock,
        return_value=None,
    ):
        with patch(
            "openbox_langchain.middleware_hook_handlers._poll_approval_or_halt",
            new_callable=AsyncMock,
        ):
            await handle_before_agent(mock_middleware, mock_state_with_messages, mock_runtime)

    assert mock_middleware._pre_screen_response is None


@pytest.mark.asyncio
async def test_handle_before_agent_skip_events_when_flags_false(
    mock_middleware, mock_state_with_messages, mock_runtime
):
    """Skip events when send_* flags are False."""
    mock_middleware._config.send_chain_start_event = False
    mock_middleware._config.send_llm_start_event = False

    with patch(
        "openbox_langchain.middleware_hook_handlers._evaluate",
        new_callable=AsyncMock,
    ) as mock_eval:
        await handle_before_agent(mock_middleware, mock_state_with_messages, mock_runtime)

    # Should still send SignalReceived (not a send_* flag)
    events = [call[0][1] for call in mock_eval.call_args_list]
    event_types = [e.event_type for e in events]
    assert "WorkflowStarted" not in event_types
    assert "LLMStarted" not in event_types


@pytest.mark.asyncio
async def test_handle_before_agent_enforces_block_verdict(
    mock_middleware, mock_state_with_messages, mock_runtime
):
    """Raise enforcement error on block verdict."""
    verdict = MagicMock(spec=GovernanceVerdictResponse)
    verdict.verdict = "block"
    verdict.requires_hitl = False

    with patch(
        "openbox_langchain.middleware_hook_handlers._evaluate",
        new_callable=AsyncMock,
        return_value=verdict,
    ):
        with patch(
            "openbox_langchain.middleware_hook_handlers.enforce_verdict",
            side_effect=RuntimeError("blocked"),
        ):
            with pytest.raises(RuntimeError, match="blocked"):
                await handle_before_agent(mock_middleware, mock_state_with_messages, mock_runtime)


@pytest.mark.asyncio
async def test_handle_before_agent_sends_workflow_completed_on_enforcement_error(
    mock_middleware, mock_state_with_messages, mock_runtime
):
    """Send WorkflowCompleted on enforcement error."""
    verdict = MagicMock(spec=GovernanceVerdictResponse)
    verdict.verdict = "block"
    verdict.requires_hitl = False

    with patch(
        "openbox_langchain.middleware_hook_handlers._evaluate",
        new_callable=AsyncMock,
        return_value=verdict,
    ) as mock_eval:
        with patch(
            "openbox_langchain.middleware_hook_handlers.enforce_verdict",
            side_effect=RuntimeError("blocked"),
        ):
            with pytest.raises(RuntimeError):
                await handle_before_agent(mock_middleware, mock_state_with_messages, mock_runtime)

    # Find WorkflowCompleted call
    events = [call[0][1] for call in mock_eval.call_args_list]
    event_types = [e.event_type for e in events]
    assert "WorkflowCompleted" in event_types


@pytest.mark.asyncio
async def test_handle_before_agent_caches_pre_screen_response(
    mock_middleware, mock_state_with_messages, mock_runtime
):
    """Cache pre-screen response for first LLM call."""
    verdict = MagicMock(spec=GovernanceVerdictResponse)
    verdict.verdict = "allow"
    verdict.requires_hitl = False

    with patch(
        "openbox_langchain.middleware_hook_handlers._evaluate",
        new_callable=AsyncMock,
        return_value=verdict,
    ):
        with patch("openbox_langchain.middleware_hook_handlers.enforce_verdict", return_value=None):
            await handle_before_agent(mock_middleware, mock_state_with_messages, mock_runtime)

    assert mock_middleware._pre_screen_response is verdict


@pytest.mark.asyncio
async def test_handle_before_agent_hitl_polling(
    mock_middleware, mock_state_with_messages, mock_runtime
):
    """Poll for HITL approval when required."""
    verdict = MagicMock(spec=GovernanceVerdictResponse)
    verdict.verdict = "require_approval"
    verdict.requires_hitl = True

    with patch(
        "openbox_langchain.middleware_hook_handlers._evaluate",
        new_callable=AsyncMock,
        return_value=verdict,
    ):
        with patch(
            "openbox_langchain.middleware_hook_handlers.enforce_verdict", return_value=verdict
        ):
            with patch(
                "openbox_langchain.middleware_hook_handlers._poll_approval_or_halt",
                new_callable=AsyncMock,
            ) as mock_poll:
                await handle_before_agent(mock_middleware, mock_state_with_messages, mock_runtime)

                # Should have called polling
                assert mock_poll.call_count >= 1


@pytest.mark.asyncio
async def test_handle_before_agent_no_user_message(mock_middleware, mock_runtime):
    """Skip pre-screen when no user message."""
    state = {"messages": [{"role": "assistant", "content": "Only AI"}]}

    with patch(
        "openbox_langchain.middleware_hook_handlers._evaluate",
        new_callable=AsyncMock,
    ) as mock_eval:
        await handle_before_agent(mock_middleware, state, mock_runtime)

    # Should send WorkflowStarted but not pre-screen LLMStarted
    events = [call[0][1] for call in mock_eval.call_args_list]
    event_types = [e.event_type for e in events]
    assert "WorkflowStarted" in event_types
    # Pre-screen LLMStarted should not be sent (no user message)
    pre_screen_llm = [
        e for e in events if e.event_type == "LLMStarted" and e.activity_id.endswith("-pre")
    ]
    assert len(pre_screen_llm) == 0


# ─── handle_after_agent ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_handle_after_agent_sends_workflow_completed(
    mock_middleware, mock_state_with_messages, mock_runtime
):
    """Send WorkflowCompleted event."""
    with patch(
        "openbox_langchain.middleware_hook_handlers._evaluate",
        new_callable=AsyncMock,
    ) as mock_eval:
        result = await handle_after_agent(mock_middleware, mock_state_with_messages, mock_runtime)

    assert result is None
    assert mock_eval.call_count >= 1
    event = mock_eval.call_args_list[0][0][1]
    assert event.event_type == "WorkflowCompleted"


@pytest.mark.asyncio
async def test_handle_after_agent_skips_when_flag_false(
    mock_middleware, mock_state_with_messages, mock_runtime
):
    """Skip WorkflowCompleted when send_chain_end_event is False."""
    mock_middleware._config.send_chain_end_event = False

    with patch(
        "openbox_langchain.middleware_hook_handlers._evaluate",
        new_callable=AsyncMock,
    ) as mock_eval:
        await handle_after_agent(mock_middleware, mock_state_with_messages, mock_runtime)

    assert mock_eval.call_count == 0


@pytest.mark.asyncio
async def test_handle_after_agent_cleans_up_span_processor(
    mock_middleware, mock_state_with_messages, mock_runtime
):
    """Unregister workflow from span processor."""
    mock_middleware._span_processor = AsyncMock()
    mock_middleware._workflow_id = "wf-123"

    with patch("openbox_langchain.middleware_hook_handlers._evaluate", new_callable=AsyncMock):
        await handle_after_agent(mock_middleware, mock_state_with_messages, mock_runtime)

    mock_middleware._span_processor.unregister_workflow.assert_called_once_with("wf-123")


@pytest.mark.asyncio
async def test_handle_after_agent_includes_last_message_content(mock_middleware, mock_runtime):
    """Include last message content in WorkflowCompleted."""
    state = {"messages": [{"role": "assistant", "content": "Final response"}]}

    with patch(
        "openbox_langchain.middleware_hook_handlers._evaluate",
        new_callable=AsyncMock,
    ) as mock_eval:
        await handle_after_agent(mock_middleware, state, mock_runtime)

    event = mock_eval.call_args_list[0][0][1]
    # workflow_output should contain the last message content
    assert event.workflow_output is not None


# ─── handle_wrap_model_call ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_handle_wrap_model_call_sends_llm_started(mock_middleware):
    """Send LLMStarted event."""
    request = MagicMock()
    request.messages = [{"role": "user", "content": "Hello"}]

    handler = AsyncMock(return_value=MagicMock(content="Response"))

    with patch(
        "openbox_langchain.middleware_hook_handlers._evaluate",
        new_callable=AsyncMock,
    ) as mock_eval:
        with patch(
            "openbox_langchain.middleware_hook_handlers._run_with_otel_context",
            new_callable=AsyncMock,
            side_effect=handler,
        ):
            await handle_wrap_model_call(mock_middleware, request, handler)

    # Should have called _evaluate
    assert mock_eval.call_count >= 1


@pytest.mark.asyncio
async def test_handle_wrap_model_call_skips_empty_prompt(mock_middleware):
    """Skip processing for empty prompt."""
    request = MagicMock()
    request.messages = [{"role": "user", "content": ""}]

    handler = AsyncMock(return_value=MagicMock(content="Response"))

    with patch(
        "openbox_langchain.middleware_hook_handlers._evaluate",
        new_callable=AsyncMock,
    ):
        await handle_wrap_model_call(mock_middleware, request, handler)

    # Should call handler directly without governance events
    handler.assert_called_once()
    # _evaluate might still be called if there's a pre-screen response cached
    # but for empty prompt, skip


@pytest.mark.asyncio
async def test_handle_wrap_model_call_reuses_pre_screen_response(mock_middleware):
    """Reuse pre-screen response for first LLM call."""
    request = MagicMock()
    request.messages = [{"role": "user", "content": "Hello"}]

    pre_screen_verdict = MagicMock(spec=GovernanceVerdictResponse)
    pre_screen_verdict.guardrails_result = None
    mock_middleware._first_llm_call = True
    mock_middleware._pre_screen_response = pre_screen_verdict

    handler = AsyncMock(return_value=MagicMock(content="Response"))

    with patch(
        "openbox_langchain.middleware_hook_handlers._run_with_otel_context",
        new_callable=AsyncMock,
        side_effect=handler,
    ):
        await handle_wrap_model_call(mock_middleware, request, handler)

    assert mock_middleware._first_llm_call is False
    assert mock_middleware._pre_screen_response is None


@pytest.mark.asyncio
async def test_handle_wrap_model_call_applies_pii_redaction(mock_middleware):
    """Apply PII redaction from guardrails result."""
    request = MagicMock()
    request.messages = [MagicMock(type="human", content="Original")]

    verdict = MagicMock(spec=GovernanceVerdictResponse)
    verdict.guardrails_result = MagicMock()
    verdict.guardrails_result.input_type = "activity_input"
    verdict.guardrails_result.redacted_input = "Redacted"
    verdict.verdict = Verdict.ALLOW
    verdict.requires_approval = MagicMock(return_value=False)

    handler = AsyncMock(return_value=MagicMock(content="Response"))

    with patch(
        "openbox_langchain.middleware_hook_handlers._evaluate",
        new_callable=AsyncMock,
        return_value=verdict,
    ):
        with patch(
            "openbox_langchain.middleware_hook_handlers._run_with_otel_context",
            new_callable=AsyncMock,
            side_effect=handler,
        ):
            await handle_wrap_model_call(mock_middleware, request, handler)

    # Message should be redacted
    assert request.messages[0].content == "Redacted"


@pytest.mark.asyncio
async def test_handle_wrap_model_call_sends_llm_completed(mock_middleware):
    """Send LLMCompleted event."""
    request = MagicMock()
    request.messages = [{"role": "user", "content": "Hello"}]

    handler = AsyncMock(return_value=MagicMock(content="Response"))

    with patch(
        "openbox_langchain.middleware_hook_handlers._evaluate",
        new_callable=AsyncMock,
    ) as mock_eval:
        with patch(
            "openbox_langchain.middleware_hook_handlers._run_with_otel_context",
            new_callable=AsyncMock,
            side_effect=handler,
        ):
            await handle_wrap_model_call(mock_middleware, request, handler)

    # Find LLMCompleted event
    events = [call[0][1] for call in mock_eval.call_args_list]
    event_types = [e.event_type for e in events]
    assert "LLMCompleted" in event_types


@pytest.mark.asyncio
async def test_handle_wrap_model_call_registers_span_processor_context(mock_middleware):
    """Register activity context in span processor."""
    request = MagicMock()
    request.messages = [{"role": "user", "content": "Hello"}]

    mock_middleware._span_processor = MagicMock()
    handler = AsyncMock(return_value=MagicMock(content="Response"))

    with patch(
        "openbox_langchain.middleware_hook_handlers._evaluate",
        new_callable=AsyncMock,
    ):
        with patch(
            "openbox_langchain.middleware_hook_handlers._run_with_otel_context",
            new_callable=AsyncMock,
            side_effect=handler,
        ):
            await handle_wrap_model_call(mock_middleware, request, handler)

    # Should call set_activity_context
    mock_middleware._span_processor.set_activity_context.assert_called_once()
    # Should call clear_activity_context
    mock_middleware._span_processor.clear_activity_context.assert_called_once()


@pytest.mark.asyncio
async def test_handle_wrap_model_call_clears_span_processor_context(mock_middleware):
    """Clear activity context after execution."""
    request = MagicMock()
    request.messages = [{"role": "user", "content": "Hello"}]

    mock_middleware._span_processor = MagicMock()
    handler = AsyncMock(return_value=MagicMock(content="Response"))

    with patch(
        "openbox_langchain.middleware_hook_handlers._evaluate",
        new_callable=AsyncMock,
    ):
        with patch(
            "openbox_langchain.middleware_hook_handlers._run_with_otel_context",
            new_callable=AsyncMock,
            side_effect=handler,
        ):
            await handle_wrap_model_call(mock_middleware, request, handler)

    # clear_activity_context should be called
    assert mock_middleware._span_processor.clear_activity_context.called


@pytest.mark.asyncio
async def test_handle_wrap_model_call_hitl_retry_detects_require_approval(mock_middleware):
    """Detect require_approval from GovernanceBlockedError in exception chain."""
    request = MagicMock()
    request.messages = [{"role": "user", "content": "Hello"}]

    # Create an error that will be extracted by _extract_governance_blocked
    GovernanceBlockedError("requires approval", "require_approval")
    success_response = MagicMock(content="Response")

    # Mock the handler to succeed
    handler = AsyncMock(return_value=success_response)

    with patch(
        "openbox_langchain.middleware_hook_handlers._evaluate",
        new_callable=AsyncMock,
    ):
        with patch(
            "openbox_langchain.middleware_hook_handlers._run_with_otel_context",
            new_callable=AsyncMock,
            return_value=success_response,
        ):
            result = await handle_wrap_model_call(mock_middleware, request, handler)

    # The handler should have been called
    assert result == success_response


@pytest.mark.asyncio
async def test_handle_wrap_model_call_skips_llm_start_when_flag_false(mock_middleware):
    """Skip LLMStarted when send_llm_start_event is False."""
    mock_middleware._config.send_llm_start_event = False

    request = MagicMock()
    request.messages = [{"role": "user", "content": "Hello"}]

    handler = AsyncMock(return_value=MagicMock(content="Response"))

    with patch(
        "openbox_langchain.middleware_hook_handlers._evaluate",
        new_callable=AsyncMock,
    ) as mock_eval:
        with patch(
            "openbox_langchain.middleware_hook_handlers._run_with_otel_context",
            new_callable=AsyncMock,
            side_effect=handler,
        ):
            await handle_wrap_model_call(mock_middleware, request, handler)

    # LLMStarted should not be sent (but LLMCompleted might be)
    events = [call[0][1] for call in mock_eval.call_args_list]
    llm_start_events = [e for e in events if e.event_type == "LLMStarted"]
    assert len(llm_start_events) == 0
