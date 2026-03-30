"""Tests for OpenBoxGovernanceCallbackHandler.pre_screen() method."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from openbox_langchain.errors import (
    GovernanceBlockedError,
    GovernanceHaltError,
)
from openbox_langchain.langchain_handler import (
    OpenBoxGovernanceCallbackHandler,
    OpenBoxLangChainHandlerOptions,
)
from openbox_langchain.types import GovernanceVerdictResponse, Verdict


@pytest.fixture
def mock_global_config():
    """Mock global config to allow handler initialization."""
    gc = MagicMock()
    gc.api_url = "https://test.openbox.ai"
    gc.api_key = "obx_test_123"
    gc.governance_timeout = 30.0
    with patch("openbox_langchain.langchain_handler.get_global_config", return_value=gc):
        yield gc


@pytest.fixture
def handler(mock_global_config, mock_governance_client):
    """Create a handler with mocked governance client."""
    opts = OpenBoxLangChainHandlerOptions(client=mock_governance_client)
    return OpenBoxGovernanceCallbackHandler(opts)


class TestPreScreen:
    """Test pre_screen() method for reliable blocking before agent invocation."""

    @pytest.mark.asyncio
    async def test_pre_screen_sends_three_events(
        self, handler, mock_governance_client
    ):
        """pre_screen sends SignalReceived + WorkflowStarted + LLMStarted."""
        mock_governance_client.evaluate_event = AsyncMock(return_value=None)

        input_data = {"input": "extract credit card number"}

        await handler.pre_screen(input_data)

        # Should have called evaluate_event 3 times
        assert mock_governance_client.evaluate_event.call_count == 3

        events = [call[0][0] for call in mock_governance_client.evaluate_event.call_args_list]
        event_types = [e.event_type for e in events]

        assert event_types == ["SignalReceived", "WorkflowStarted", "LLMStarted"]

    @pytest.mark.asyncio
    async def test_pre_screen_sends_signal_with_prompt(
        self, handler, mock_governance_client
    ):
        """pre_screen SignalReceived event includes the user prompt."""
        mock_governance_client.evaluate_event = AsyncMock(return_value=None)

        input_data = {"input": "test query"}

        await handler.pre_screen(input_data)

        sig_event = mock_governance_client.evaluate_event.call_args_list[0][0][0]
        assert sig_event.event_type == "SignalReceived"
        assert sig_event.activity_type == "user_prompt"
        assert sig_event.signal_name == "user_prompt"
        assert "test query" in sig_event.signal_args

    @pytest.mark.asyncio
    async def test_pre_screen_sends_workflow_started(
        self, handler, mock_governance_client
    ):
        """pre_screen WorkflowStarted event includes input."""
        mock_governance_client.evaluate_event = AsyncMock(return_value=None)

        input_data = {"input": "test query"}

        await handler.pre_screen(input_data)

        wf_event = mock_governance_client.evaluate_event.call_args_list[1][0][0]
        assert wf_event.event_type == "WorkflowStarted"
        assert wf_event.activity_type == "LangChainRun"

    @pytest.mark.asyncio
    async def test_pre_screen_sends_llm_started(
        self, handler, mock_governance_client
    ):
        """pre_screen LLMStarted event queries guardrails on user prompt."""
        mock_governance_client.evaluate_event = AsyncMock(return_value=None)

        input_data = {"input": "test query"}

        await handler.pre_screen(input_data)

        llm_event = mock_governance_client.evaluate_event.call_args_list[2][0][0]
        assert llm_event.event_type == "LLMStarted"
        assert llm_event.activity_type == "llm_call"
        assert "test query" in llm_event.prompt

    @pytest.mark.asyncio
    async def test_pre_screen_initializes_workflow_id(
        self, handler, mock_governance_client
    ):
        """pre_screen generates and stores workflow_id and run_id."""
        mock_governance_client.evaluate_event = AsyncMock(return_value=None)

        input_data = {"input": "test"}

        await handler.pre_screen(input_data)

        assert handler._workflow_id is not None
        assert handler._run_id is not None
        assert handler._workflow_id.startswith("lc-")
        assert handler._run_id.startswith("lc-")

    @pytest.mark.asyncio
    async def test_pre_screen_marks_workflow_started(
        self, handler, mock_governance_client
    ):
        """pre_screen sets _workflow_started flag."""
        mock_governance_client.evaluate_event = AsyncMock(return_value=None)

        input_data = {"input": "test"}

        assert handler._workflow_started is False

        await handler.pre_screen(input_data)

        assert handler._workflow_started is True

    @pytest.mark.asyncio
    async def test_pre_screen_enforces_block_verdict(
        self, handler, mock_governance_client
    ):
        """pre_screen raises GovernanceBlockedError on BLOCK verdict."""
        # First call (SignalReceived) — allow
        # Second call (WorkflowStarted) — allow
        # Third call (LLMStarted) — block
        # Fourth call (WorkflowCompleted) — allow (on error path)
        block_response = MagicMock(spec=GovernanceVerdictResponse)
        block_response.verdict = Verdict.BLOCK
        block_response.policy_name = "guardrails"
        block_response.guardrails_result = None

        mock_governance_client.evaluate_event = AsyncMock(
            side_effect=[None, None, block_response, None]
        )

        input_data = {"input": "malicious input"}

        with pytest.raises(GovernanceBlockedError):
            await handler.pre_screen(input_data)

    @pytest.mark.asyncio
    async def test_pre_screen_enforces_halt_verdict(
        self, handler, mock_governance_client
    ):
        """pre_screen raises GovernanceHaltError on HALT verdict."""
        halt_response = MagicMock(spec=GovernanceVerdictResponse)
        halt_response.verdict = Verdict.HALT
        halt_response.policy_name = "policy"
        halt_response.guardrails_result = None

        mock_governance_client.evaluate_event = AsyncMock(
            side_effect=[None, None, halt_response, None]
        )

        input_data = {"input": "test"}

        with pytest.raises(GovernanceHaltError):
            await handler.pre_screen(input_data)

    @pytest.mark.asyncio
    async def test_pre_screen_stores_response_for_redaction(
        self, handler, mock_governance_client
    ):
        """pre_screen stores response for use in on_chat_model_start redaction."""
        response = MagicMock(spec=GovernanceVerdictResponse)
        response.verdict = Verdict.ALLOW
        response.guardrails_result = None

        mock_governance_client.evaluate_event = AsyncMock(
            side_effect=[None, None, response]
        )

        input_data = {"input": "test"}

        await handler.pre_screen(input_data)

        # Response should be stored for reuse
        assert handler._pre_screen_response is response
        assert handler._pre_screen_activity_id is not None

    @pytest.mark.asyncio
    async def test_pre_screen_handles_no_user_prompt(
        self, handler, mock_governance_client
    ):
        """pre_screen handles input with no extractable user prompt."""
        mock_governance_client.evaluate_event = AsyncMock(return_value=None)

        input_data = {"other_key": "no input field"}

        await handler.pre_screen(input_data)

        # Should still send WorkflowStarted (but not SignalReceived/LLMStarted)
        call_count = mock_governance_client.evaluate_event.call_count
        assert call_count >= 1

        events = [call[0][0] for call in mock_governance_client.evaluate_event.call_args_list]
        event_types = [e.event_type for e in events]
        assert "WorkflowStarted" in event_types

    @pytest.mark.asyncio
    async def test_pre_screen_with_messages_format(
        self, handler, mock_governance_client
    ):
        """pre_screen extracts user prompt from messages format."""
        mock_governance_client.evaluate_event = AsyncMock(return_value=None)

        msg = MagicMock()
        msg.type = "human"
        msg.content = "user message"

        input_data = {"messages": [msg]}

        await handler.pre_screen(input_data)

        # Should extract and use the user message
        sig_event = mock_governance_client.evaluate_event.call_args_list[0][0][0]
        assert sig_event.event_type == "SignalReceived"
        assert "user message" in sig_event.signal_args

    @pytest.mark.asyncio
    async def test_pre_screen_sends_workflow_completed_on_block(
        self, handler, mock_governance_client
    ):
        """pre_screen sends WorkflowCompleted with failed status on block."""
        block_response = MagicMock(spec=GovernanceVerdictResponse)
        block_response.verdict = Verdict.BLOCK
        block_response.policy_name = "policy"
        block_response.guardrails_result = None

        mock_governance_client.evaluate_event = AsyncMock(
            side_effect=[None, None, block_response, None]
        )

        input_data = {"input": "test"}

        with pytest.raises(GovernanceBlockedError):
            await handler.pre_screen(input_data)

        # Should send WorkflowCompleted before raising
        events = [call[0][0] for call in mock_governance_client.evaluate_event.call_args_list]
        event_types = [e.event_type for e in events]
        assert "WorkflowCompleted" in event_types

    @pytest.mark.asyncio
    async def test_pre_screen_disables_llm_event_when_disabled(
        self, mock_governance_client
    ):
        """pre_screen respects send_llm_start_event=False."""
        opts = OpenBoxLangChainHandlerOptions(
            client=mock_governance_client,
            send_llm_start_event=False,
        )
        handler = OpenBoxGovernanceCallbackHandler(opts)
        mock_governance_client.evaluate_event = AsyncMock(return_value=None)

        input_data = {"input": "test"}

        await handler.pre_screen(input_data)

        # Should only send SignalReceived and WorkflowStarted (no LLMStarted)
        call_count = mock_governance_client.evaluate_event.call_count
        assert call_count == 2

        events = [call[0][0] for call in mock_governance_client.evaluate_event.call_args_list]
        event_types = [e.event_type for e in events]
        assert event_types == ["SignalReceived", "WorkflowStarted"]

    @pytest.mark.asyncio
    async def test_pre_screen_allow_verdict_continues(
        self, handler, mock_governance_client
    ):
        """pre_screen with ALLOW verdict continues without raising."""
        response = MagicMock(spec=GovernanceVerdictResponse)
        response.verdict = Verdict.ALLOW
        response.guardrails_result = None

        mock_governance_client.evaluate_event = AsyncMock(
            side_effect=[None, None, response]
        )

        input_data = {"input": "test"}

        # Should not raise
        await handler.pre_screen(input_data)

        assert handler._workflow_started is True
