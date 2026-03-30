"""Tests for OpenBoxGovernanceCallbackHandler — core callback handler functionality."""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from openbox_langchain.langchain_handler import (
    OpenBoxGovernanceCallbackHandler,
    OpenBoxLangChainHandlerOptions,
)
from openbox_langchain.types import GovernanceVerdictResponse


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


@pytest.fixture
def root_run_id():
    """Generate a UUID for testing."""
    return uuid.uuid4()


@pytest.fixture
def child_run_id():
    """Generate a different UUID for nested runs."""
    return uuid.uuid4()


class TestChainCallbacks:
    """Test on_chain_start and on_chain_end behavior."""

    @pytest.mark.asyncio
    async def test_on_chain_start_root_detection(
        self, handler, mock_governance_client, root_run_id
    ):
        """First chain_start with parent_run_id=None sets root_run_id."""
        serialized = {"name": "AgentExecutor", "id": ["langchain", "agents", "AgentExecutor"]}
        inputs = {"input": "test query"}

        await handler.on_chain_start(
            serialized=serialized,
            inputs=inputs,
            run_id=root_run_id,
            parent_run_id=None,
        )

        assert handler._root_run_id == str(root_run_id)
        assert handler._buffer.get(str(root_run_id)) is not None

    @pytest.mark.asyncio
    async def test_on_chain_start_sends_workflow_started(
        self, handler, mock_governance_client, root_run_id
    ):
        """Root chain_start sends WorkflowStarted event with user prompt."""
        serialized = {"name": "AgentExecutor"}
        inputs = {"input": "test query"}
        mock_governance_client.evaluate_event = AsyncMock(return_value=None)

        await handler.on_chain_start(
            serialized=serialized,
            inputs=inputs,
            run_id=root_run_id,
            parent_run_id=None,
        )

        # Verify evaluate_event was called with WorkflowStarted event
        assert mock_governance_client.evaluate_event.called
        events = [call[0][0] for call in mock_governance_client.evaluate_event.call_args_list]

        # Should have WorkflowStarted and potentially LLMStarted (pre-screen)
        event_types = [e.event_type for e in events]
        assert "WorkflowStarted" in event_types

    @pytest.mark.asyncio
    async def test_on_chain_start_sends_signal_received(
        self, handler, mock_governance_client, root_run_id
    ):
        """Root chain_start sends SignalReceived with user prompt."""
        serialized = {"name": "AgentExecutor"}
        inputs = {"input": "extract credit card"}
        mock_governance_client.evaluate_event = AsyncMock(return_value=None)

        await handler.on_chain_start(
            serialized=serialized,
            inputs=inputs,
            run_id=root_run_id,
            parent_run_id=None,
        )

        events = [call[0][0] for call in mock_governance_client.evaluate_event.call_args_list]
        event_types = [e.event_type for e in events]
        assert "SignalReceived" in event_types

    @pytest.mark.asyncio
    async def test_on_chain_start_nested_sends_chain_started(
        self, handler, mock_governance_client, root_run_id, child_run_id
    ):
        """Nested chain_start sends ChainStarted event."""
        # Setup root
        handler._root_run_id = str(root_run_id)
        handler._workflow_id = "test-wf"
        handler._run_id = "test-run"
        handler._workflow_started = True

        mock_governance_client.evaluate_event = AsyncMock(return_value=None)

        serialized = {"name": "NestedChain"}
        inputs = {"key": "value"}

        await handler.on_chain_start(
            serialized=serialized,
            inputs=inputs,
            run_id=child_run_id,
            parent_run_id=root_run_id,
        )

        # Should send ChainStarted event
        call_args = mock_governance_client.evaluate_event.call_args_list[0][0][0]
        assert call_args.event_type == "ChainStarted"
        assert call_args.activity_type == "NestedChain"

    @pytest.mark.asyncio
    async def test_on_chain_end_root_sends_workflow_completed(
        self, handler, mock_governance_client, root_run_id
    ):
        """Root chain_end sends WorkflowCompleted event."""
        # Setup root state
        handler._root_run_id = str(root_run_id)
        handler._workflow_id = "test-wf"
        handler._run_id = "test-run"
        handler._workflow_started = True
        handler._buffer.register(str(root_run_id), "chain", "Root")

        mock_governance_client.evaluate_event = AsyncMock(return_value=None)

        outputs = {"output": "result"}

        await handler.on_chain_end(
            outputs=outputs,
            run_id=root_run_id,
            parent_run_id=None,
        )

        # Should send WorkflowCompleted
        call_args = mock_governance_client.evaluate_event.call_args_list[0][0][0]
        assert call_args.event_type == "WorkflowCompleted"
        assert call_args.status == "completed"

    @pytest.mark.asyncio
    async def test_on_chain_end_resets_state(
        self, handler, mock_governance_client, root_run_id
    ):
        """Root chain_end resets handler state for next invocation."""
        # Setup root state
        handler._root_run_id = str(root_run_id)
        handler._workflow_id = "test-wf"
        handler._run_id = "test-run"
        handler._workflow_started = True
        handler._buffer.register(str(root_run_id), "chain", "Root")
        mock_governance_client.evaluate_event = AsyncMock(return_value=None)

        await handler.on_chain_end(
            outputs={},
            run_id=root_run_id,
            parent_run_id=None,
        )

        # State should be reset
        assert handler._root_run_id is None
        assert handler._workflow_id is None
        assert handler._workflow_started is False
        assert len(handler._buffer._runs) == 0


class TestToolCallbacks:
    """Test on_tool_start and on_tool_end behavior."""

    @pytest.mark.asyncio
    async def test_on_tool_start_sends_tool_started(
        self, handler, mock_governance_client, root_run_id, child_run_id
    ):
        """on_tool_start sends ToolStarted event with tool name and input."""
        # Setup context
        handler._root_run_id = str(root_run_id)
        handler._workflow_id = "test-wf"
        handler._run_id = "test-run"
        handler._workflow_started = True

        mock_governance_client.evaluate_event = AsyncMock(return_value=None)

        serialized = {"name": "search_web"}
        input_str = '{"query": "test"}'

        await handler.on_tool_start(
            serialized=serialized,
            input_str=input_str,
            run_id=child_run_id,
        )

        call_args = mock_governance_client.evaluate_event.call_args_list[0][0][0]
        assert call_args.event_type == "ToolStarted"
        assert call_args.activity_type == "search_web"
        assert call_args.tool_name == "search_web"

    @pytest.mark.asyncio
    async def test_on_tool_start_skips_when_disabled(
        self, mock_governance_client, root_run_id, child_run_id
    ):
        """on_tool_start respects send_tool_start_event=False."""
        opts = OpenBoxLangChainHandlerOptions(
            client=mock_governance_client,
            send_tool_start_event=False,
        )
        handler = OpenBoxGovernanceCallbackHandler(opts)
        handler._root_run_id = str(root_run_id)
        handler._workflow_id = "test-wf"
        handler._run_id = "test-run"

        mock_governance_client.evaluate_event = AsyncMock(return_value=None)

        serialized = {"name": "search_web"}

        await handler.on_tool_start(
            serialized=serialized,
            input_str="{}",
            run_id=child_run_id,
        )

        # Should not call evaluate_event
        assert not mock_governance_client.evaluate_event.called

    @pytest.mark.asyncio
    async def test_on_tool_start_skips_tool_types(
        self, mock_governance_client, root_run_id, child_run_id
    ):
        """on_tool_start respects skip_tool_types config."""
        opts = OpenBoxLangChainHandlerOptions(
            client=mock_governance_client,
            skip_tool_types={"_deprecated"},
        )
        handler = OpenBoxGovernanceCallbackHandler(opts)
        handler._root_run_id = str(root_run_id)
        handler._workflow_id = "test-wf"
        handler._run_id = "test-run"
        handler._workflow_started = True

        mock_governance_client.evaluate_event = AsyncMock(return_value=None)

        serialized = {"name": "_deprecated"}

        await handler.on_tool_start(
            serialized=serialized,
            input_str="{}",
            run_id=child_run_id,
        )

        # Should not call evaluate_event for skipped tool
        assert not mock_governance_client.evaluate_event.called

    @pytest.mark.asyncio
    async def test_on_tool_end_sends_tool_completed(
        self, handler, mock_governance_client, child_run_id
    ):
        """on_tool_end sends ToolCompleted event with output and duration."""
        handler._root_run_id = "root"
        handler._workflow_id = "test-wf"
        handler._run_id = "test-run"
        handler._workflow_started = True
        handler._buffer.register(str(child_run_id), "tool", "search_web")

        mock_governance_client.evaluate_event = AsyncMock(return_value=None)

        await handler.on_tool_end(
            output="search results",
            run_id=child_run_id,
        )

        call_args = mock_governance_client.evaluate_event.call_args_list[0][0][0]
        assert call_args.event_type == "ToolCompleted"
        assert call_args.activity_type == "search_web"
        assert call_args.status == "completed"


class TestLLMCallbacks:
    """Test on_chat_model_start and on_llm_end behavior."""

    @pytest.mark.asyncio
    async def test_on_chat_model_start_sends_llm_started(
        self, handler, mock_governance_client, child_run_id
    ):
        """on_chat_model_start sends LLMStarted event with prompt."""
        handler._root_run_id = "root"
        handler._workflow_id = "test-wf"
        handler._run_id = "test-run"
        handler._workflow_started = True

        mock_governance_client.evaluate_event = AsyncMock(return_value=None)

        serialized = {"name": "ChatOpenAI"}
        msg = MagicMock()
        msg.type = "human"
        msg.content = "What is 2+2?"
        messages = [[msg]]

        await handler.on_chat_model_start(
            serialized=serialized,
            messages=messages,
            run_id=child_run_id,
        )

        # Should send LLMStarted event
        call_args = mock_governance_client.evaluate_event.call_args_list[0][0][0]
        assert call_args.event_type == "LLMStarted"
        assert call_args.activity_type == "llm_call"

    @pytest.mark.asyncio
    async def test_on_chat_model_start_skips_empty_prompt(
        self, handler, mock_governance_client, child_run_id
    ):
        """on_chat_model_start skips when no human messages present."""
        handler._root_run_id = "root"
        handler._workflow_id = "test-wf"
        handler._run_id = "test-run"
        handler._workflow_started = True

        mock_governance_client.evaluate_event = AsyncMock(return_value=None)

        serialized = {"name": "ChatOpenAI"}
        # AI message, not human
        msg = MagicMock()
        msg.type = "ai"
        msg.content = "Response"
        messages = [[msg]]

        await handler.on_chat_model_start(
            serialized=serialized,
            messages=messages,
            run_id=child_run_id,
        )

        # Should not send event for non-human messages
        assert not mock_governance_client.evaluate_event.called

    @pytest.mark.asyncio
    async def test_on_chat_model_start_reuses_pre_screen(
        self, handler, mock_governance_client, child_run_id
    ):
        """on_chat_model_start reuses pre-screen response on first LLM call."""
        handler._root_run_id = "root"
        handler._workflow_id = "test-wf"
        handler._run_id = "test-run"
        handler._workflow_started = True
        handler._pre_screen_response = MagicMock(spec=GovernanceVerdictResponse)
        handler._pre_screen_response.guardrails_result = None
        handler._pre_screen_activity_id = "pre-123"

        mock_governance_client.evaluate_event = AsyncMock(return_value=None)

        serialized = {"name": "ChatOpenAI"}
        msg = MagicMock()
        msg.type = "human"
        msg.content = "Query"
        messages = [[msg]]

        await handler.on_chat_model_start(
            serialized=serialized,
            messages=messages,
            run_id=child_run_id,
        )

        # Should consume pre_screen_response
        assert handler._pre_screen_response is None
        assert str(child_run_id) in handler._llm_activity_map

    @pytest.mark.asyncio
    async def test_on_llm_end_sends_llm_completed(
        self, handler, mock_governance_client, child_run_id
    ):
        """on_llm_end sends LLMCompleted observation."""
        handler._root_run_id = "root"
        handler._workflow_id = "test-wf"
        handler._run_id = "test-run"
        handler._llm_activity_map[str(child_run_id)] = str(child_run_id)

        mock_governance_client.evaluate_event = AsyncMock(return_value=None)

        # Mock LLMResult from LangChain
        gen = MagicMock()
        gen.text = "The answer is 4"
        response = MagicMock()
        response.generations = [[gen]]
        response.llm_output = {"token_usage": {"prompt_tokens": 10, "completion_tokens": 5}}

        await handler.on_llm_end(response=response, run_id=child_run_id)

        call_args = mock_governance_client.evaluate_event.call_args_list[0][0][0]
        assert call_args.event_type == "LLMCompleted"
        assert call_args.status == "completed"
        assert call_args.input_tokens == 10
        assert call_args.output_tokens == 5


class TestAgentCallbacks:
    """Test on_agent_action callback."""

    @pytest.mark.asyncio
    async def test_on_agent_action_sends_tool_started(
        self, handler, mock_governance_client, root_run_id
    ):
        """on_agent_action sends ToolStarted event before tool dispatch."""
        handler._root_run_id = str(root_run_id)
        handler._workflow_id = "test-wf"
        handler._run_id = "test-run"

        mock_governance_client.evaluate_event = AsyncMock(return_value=None)

        action = MagicMock()
        action.tool = "search_web"
        action.tool_input = {"query": "test"}

        await handler.on_agent_action(action=action, run_id=root_run_id)

        call_args = mock_governance_client.evaluate_event.call_args_list[0][0][0]
        assert call_args.event_type == "ToolStarted"
        assert call_args.tool_name == "search_web"

    @pytest.mark.asyncio
    async def test_on_agent_action_dedup_with_tool_start(
        self, handler, mock_governance_client, root_run_id, child_run_id
    ):
        """on_agent_action marks tool as seen to prevent duplicate in on_tool_start."""
        handler._root_run_id = str(root_run_id)
        handler._workflow_id = "test-wf"
        handler._run_id = "test-run"
        handler._workflow_started = True

        mock_governance_client.evaluate_event = AsyncMock(return_value=None)

        action = MagicMock()
        action.tool = "search_web"
        action.tool_input = {}

        # Call on_agent_action
        await handler.on_agent_action(action=action, run_id=root_run_id)

        # Now run_id is marked as seen
        assert str(root_run_id) in handler._seen_tool_run_ids

        # on_tool_start should skip governance if parent_run_id matches
        await handler.on_tool_start(
            serialized={"name": "search_web"},
            input_str="{}",
            run_id=child_run_id,
            parent_run_id=root_run_id,
        )

        # The run_id should have been removed from seen set
        assert str(root_run_id) not in handler._seen_tool_run_ids


class TestErrorHandling:
    """Test error handling in callbacks."""

    @pytest.mark.asyncio
    async def test_on_chain_error_root_sends_workflow_failed(
        self, handler, mock_governance_client, root_run_id
    ):
        """on_chain_error sends WorkflowCompleted with failed status."""
        handler._root_run_id = str(root_run_id)
        handler._workflow_id = "test-wf"
        handler._run_id = "test-run"
        handler._workflow_started = True
        handler._buffer.register(str(root_run_id), "chain", "Root")

        mock_governance_client.evaluate_event = AsyncMock(return_value=None)

        error = Exception("Test error")

        await handler.on_chain_error(error=error, run_id=root_run_id)

        call_args = mock_governance_client.evaluate_event.call_args_list[0][0][0]
        assert call_args.event_type == "WorkflowCompleted"
        assert call_args.status == "failed"
        assert call_args.error["message"] == "Test error"

    @pytest.mark.asyncio
    async def test_on_tool_error_cleans_buffer(
        self, handler, child_run_id
    ):
        """on_tool_error removes tool from buffer."""
        handler._buffer.register(str(child_run_id), "tool", "search_web")

        error = Exception("Tool failed")
        await handler.on_tool_error(error=error, run_id=child_run_id)

        assert handler._buffer.get(str(child_run_id)) is None

    @pytest.mark.asyncio
    async def test_on_llm_error_cleans_activity_map(
        self, handler, child_run_id
    ):
        """on_llm_error removes LLM from activity map."""
        handler._llm_activity_map[str(child_run_id)] = "activity-123"

        error = Exception("LLM failed")
        await handler.on_llm_error(error=error, run_id=child_run_id)

        assert str(child_run_id) not in handler._llm_activity_map


class TestStateManagement:
    """Test internal state management."""

    def test_reset_state_clears_all_fields(self, handler):
        """_reset_state clears all workflow state."""
        # Set some state
        handler._root_run_id = "root-123"
        handler._workflow_id = "wf-123"
        handler._run_id = "run-123"
        handler._workflow_started = True
        handler._pre_screen_response = MagicMock()
        handler._llm_activity_map = {"llm-1": "activity-1"}
        handler._seen_tool_run_ids = {"tool-1"}

        # Reset
        handler._reset_state()

        # All should be cleared
        assert handler._root_run_id is None
        assert handler._workflow_id is None
        assert handler._run_id is None
        assert handler._workflow_started is False
        assert handler._pre_screen_response is None
        assert handler._llm_activity_map == {}
        assert handler._seen_tool_run_ids == set()

    def test_buffer_manager_register_and_get(self, handler):
        """_RunBufferManager registers and retrieves runs."""
        run_id = "test-run-123"
        buf = handler._buffer.register(run_id, "tool", "search_web")

        assert buf.run_id == run_id
        assert buf.run_type == "tool"
        assert buf.name == "search_web"

        retrieved = handler._buffer.get(run_id)
        assert retrieved is buf

    def test_buffer_manager_duration_ms(self, handler):
        """_RunBufferManager calculates duration."""
        import time

        run_id = "test-run-123"
        handler._buffer.register(run_id, "tool", "search_web")
        time.sleep(0.01)  # Sleep 10ms
        dur = handler._buffer.duration_ms(run_id)

        assert dur is not None
        assert dur >= 10  # At least 10ms
        assert dur < 100  # Less than 100ms

    def test_buffer_manager_remove(self, handler):
        """_RunBufferManager removes runs."""
        run_id = "test-run-123"
        handler._buffer.register(run_id, "tool", "search_web")
        handler._buffer.remove(run_id)

        assert handler._buffer.get(run_id) is None

    def test_buffer_manager_clear(self, handler):
        """_RunBufferManager clears all runs."""
        handler._buffer.register("run-1", "tool", "tool1")
        handler._buffer.register("run-2", "tool", "tool2")

        handler._buffer.clear()

        assert handler._buffer.get("run-1") is None
        assert handler._buffer.get("run-2") is None
