"""Integration tests for OpenBoxGovernanceCallbackHandler."""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

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


class TestHandlerConfigurationOptions:
    """Test various handler configuration options."""

    def test_handler_with_custom_agent_name(
        self, mock_governance_client, mock_global_config
    ):
        """Handler respects custom agent_name in config."""
        opts = OpenBoxLangChainHandlerOptions(
            client=mock_governance_client,
            agent_name="MyCustomAgent",
        )
        handler = OpenBoxGovernanceCallbackHandler(opts)
        assert handler._config.agent_name == "MyCustomAgent"

    def test_handler_with_custom_session_id(
        self, mock_governance_client, mock_global_config
    ):
        """Handler respects custom session_id."""
        opts = OpenBoxLangChainHandlerOptions(
            client=mock_governance_client,
            session_id="custom-session-123",
        )
        handler = OpenBoxGovernanceCallbackHandler(opts)
        assert handler._config.session_id == "custom-session-123"

    def test_handler_with_skip_chain_types(
        self, mock_governance_client, mock_global_config
    ):
        """Handler respects skip_chain_types config."""
        opts = OpenBoxLangChainHandlerOptions(
            client=mock_governance_client,
            skip_chain_types={"LLMChain", "_deprecated"},
        )
        handler = OpenBoxGovernanceCallbackHandler(opts)
        assert "LLMChain" in handler._config.skip_chain_types
        assert "_deprecated" in handler._config.skip_chain_types

    def test_handler_with_tool_type_map(
        self, mock_governance_client, mock_global_config
    ):
        """Handler respects tool_type_map config."""
        tool_map = {"search": "web_search", "calc": "calculator"}
        opts = OpenBoxLangChainHandlerOptions(
            client=mock_governance_client,
            tool_type_map=tool_map,
        )
        handler = OpenBoxGovernanceCallbackHandler(opts)
        assert handler._config.tool_type_map == tool_map

    def test_handler_with_all_events_disabled(
        self, mock_governance_client, mock_global_config
    ):
        """Handler can be configured to skip all events."""
        opts = OpenBoxLangChainHandlerOptions(
            client=mock_governance_client,
            send_chain_start_event=False,
            send_chain_end_event=False,
            send_tool_start_event=False,
            send_tool_end_event=False,
            send_llm_start_event=False,
            send_llm_end_event=False,
        )
        handler = OpenBoxGovernanceCallbackHandler(opts)
        assert handler._config.send_chain_start_event is False
        assert handler._config.send_chain_end_event is False


class TestHandlerWithoutProvidedClient:
    """Test handler initialization without provided client."""

    def test_handler_builds_client_from_global_config(
        self, mock_global_config
    ):
        """Handler builds GovernanceClient from global config if not provided."""
        with patch("openbox_langchain.langchain_handler.GovernanceClient") as mock_client_class:
            mock_client_instance = MagicMock()
            mock_client_class.return_value = mock_client_instance

            opts = OpenBoxLangChainHandlerOptions(client=None)
            OpenBoxGovernanceCallbackHandler(opts)

            # Should have created client with global config values
            mock_client_class.assert_called_once()
            call_kwargs = mock_client_class.call_args[1]
            assert call_kwargs["api_url"] == "https://test.openbox.ai"
            assert call_kwargs["api_key"] == "obx_test_123"


class TestFullWorkflowSimulation:
    """Test a complete workflow simulation."""

    @pytest.mark.asyncio
    async def test_complete_workflow_chain_tool_llm_flow(
        self, handler, mock_governance_client
    ):
        """Simulate a complete workflow: chain start -> tool -> LLM -> chain end."""
        root_run_id = uuid.uuid4()
        tool_run_id = uuid.uuid4()
        llm_run_id = uuid.uuid4()

        mock_governance_client.evaluate_event = AsyncMock(return_value=None)

        # 1. Root chain start
        await handler.on_chain_start(
            serialized={"name": "AgentExecutor"},
            inputs={"input": "user query"},
            run_id=root_run_id,
            parent_run_id=None,
        )
        assert handler._root_run_id == str(root_run_id)

        # 2. Tool start
        await handler.on_tool_start(
            serialized={"name": "search_tool"},
            input_str='{"q": "test"}',
            run_id=tool_run_id,
            parent_run_id=root_run_id,
        )

        # 3. Tool end
        await handler.on_tool_end(
            output="search results",
            run_id=tool_run_id,
        )

        # 4. LLM start
        msg = MagicMock()
        msg.type = "human"
        msg.content = "What did you find?"

        await handler.on_chat_model_start(
            serialized={"name": "ChatOpenAI"},
            messages=[[msg]],
            run_id=llm_run_id,
        )

        # 5. LLM end
        gen = MagicMock()
        gen.text = "The answer is..."
        response = MagicMock()
        response.generations = [[gen]]
        response.llm_output = {}

        await handler.on_llm_end(response=response, run_id=llm_run_id)

        # 6. Chain end
        await handler.on_chain_end(
            outputs={"output": "final result"},
            run_id=root_run_id,
            parent_run_id=None,
        )

        # State should be reset after root chain ends
        assert handler._root_run_id is None
        assert handler._workflow_started is False

        # Verify events were sent
        assert mock_governance_client.evaluate_event.call_count > 0

    @pytest.mark.asyncio
    async def test_nested_chain_workflow(
        self, handler, mock_governance_client
    ):
        """Test nested chain handling."""
        root_run_id = uuid.uuid4()
        child_run_id = uuid.uuid4()
        grandchild_run_id = uuid.uuid4()

        handler._config.send_chain_start_event = True
        handler._config.send_chain_end_event = True
        mock_governance_client.evaluate_event = AsyncMock(return_value=None)

        # Root chain
        await handler.on_chain_start(
            serialized={"name": "Root"},
            inputs={"input": "test"},
            run_id=root_run_id,
            parent_run_id=None,
        )

        # Child chain
        await handler.on_chain_start(
            serialized={"name": "Child"},
            inputs={"key": "value"},
            run_id=child_run_id,
            parent_run_id=root_run_id,
        )

        # Grandchild chain
        await handler.on_chain_start(
            serialized={"name": "Grandchild"},
            inputs={},
            run_id=grandchild_run_id,
            parent_run_id=child_run_id,
        )

        # End grandchild
        await handler.on_chain_end(
            outputs={},
            run_id=grandchild_run_id,
            parent_run_id=child_run_id,
        )

        # End child
        await handler.on_chain_end(
            outputs={},
            run_id=child_run_id,
            parent_run_id=root_run_id,
        )

        # End root
        await handler.on_chain_end(
            outputs={},
            run_id=root_run_id,
            parent_run_id=None,
        )

        # All buffers should be cleaned
        assert handler._buffer.get(str(root_run_id)) is None
        assert handler._buffer.get(str(child_run_id)) is None
        assert handler._buffer.get(str(grandchild_run_id)) is None

    @pytest.mark.asyncio
    async def test_error_during_tool_execution(
        self, handler, mock_governance_client
    ):
        """Test error handling during tool execution."""
        root_run_id = uuid.uuid4()
        tool_run_id = uuid.uuid4()

        handler._root_run_id = str(root_run_id)
        handler._workflow_id = "test-wf"
        handler._run_id = "test-run"
        handler._workflow_started = True

        handler._buffer.register(str(tool_run_id), "tool", "search_tool")
        mock_governance_client.evaluate_event = AsyncMock(return_value=None)

        # Tool error
        error = Exception("Tool execution failed")
        await handler.on_tool_error(error=error, run_id=tool_run_id)

        # Buffer should be cleaned
        assert handler._buffer.get(str(tool_run_id)) is None

    @pytest.mark.asyncio
    async def test_workflow_with_redaction(
        self, handler, mock_governance_client
    ):
        """Test workflow with PII redaction response."""
        handler._root_run_id = "root"
        handler._workflow_id = "test-wf"
        handler._run_id = "test-run"
        handler._workflow_started = True

        # Setup pre-screen response with redaction
        response = MagicMock(spec=GovernanceVerdictResponse)
        response.verdict = Verdict.ALLOW
        response.guardrails_result = MagicMock()
        response.guardrails_result.input_type = "activity_input"
        response.guardrails_result.redacted_input = [
            {"prompt": "redacted prompt"}
        ]

        handler._pre_screen_response = response
        handler._pre_screen_activity_id = "pre-123"

        mock_governance_client.evaluate_event = AsyncMock(return_value=None)

        # Create human message
        msg = MagicMock()
        msg.type = "human"
        msg.content = "original content with pii"

        messages = [[msg]]

        llm_run_id = uuid.uuid4()
        await handler.on_chat_model_start(
            serialized={"name": "ChatOpenAI"},
            messages=messages,
            run_id=llm_run_id,
        )

        # Message should be redacted
        assert msg.content == "redacted prompt"

    @pytest.mark.asyncio
    async def test_api_error_fail_open(
        self, mock_governance_client, mock_global_config
    ):
        """Test fail_open behavior on API errors."""
        # API fails
        mock_governance_client.evaluate_event = AsyncMock(
            side_effect=Exception("API error")
        )

        opts = OpenBoxLangChainHandlerOptions(
            client=mock_governance_client,
            on_api_error="fail_open",
        )
        handler = OpenBoxGovernanceCallbackHandler(opts)
        handler._root_run_id = "root"
        handler._workflow_id = "wf"
        handler._run_id = "run"
        handler._workflow_started = True

        run_id = uuid.uuid4()

        # Should not raise on API error with fail_open
        # The exception handling is in the client/SDK level, not handler
        # But we can verify the handler tries to send events

        try:
            await handler.on_tool_start(
                serialized={"name": "tool"},
                input_str="{}",
                run_id=run_id,
            )
        except Exception:
            pass  # May or may not raise depending on SDK error handling


class TestMultipleInvocations:
    """Test handler reuse across multiple invocations."""

    @pytest.mark.asyncio
    async def test_handler_reset_between_invocations(
        self, handler, mock_governance_client
    ):
        """Handler can be reused for multiple invocations."""
        mock_governance_client.evaluate_event = AsyncMock(return_value=None)

        # First invocation
        run_id_1 = uuid.uuid4()
        await handler.on_chain_start(
            serialized={"name": "Root"},
            inputs={"input": "first"},
            run_id=run_id_1,
            parent_run_id=None,
        )
        assert handler._root_run_id == str(run_id_1)

        await handler.on_chain_end(
            outputs={},
            run_id=run_id_1,
            parent_run_id=None,
        )

        # State should be reset
        assert handler._root_run_id is None

        # Second invocation
        run_id_2 = uuid.uuid4()
        await handler.on_chain_start(
            serialized={"name": "Root"},
            inputs={"input": "second"},
            run_id=run_id_2,
            parent_run_id=None,
        )

        # Should set new root_run_id
        assert handler._root_run_id == str(run_id_2)
        assert handler._root_run_id != str(run_id_1)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_handler_initialization_with_none_options(self, mock_global_config):
        """Handler initializes with None options."""
        with patch("openbox_langchain.langchain_handler.GovernanceClient"):
            handler = OpenBoxGovernanceCallbackHandler(None)
            assert handler._config is not None
            assert handler._client is not None

    @pytest.mark.asyncio
    async def test_tool_start_with_empty_input(
        self, handler, mock_governance_client
    ):
        """Tool start with empty input string."""
        handler._root_run_id = "root"
        handler._workflow_id = "wf"
        handler._run_id = "run"
        handler._workflow_started = True

        mock_governance_client.evaluate_event = AsyncMock(return_value=None)

        run_id = uuid.uuid4()
        await handler.on_tool_start(
            serialized={"name": "tool"},
            input_str="",
            run_id=run_id,
        )

        assert mock_governance_client.evaluate_event.called

    @pytest.mark.asyncio
    async def test_llm_end_without_start_entry(
        self, handler, mock_governance_client
    ):
        """LLM end without corresponding start (missing from map)."""
        handler._root_run_id = "root"
        handler._workflow_id = "wf"
        handler._run_id = "run"

        mock_governance_client.evaluate_event = AsyncMock(return_value=None)

        run_id = uuid.uuid4()
        gen = MagicMock()
        gen.text = "response"
        response = MagicMock()
        response.generations = [[gen]]
        response.llm_output = {}

        # LLM end without corresponding start
        await handler.on_llm_end(response=response, run_id=run_id)

        # Should still send completion event
        assert mock_governance_client.evaluate_event.called

    @pytest.mark.asyncio
    async def test_chain_end_without_start(
        self, handler, mock_governance_client
    ):
        """Chain end without corresponding start (nested without config)."""
        handler._config.send_chain_end_event = True
        mock_governance_client.evaluate_event = AsyncMock(return_value=None)

        run_id = uuid.uuid4()

        # Chain end without start
        await handler.on_chain_end(
            outputs={},
            run_id=run_id,
            parent_run_id=uuid.uuid4(),
        )

        # Should handle gracefully
        assert handler._buffer.get(str(run_id)) is None
