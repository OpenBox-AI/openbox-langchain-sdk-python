"""Shared test fixtures for OpenBox LangChain SDK tests."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest


@pytest.fixture
def mock_governance_client():
    """Create a mock GovernanceClient that returns allow verdicts."""
    client = AsyncMock()
    client.evaluate_event = AsyncMock(return_value=None)
    client.poll_approval = AsyncMock(return_value={"action": "allow"})
    return client


@pytest.fixture
def mock_config():
    """Create a minimal GovernanceConfig-like dict for testing."""
    return {
        "on_api_error": "fail_open",
        "api_timeout": 30_000,
        "send_chain_start_event": True,
        "send_chain_end_event": True,
        "send_tool_start_event": True,
        "send_tool_end_event": True,
        "send_llm_start_event": True,
        "send_llm_end_event": True,
        "skip_chain_types": set(),
        "skip_tool_types": set(),
        "hitl": None,
        "session_id": None,
        "agent_name": "TestAgent",
        "task_queue": "langchain",
        "tool_type_map": None,
    }


@pytest.fixture
def mock_serialized_chain():
    """Serialized chain metadata as passed to on_chain_start."""
    return {"name": "AgentExecutor", "id": ["langchain", "agents", "AgentExecutor"]}


@pytest.fixture
def mock_serialized_tool():
    """Serialized tool metadata as passed to on_tool_start."""
    return {"name": "search_web", "id": ["langchain", "tools", "search_web"]}


@pytest.fixture
def mock_serialized_llm():
    """Serialized LLM metadata as passed to on_chat_model_start."""
    return {"name": "ChatOpenAI", "id": ["langchain_openai", "ChatOpenAI"]}
