"""Tests for middleware_hooks.py utility functions."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from openbox_langgraph.errors import GovernanceBlockedError

from openbox_langchain.middleware_hooks import (
    _append_human_content,
    _apply_pii_redaction,
    _base_event_fields,
    _evaluate,
    _extract_governance_blocked,
    _extract_last_user_message,
    _extract_prompt_from_messages,
    _extract_response_metadata,
)

# ─── _extract_last_user_message ────────────────────────────────────────


class TestExtractLastUserMessage:
    """Test _extract_last_user_message function."""

    def test_dict_message_with_user_role(self):
        """Extract text from dict message with 'user' role."""
        messages = [{"role": "user", "content": "Hello world"}]
        assert _extract_last_user_message(messages) == "Hello world"

    def test_dict_message_with_human_role(self):
        """Extract text from dict message with 'human' role."""
        messages = [{"role": "human", "content": "Hello"}]
        assert _extract_last_user_message(messages) == "Hello"

    def test_baseobject_with_human_type(self):
        """Extract text from BaseMessage-like object with 'human' type."""
        msg = MagicMock()
        msg.type = "human"
        msg.content = "Test message"
        messages = [msg]
        assert _extract_last_user_message(messages) == "Test message"

    def test_baseobject_with_generic_type(self):
        """Extract text from BaseMessage-like object with 'generic' type."""
        msg = MagicMock()
        msg.type = "generic"
        msg.content = "Generic content"
        messages = [msg]
        assert _extract_last_user_message(messages) == "Generic content"

    def test_empty_message_list(self):
        """Return None for empty message list."""
        assert _extract_last_user_message([]) is None

    def test_no_user_messages(self):
        """Return None when no user/human messages present."""
        messages = [
            {"role": "assistant", "content": "Response"},
            {"role": "system", "content": "System prompt"},
        ]
        assert _extract_last_user_message(messages) is None

    def test_last_user_message_wins(self):
        """Return last user message when multiple exist."""
        messages = [
            {"role": "user", "content": "First"},
            {"role": "assistant", "content": "Response"},
            {"role": "user", "content": "Second"},
        ]
        assert _extract_last_user_message(messages) == "Second"

    def test_non_string_content_ignored(self):
        """Skip messages with non-string content."""
        messages = [
            {"role": "user", "content": ["not", "a", "string"]},
            {"role": "user", "content": "Valid string"},
        ]
        assert _extract_last_user_message(messages) == "Valid string"

    def test_mixed_message_types(self):
        """Handle mixed dict and object messages."""
        msg1 = {"role": "user", "content": "First"}
        msg2 = MagicMock()
        msg2.type = "human"
        msg2.content = "Second"
        messages = [msg1, {"role": "assistant", "content": "resp"}, msg2]
        assert _extract_last_user_message(messages) == "Second"


# ─── _extract_prompt_from_messages ─────────────────────────────────────


class TestExtractPromptFromMessages:
    """Test _extract_prompt_from_messages function."""

    def test_flat_dict_list(self):
        """Extract text from flat dict list."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]
        prompt = _extract_prompt_from_messages(messages)
        assert "Hello" in prompt

    def test_flat_baseobject_list(self):
        """Extract text from flat BaseMessage-like list."""
        msg1 = MagicMock()
        msg1.type = "human"
        msg1.content = "Hello"
        msg2 = MagicMock()
        msg2.type = "generic"
        msg2.content = "World"
        messages = [msg1, msg2]
        prompt = _extract_prompt_from_messages(messages)
        assert "Hello" in prompt
        assert "World" in prompt

    def test_nested_list(self):
        """Extract text from nested message list."""
        msg1 = MagicMock()
        msg1.type = "human"
        msg1.content = "Inner"
        messages = [
            [
                {"role": "user", "content": "Outer"},
                msg1,
            ]
        ]
        prompt = _extract_prompt_from_messages(messages)
        assert "Outer" in prompt
        assert "Inner" in prompt

    def test_multimodal_content(self):
        """Extract text from multimodal (list) content."""
        msg = {
            "role": "user",
            "content": [
                {"type": "text", "text": "Hello"},
                {"type": "image", "data": "..."},
                {"type": "text", "text": "World"},
            ],
        }
        messages = [msg]
        prompt = _extract_prompt_from_messages(messages)
        assert "Hello" in prompt
        assert "World" in prompt

    def test_empty_message_list(self):
        """Return empty string for empty message list."""
        assert _extract_prompt_from_messages([]) == ""

    def test_non_list_input(self):
        """Return empty string for non-list input."""
        assert _extract_prompt_from_messages("not a list") == ""
        assert _extract_prompt_from_messages(123) == ""

    def test_multipart_join(self):
        """Join multiple parts with newline."""
        messages = [
            {"role": "user", "content": "Line 1"},
            {"role": "user", "content": "Line 2"},
        ]
        prompt = _extract_prompt_from_messages(messages)
        assert "Line 1\nLine 2" in prompt

    def test_skip_non_human_messages(self):
        """Skip messages without human/user/generic role."""
        messages = [
            {"role": "user", "content": "User"},
            {"role": "assistant", "content": "Assistant"},
            {"role": "system", "content": "System"},
        ]
        prompt = _extract_prompt_from_messages(messages)
        assert "User" in prompt
        assert "Assistant" not in prompt
        assert "System" not in prompt


# ─── _append_human_content ────────────────────────────────────────────


class TestAppendHumanContent:
    """Test _append_human_content function."""

    def test_dict_with_user_role(self):
        """Append content from dict with 'user' role."""
        msg = {"role": "user", "content": "Hello"}
        parts = []
        _append_human_content(msg, parts)
        assert parts == ["Hello"]

    def test_dict_with_type_field(self):
        """Append content from dict with 'type' field."""
        msg = {"type": "human", "content": "Hello"}
        parts = []
        _append_human_content(msg, parts)
        assert parts == ["Hello"]

    def test_baseobject_with_type(self):
        """Append content from BaseMessage-like object."""
        msg = MagicMock()
        msg.type = "human"
        msg.content = "Hello"
        parts = []
        _append_human_content(msg, parts)
        assert parts == ["Hello"]

    def test_skip_non_human_role(self):
        """Skip messages with non-human role."""
        msg = {"role": "assistant", "content": "Response"}
        parts = []
        _append_human_content(msg, parts)
        assert parts == []

    def test_multimodal_content(self):
        """Extract text from multimodal content list."""
        msg = {
            "role": "user",
            "content": [
                {"type": "text", "text": "Hello"},
                {"type": "image", "data": "..."},
                {"type": "text", "text": "World"},
            ],
        }
        parts = []
        _append_human_content(msg, parts)
        assert parts == ["Hello", "World"]

    def test_skip_multimodal_non_text(self):
        """Skip non-text parts in multimodal content."""
        msg = {
            "role": "user",
            "content": [
                {"type": "image", "data": "..."},
                {"type": "audio", "data": "..."},
            ],
        }
        parts = []
        _append_human_content(msg, parts)
        assert parts == []

    def test_missing_text_field(self):
        """Handle multimodal parts missing 'text' field."""
        msg = {
            "role": "user",
            "content": [
                {"type": "text"},  # missing 'text' field
                {"type": "text", "text": "Hello"},
            ],
        }
        parts = []
        _append_human_content(msg, parts)
        assert parts == ["", "Hello"]


# ─── _apply_pii_redaction ─────────────────────────────────────────────


class TestApplyPIIRedaction:
    """Test _apply_pii_redaction function."""

    def test_string_redaction(self):
        """Apply string redaction to last user message."""
        messages = [
            MagicMock(type="human", content="Original"),
            MagicMock(type="assistant", content="Response"),
        ]
        _apply_pii_redaction(messages, "Redacted")
        assert messages[0].content == "Redacted"
        assert messages[1].content == "Response"

    def test_dict_message_redaction(self):
        """Apply redaction to dict messages."""
        messages = [
            {"role": "user", "content": "Original"},
            {"role": "assistant", "content": "Response"},
        ]
        _apply_pii_redaction(messages, "Redacted")
        assert messages[0]["content"] == "Redacted"

    def test_list_redaction_with_dict(self):
        """Apply redaction from list with dict structure."""
        messages = [MagicMock(type="human", content="Original")]
        redacted = [{"prompt": "Redacted"}]
        _apply_pii_redaction(messages, redacted)
        assert messages[0].content == "Redacted"

    def test_list_redaction_with_string(self):
        """Apply redaction from list with string."""
        messages = [MagicMock(type="human", content="Original")]
        redacted = ["Redacted"]
        _apply_pii_redaction(messages, redacted)
        assert messages[0].content == "Redacted"

    def test_no_redaction_needed(self):
        """Skip redaction when redacted_input is empty."""
        messages = [MagicMock(type="human", content="Original")]
        _apply_pii_redaction(messages, None)
        assert messages[0].content == "Original"

    def test_no_user_message_to_redact(self):
        """Handle case where no user message exists."""
        messages = [MagicMock(type="assistant", content="Response")]
        _apply_pii_redaction(messages, "Redacted")
        # Should not crash, just not modify anything
        assert messages[0].content == "Response"

    def test_empty_message_list(self):
        """Handle empty message list."""
        messages = []
        _apply_pii_redaction(messages, "Redacted")
        assert messages == []

    def test_last_user_message_wins(self):
        """Redact the last user message when multiple exist."""
        messages = [
            MagicMock(type="human", content="First"),
            MagicMock(type="assistant", content="Response"),
            MagicMock(type="human", content="Second"),
        ]
        _apply_pii_redaction(messages, "Redacted")
        assert messages[0].content == "First"  # unchanged
        assert messages[2].content == "Redacted"  # changed


# ─── _extract_response_metadata ────────────────────────────────────────


class TestExtractResponseMetadata:
    """Test _extract_response_metadata function."""

    def test_extract_tokens(self):
        """Extract token counts from response."""
        ai_msg = MagicMock()
        ai_msg.usage_metadata = {
            "input_tokens": 10,
            "output_tokens": 20,
        }
        ai_msg.response_metadata = {"model_name": "gpt-4"}
        ai_msg.content = "Response"
        ai_msg.tool_calls = None

        response = MagicMock()
        response.message = ai_msg

        meta = _extract_response_metadata(response)
        assert meta["input_tokens"] == 10
        assert meta["output_tokens"] == 20
        assert meta["total_tokens"] == 30
        assert meta["llm_model"] == "gpt-4"

    def test_extract_tokens_from_prompt_tokens(self):
        """Extract tokens using prompt_tokens/completion_tokens keys."""
        ai_msg = MagicMock()
        ai_msg.usage_metadata = {
            "prompt_tokens": 5,
            "completion_tokens": 15,
        }
        ai_msg.response_metadata = {}
        ai_msg.content = "Response"
        ai_msg.tool_calls = None

        response = MagicMock()
        response.message = ai_msg

        meta = _extract_response_metadata(response)
        assert meta["input_tokens"] == 5
        assert meta["output_tokens"] == 15
        assert meta["total_tokens"] == 20

    def test_extract_without_tokens(self):
        """Extract metadata without token counts."""
        ai_msg = MagicMock()
        ai_msg.usage_metadata = {}
        ai_msg.response_metadata = {"model_name": "gpt-4"}
        ai_msg.content = "Response"
        ai_msg.tool_calls = None

        response = MagicMock()
        response.message = ai_msg

        meta = _extract_response_metadata(response)
        assert meta["input_tokens"] is None
        assert meta["output_tokens"] is None
        assert meta["total_tokens"] is None
        assert meta["llm_model"] == "gpt-4"

    def test_extract_completion_string(self):
        """Extract completion from string content."""
        ai_msg = MagicMock()
        ai_msg.usage_metadata = {}
        ai_msg.response_metadata = {}
        ai_msg.content = "Hello world"
        ai_msg.tool_calls = None

        response = MagicMock()
        response.message = ai_msg

        meta = _extract_response_metadata(response)
        assert meta["completion"] == "Hello world"

    def test_extract_completion_multimodal(self):
        """Extract completion from multimodal content list."""
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

        meta = _extract_response_metadata(response)
        assert meta["completion"] == "Hello World"

    def test_extract_tool_calls(self):
        """Detect presence of tool calls."""
        ai_msg = MagicMock()
        ai_msg.usage_metadata = {}
        ai_msg.response_metadata = {}
        ai_msg.content = "Call tool"
        ai_msg.tool_calls = [{"name": "search", "args": {}}]

        response = MagicMock()
        response.message = ai_msg

        meta = _extract_response_metadata(response)
        assert meta["has_tool_calls"] is True

    def test_no_tool_calls(self):
        """Detect absence of tool calls."""
        ai_msg = MagicMock()
        ai_msg.usage_metadata = {}
        ai_msg.response_metadata = {}
        ai_msg.content = "No tools"
        ai_msg.tool_calls = None

        response = MagicMock()
        response.message = ai_msg

        meta = _extract_response_metadata(response)
        assert meta["has_tool_calls"] is False

    def test_response_without_message_attribute(self):
        """Handle response that IS the ai_msg."""
        ai_msg = MagicMock()
        ai_msg.usage_metadata = {"input_tokens": 10}
        ai_msg.response_metadata = {"model_name": "gpt-4"}
        ai_msg.content = "Response"
        ai_msg.tool_calls = None
        # Ensure response has no message attribute
        del ai_msg.message

        meta = _extract_response_metadata(ai_msg)
        # When usage_metadata is directly on ai_msg, it should be extracted
        assert meta.get("input_tokens") == 10
        # response_metadata is also directly on ai_msg
        assert meta.get("completion") == "Response"


# ─── _extract_governance_blocked ───────────────────────────────────────


class TestExtractGovernanceBlocked:
    """Test _extract_governance_blocked function."""

    def test_direct_governance_error(self):
        """Extract GovernanceBlockedError from exception directly."""
        gov_err = GovernanceBlockedError("blocked", "block")
        result = _extract_governance_blocked(gov_err)
        assert result is gov_err

    def test_wrapped_governance_error_via_cause(self):
        """Extract GovernanceBlockedError wrapped via __cause__."""
        gov_err = GovernanceBlockedError("blocked", "block")
        outer_err = RuntimeError("outer")
        outer_err.__cause__ = gov_err
        result = _extract_governance_blocked(outer_err)
        assert result is gov_err

    def test_wrapped_governance_error_via_context(self):
        """Extract GovernanceBlockedError wrapped via __context__."""
        gov_err = GovernanceBlockedError("blocked", "block")
        outer_err = RuntimeError("outer")
        outer_err.__context__ = gov_err
        result = _extract_governance_blocked(outer_err)
        assert result is gov_err

    def test_chain_of_errors(self):
        """Extract from chain of wrapped errors."""
        gov_err = GovernanceBlockedError("blocked", "block")
        mid_err = RuntimeError("middle")
        mid_err.__cause__ = gov_err
        outer_err = RuntimeError("outer")
        outer_err.__cause__ = mid_err
        result = _extract_governance_blocked(outer_err)
        assert result is gov_err

    def test_no_governance_error(self):
        """Return None when no GovernanceBlockedError in chain."""
        err = RuntimeError("no governance error")
        result = _extract_governance_blocked(err)
        assert result is None

    def test_circular_error_chain(self):
        """Handle circular exception chains (prevent infinite loop)."""
        err1 = RuntimeError("err1")
        err2 = RuntimeError("err2")
        err1.__cause__ = err2
        err2.__cause__ = err1  # cycle
        result = _extract_governance_blocked(err1)
        assert result is None

    def test_governance_error_in_deep_chain(self):
        """Extract GovernanceBlockedError from deep chain."""
        gov_err = GovernanceBlockedError("blocked", "require_approval")
        chain = RuntimeError()
        chain.__cause__ = RuntimeError()
        chain.__cause__.__cause__ = RuntimeError()
        chain.__cause__.__cause__.__cause__ = gov_err
        result = _extract_governance_blocked(chain)
        assert result is gov_err


# ─── _base_event_fields ────────────────────────────────────────────────


class TestBaseEventFields:
    """Test _base_event_fields function."""

    def test_returns_dict_with_required_fields(self):
        """Return dict with all required event fields."""
        mw = MagicMock()
        mw._workflow_id = "wf-123"
        mw._run_id = "run-456"
        mw._workflow_type = "TestAgent"
        mw._config = MagicMock(task_queue="langchain", session_id="sess-789")

        fields = _base_event_fields(mw)
        assert fields["source"] == "workflow-telemetry"
        assert fields["workflow_id"] == "wf-123"
        assert fields["run_id"] == "run-456"
        assert fields["workflow_type"] == "TestAgent"
        assert fields["task_queue"] == "langchain"
        assert fields["session_id"] == "sess-789"

    def test_includes_timestamp(self):
        """Include timestamp field in event."""
        mw = MagicMock()
        mw._workflow_id = "wf-123"
        mw._run_id = "run-456"
        mw._workflow_type = "TestAgent"
        mw._config = MagicMock(task_queue="langchain", session_id=None)

        fields = _base_event_fields(mw)
        assert "timestamp" in fields
        assert fields["timestamp"]  # non-empty


# ─── _evaluate ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_evaluate_async_mode():
    """Call evaluate_event in async mode."""
    mw = MagicMock()
    mw._sync_mode = False
    mw._client = AsyncMock()
    mw._client.evaluate_event = AsyncMock(return_value={"verdict": "allow"})

    event = {"event_type": "LLMStarted"}
    result = await _evaluate(mw, event)

    assert result == {"verdict": "allow"}
    mw._client.evaluate_event.assert_called_once_with(event)


@pytest.mark.asyncio
async def test_evaluate_sync_mode():
    """Call evaluate_event_sync in sync mode."""
    mw = MagicMock()
    mw._sync_mode = True
    mw._client = MagicMock()
    mw._client.evaluate_event_sync = MagicMock(return_value={"verdict": "allow"})

    event = {"event_type": "LLMStarted"}
    result = await _evaluate(mw, event)

    assert result == {"verdict": "allow"}
    mw._client.evaluate_event_sync.assert_called_once_with(event)
