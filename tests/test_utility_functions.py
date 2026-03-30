"""Tests for utility functions in langchain_handler module."""

from __future__ import annotations

from unittest.mock import MagicMock

from openbox_langchain.langchain_handler import (
    _apply_pii_redaction,
    _extract_llm_output,
    _extract_name,
    _extract_prompt_from_messages,
    _extract_user_prompt,
    _unwrap_tool_input,
)
from openbox_langchain.types import GovernanceVerdictResponse


class TestExtractName:
    """Test _extract_name() utility function."""

    def test_extract_name_from_serialized_field(self):
        """Extract name from 'name' field in serialized dict."""
        serialized = {"name": "AgentExecutor", "id": ["other"]}
        assert _extract_name(serialized) == "AgentExecutor"

    def test_extract_name_from_id_list(self):
        """Extract last element from 'id' list when 'name' absent."""
        serialized = {"id": ["langchain", "agents", "AgentExecutor"]}
        assert _extract_name(serialized) == "AgentExecutor"

    def test_extract_name_from_id_single_element(self):
        """Extract from single-element 'id' list."""
        serialized = {"id": ["SearchTool"]}
        assert _extract_name(serialized) == "SearchTool"

    def test_extract_name_default_when_missing(self):
        """Return 'Unknown' when both name and id missing."""
        serialized = {"other_field": "value"}
        assert _extract_name(serialized) == "Unknown"

    def test_extract_name_empty_id_list(self):
        """Return 'Unknown' when id list is empty."""
        serialized = {"id": []}
        assert _extract_name(serialized) == "Unknown"

    def test_extract_name_prefers_name_over_id(self):
        """Prefer 'name' field even if 'id' is present."""
        serialized = {
            "name": "PreferredName",
            "id": ["langchain", "OtherId"],
        }
        assert _extract_name(serialized) == "PreferredName"


class TestExtractUserPrompt:
    """Test _extract_user_prompt() utility function."""

    def test_extract_user_prompt_from_input_key(self):
        """Extract user prompt from 'input' key (AgentExecutor pattern)."""
        inputs = {"input": "What is 2+2?", "chat_history": []}
        assert _extract_user_prompt(inputs) == "What is 2+2?"

    def test_extract_user_prompt_from_messages_key(self):
        """Extract from last human message in 'messages' list."""
        msg = MagicMock()
        msg.content = "User query"
        msg.type = "human"
        inputs = {"messages": [msg]}
        assert _extract_user_prompt(inputs) == "User query"

    def test_extract_user_prompt_from_messages_finds_last_human(self):
        """Extract last human message when multiple messages exist."""
        msg1 = MagicMock()
        msg1.content = "First"
        msg1.type = "ai"

        msg2 = MagicMock()
        msg2.content = "Second (human)"
        msg2.type = "human"

        inputs = {"messages": [msg1, msg2]}
        assert _extract_user_prompt(inputs) == "Second (human)"

    def test_extract_user_prompt_recognizes_user_type(self):
        """Recognize both 'human' and 'user' types."""
        msg = MagicMock()
        msg.content = "User prompt"
        msg.type = "user"
        inputs = {"messages": [msg]}
        assert _extract_user_prompt(inputs) == "User prompt"

    def test_extract_user_prompt_handles_dict_format(self):
        """Handle dict-format messages with 'role' key."""
        inputs = {
            "messages": [
                {"role": "human", "content": "Test prompt"},
            ]
        }
        assert _extract_user_prompt(inputs) == "Test prompt"

    def test_extract_user_prompt_returns_none_when_missing(self):
        """Return None when no prompt found."""
        inputs = {"chat_history": []}
        assert _extract_user_prompt(inputs) is None

    def test_extract_user_prompt_returns_none_for_empty_messages(self):
        """Return None when messages list is empty."""
        inputs = {"messages": []}
        assert _extract_user_prompt(inputs) is None

    def test_extract_user_prompt_skips_non_human_messages(self):
        """Skip non-human messages when searching."""
        msg_ai = MagicMock()
        msg_ai.content = "AI response"
        msg_ai.type = "ai"

        inputs = {"messages": [msg_ai]}
        assert _extract_user_prompt(inputs) is None

    def test_extract_user_prompt_handles_content_as_string(self):
        """Handle messages with string content."""
        msg = MagicMock()
        msg.content = "String content"
        msg.type = "human"
        inputs = {"messages": [msg]}
        assert _extract_user_prompt(inputs) == "String content"

    def test_extract_user_prompt_converts_non_string_to_string(self):
        """Convert non-string content to string."""
        msg = MagicMock()
        msg.content = 123  # Non-string
        msg.type = "human"
        inputs = {"messages": [msg]}
        assert _extract_user_prompt(inputs) == "123"


class TestExtractPromptFromMessages:
    """Test _extract_prompt_from_messages() utility function."""

    def test_extract_prompt_from_single_human_message(self):
        """Extract text from single human message."""
        msg = MagicMock()
        msg.type = "human"
        msg.content = "What is AI?"
        messages = [[msg]]

        result = _extract_prompt_from_messages(messages)
        assert result == "What is AI?"

    def test_extract_prompt_from_multiple_groups(self):
        """Extract and join prompts from multiple message groups."""
        msg1 = MagicMock()
        msg1.type = "human"
        msg1.content = "First"

        msg2 = MagicMock()
        msg2.type = "human"
        msg2.content = "Second"

        messages = [[msg1], [msg2]]

        result = _extract_prompt_from_messages(messages)
        assert "First" in result
        assert "Second" in result

    def test_extract_prompt_skips_non_human_messages(self):
        """Skip ai and tool messages."""
        msg_human = MagicMock()
        msg_human.type = "human"
        msg_human.content = "Human"

        msg_ai = MagicMock()
        msg_ai.type = "ai"
        msg_ai.content = "AI response"

        messages = [[msg_human, msg_ai]]

        result = _extract_prompt_from_messages(messages)
        assert "Human" in result
        assert "AI response" not in result

    def test_extract_prompt_handles_multimodal_content(self):
        """Handle content list with text parts (multimodal)."""
        msg = MagicMock()
        msg.type = "human"
        msg.content = [
            {"type": "text", "text": "Tell me about"},
            {"type": "image_url", "image_url": {"url": "..."}},
            {"type": "text", "text": "this image"},
        ]

        messages = [[msg]]

        result = _extract_prompt_from_messages(messages)
        assert "Tell me about" in result
        assert "this image" in result

    def test_extract_prompt_handles_generic_type(self):
        """Include messages with 'generic' type."""
        msg = MagicMock()
        msg.type = "generic"
        msg.content = "Generic message"

        messages = [[msg]]

        result = _extract_prompt_from_messages(messages)
        assert "Generic message" in result

    def test_extract_prompt_returns_empty_for_no_human_messages(self):
        """Return empty string when no human/user messages."""
        msg = MagicMock()
        msg.type = "ai"
        msg.content = "Response"

        messages = [[msg]]

        result = _extract_prompt_from_messages(messages)
        assert result == ""

    def test_extract_prompt_handles_empty_message_groups(self):
        """Handle empty message groups."""
        messages = [[], []]
        result = _extract_prompt_from_messages(messages)
        assert result == ""


class TestApplyPIIRedaction:
    """Test _apply_pii_redaction() utility function."""

    def test_apply_pii_redaction_mutates_messages(self):
        """Mutate messages in-place with redacted content."""
        msg = MagicMock()
        msg.type = "human"
        msg.content = "Original: 1234-5678-9012-3456"

        messages = [[msg]]

        response = MagicMock(spec=GovernanceVerdictResponse)
        response.guardrails_result = MagicMock()
        response.guardrails_result.input_type = "activity_input"
        response.guardrails_result.redacted_input = [{"prompt": "Redacted: ****-****-****-3456"}]

        _apply_pii_redaction(messages, response)

        assert msg.content == "Redacted: ****-****-****-3456"

    def test_apply_pii_redaction_handles_list_format(self):
        """Extract redacted text from list format."""
        msg = MagicMock()
        msg.type = "human"
        msg.content = "Original"

        messages = [[msg]]

        response = MagicMock(spec=GovernanceVerdictResponse)
        response.guardrails_result = MagicMock()
        response.guardrails_result.input_type = "activity_input"
        response.guardrails_result.redacted_input = ["Redacted"]

        _apply_pii_redaction(messages, response)

        assert msg.content == "Redacted"

    def test_apply_pii_redaction_handles_string_format(self):
        """Handle redacted_input as string."""
        msg = MagicMock()
        msg.type = "human"
        msg.content = "Original"

        messages = [[msg]]

        response = MagicMock(spec=GovernanceVerdictResponse)
        response.guardrails_result = MagicMock()
        response.guardrails_result.input_type = "activity_input"
        response.guardrails_result.redacted_input = "Redacted String"

        _apply_pii_redaction(messages, response)

        assert msg.content == "Redacted String"

    def test_apply_pii_redaction_skips_when_no_guardrails_result(self):
        """Skip when no guardrails_result in response."""
        msg = MagicMock()
        msg.type = "human"
        original_content = "Original"
        msg.content = original_content

        messages = [[msg]]

        response = MagicMock(spec=GovernanceVerdictResponse)
        response.guardrails_result = None

        _apply_pii_redaction(messages, response)

        assert msg.content == original_content

    def test_apply_pii_redaction_skips_non_activity_input(self):
        """Skip when input_type is not 'activity_input'."""
        msg = MagicMock()
        msg.type = "human"
        original_content = "Original"
        msg.content = original_content

        messages = [[msg]]

        response = MagicMock(spec=GovernanceVerdictResponse)
        response.guardrails_result = MagicMock()
        response.guardrails_result.input_type = "other_type"
        response.guardrails_result.redacted_input = "Redacted"

        _apply_pii_redaction(messages, response)

        assert msg.content == original_content

    def test_apply_pii_redaction_skips_when_no_redacted_input(self):
        """Skip when redacted_input is None."""
        msg = MagicMock()
        msg.type = "human"
        original_content = "Original"
        msg.content = original_content

        messages = [[msg]]

        response = MagicMock(spec=GovernanceVerdictResponse)
        response.guardrails_result = MagicMock()
        response.guardrails_result.input_type = "activity_input"
        response.guardrails_result.redacted_input = None

        _apply_pii_redaction(messages, response)

        assert msg.content == original_content

    def test_apply_pii_redaction_replaces_last_human_message(self):
        """Replace the last human message in each group."""
        msg1 = MagicMock()
        msg1.type = "human"
        msg1.content = "First"

        msg2 = MagicMock()
        msg2.type = "human"
        msg2.content = "Last"

        msg_ai = MagicMock()
        msg_ai.type = "ai"
        msg_ai.content = "Response"

        messages = [[msg1, msg2, msg_ai]]

        response = MagicMock(spec=GovernanceVerdictResponse)
        response.guardrails_result = MagicMock()
        response.guardrails_result.input_type = "activity_input"
        response.guardrails_result.redacted_input = "Redacted"

        _apply_pii_redaction(messages, response)

        # Last human message should be redacted
        assert msg2.content == "Redacted"
        # First human should remain unchanged
        assert msg1.content == "First"
        # AI message should remain unchanged
        assert msg_ai.content == "Response"


class TestExtractLLMOutput:
    """Test _extract_llm_output() utility function."""

    def test_extract_llm_output_gets_text(self):
        """Extract output text from LLMResult."""
        gen = MagicMock()
        gen.text = "The answer is 42"
        response = MagicMock()
        response.generations = [[gen]]
        response.llm_output = {}

        text, _, _, _ = _extract_llm_output(response)
        assert text == "The answer is 42"

    def test_extract_llm_output_gets_token_counts(self):
        """Extract token usage from llm_output."""
        gen = MagicMock()
        gen.text = "Response"
        response = MagicMock()
        response.generations = [[gen]]
        response.llm_output = {
            "token_usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
            }
        }

        _, input_tokens, output_tokens, total_tokens = _extract_llm_output(response)
        assert input_tokens == 100
        assert output_tokens == 50
        assert total_tokens == 150

    def test_extract_llm_output_handles_usage_key(self):
        """Handle 'usage' key as fallback."""
        gen = MagicMock()
        gen.text = "Response"
        response = MagicMock()
        response.generations = [[gen]]
        response.llm_output = {
            "usage": {
                "prompt_tokens": 25,
                "completion_tokens": 10,
                "total_tokens": 35,
            }
        }

        _, input_tokens, output_tokens, total_tokens = _extract_llm_output(response)
        assert input_tokens == 25
        assert output_tokens == 10
        assert total_tokens == 35

    def test_extract_llm_output_handles_no_token_usage(self):
        """Return None for token counts when not present."""
        gen = MagicMock()
        gen.text = "Response"
        response = MagicMock()
        response.generations = [[gen]]
        response.llm_output = {}

        _, input_tokens, output_tokens, total_tokens = _extract_llm_output(response)
        assert input_tokens is None
        assert output_tokens is None
        assert total_tokens is None

    def test_extract_llm_output_handles_empty_generations(self):
        """Handle empty generations list."""
        response = MagicMock()
        response.generations = [[]]
        response.llm_output = {}

        text, _, _, _ = _extract_llm_output(response)
        assert text == ""

    def test_extract_llm_output_handles_no_generations(self):
        """Handle missing generations attribute."""
        response = MagicMock()
        response.generations = None
        response.llm_output = {}

        text, _, _, _ = _extract_llm_output(response)
        assert text == ""

    def test_extract_llm_output_handles_missing_text_attribute(self):
        """Handle generation with no 'text' attribute."""
        gen = MagicMock()
        del gen.text  # Remove text attribute
        response = MagicMock()
        response.generations = [[gen]]
        response.llm_output = {}

        text, _, _, _ = _extract_llm_output(response)
        assert text == ""


class TestUnwrapToolInput:
    """Test _unwrap_tool_input() utility function."""

    def test_unwrap_tool_input_json_string(self):
        """Unwrap JSON-encoded string to dict."""
        input_str = '{"query": "test", "param": 123}'
        result = _unwrap_tool_input(input_str)
        assert result == {"query": "test", "param": 123}

    def test_unwrap_tool_input_nested_json(self):
        """Handle nested JSON structure."""
        input_str = '{"data": {"nested": "value"}}'
        result = _unwrap_tool_input(input_str)
        assert result == {"data": {"nested": "value"}}

    def test_unwrap_tool_input_json_array(self):
        """Handle JSON array."""
        input_str = '["item1", "item2"]'
        result = _unwrap_tool_input(input_str)
        assert result == ["item1", "item2"]

    def test_unwrap_tool_input_invalid_json_passthrough(self):
        """Pass through invalid JSON string as-is."""
        input_str = "not valid json"
        result = _unwrap_tool_input(input_str)
        assert result == "not valid json"

    def test_unwrap_tool_input_non_string_passthrough(self):
        """Pass through non-string input as-is."""
        input_dict = {"already": "dict"}
        result = _unwrap_tool_input(input_dict)
        assert result == input_dict

    def test_unwrap_tool_input_integer_passthrough(self):
        """Pass through integer as-is."""
        result = _unwrap_tool_input(42)
        assert result == 42

    def test_unwrap_tool_input_empty_string(self):
        """Handle empty string as invalid JSON."""
        result = _unwrap_tool_input("")
        assert result == ""

    def test_unwrap_tool_input_json_primitives(self):
        """Handle JSON primitives."""
        assert _unwrap_tool_input("null") is None
        assert _unwrap_tool_input("true") is True
        assert _unwrap_tool_input("false") is False
        assert _unwrap_tool_input("123") == 123
        assert _unwrap_tool_input('"string"') == "string"
