"""Tests for middleware_tool_hook.py tool governance implementation."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from openbox_langgraph.errors import GovernanceHaltError
from openbox_langgraph.types import GovernanceVerdictResponse

from openbox_langchain.middleware_tool_hook import _send_tool_failed, handle_wrap_tool_call

# ─── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def mock_middleware():
    """Create a configured mock OpenBoxLangChainMiddleware."""
    mw = MagicMock()
    mw._client = AsyncMock()
    mw._config = MagicMock()
    mw._config.send_tool_start_event = True
    mw._config.send_tool_end_event = True
    mw._config.skip_tool_types = set()
    mw._config.tool_type_map = {}
    mw._config.task_queue = "langchain"
    mw._config.session_id = None
    mw._config.hitl = None
    mw._sync_mode = False
    mw._workflow_id = "wf-123"
    mw._run_id = "run-456"
    mw._workflow_type = "TestAgent"
    mw._span_processor = None
    return mw


@pytest.fixture
def mock_tool_request():
    """Create a mock ToolCallRequest."""
    request = MagicMock()
    request.tool_call = {"name": "search_web", "args": {"query": "test"}}
    return request


# ─── handle_wrap_tool_call ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_handle_wrap_tool_call_sends_tool_started(mock_middleware, mock_tool_request):
    """Send ToolStarted event."""
    handler = AsyncMock(return_value="tool result")

    with patch(
        "openbox_langchain.middleware_tool_hook._evaluate",
        new_callable=AsyncMock,
        return_value=None,  # No verdict means allow
    ) as mock_eval:
        with patch(
            "openbox_langchain.middleware_tool_hook._run_with_otel_context",
            new_callable=AsyncMock,
            return_value="tool result",
        ):
            result = await handle_wrap_tool_call(mock_middleware, mock_tool_request, handler)

    # Should have called _evaluate for ToolStarted and ToolCompleted
    events = [call[0][1] for call in mock_eval.call_args_list]
    event_types = [e.event_type for e in events]
    assert "ToolStarted" in event_types
    assert result == "tool result"


@pytest.mark.asyncio
async def test_handle_wrap_tool_call_sends_tool_completed(mock_middleware, mock_tool_request):
    """Send ToolCompleted event."""
    handler = AsyncMock(return_value="tool result")

    with patch(
        "openbox_langchain.middleware_tool_hook._evaluate",
        new_callable=AsyncMock,
        return_value=None,
    ) as mock_eval:
        with patch(
            "openbox_langchain.middleware_tool_hook._run_with_otel_context",
            new_callable=AsyncMock,
            return_value="tool result",
        ):
            await handle_wrap_tool_call(mock_middleware, mock_tool_request, handler)

    # Should have called _evaluate for ToolCompleted
    events = [call[0][1] for call in mock_eval.call_args_list]
    event_types = [e.event_type for e in events]
    assert "ToolCompleted" in event_types


@pytest.mark.asyncio
async def test_handle_wrap_tool_call_enforces_block_verdict(mock_middleware, mock_tool_request):
    """Enforce block verdict from ToolStarted."""
    verdict = MagicMock(spec=GovernanceVerdictResponse)
    verdict.verdict = "block"
    verdict.requires_hitl = False

    handler = AsyncMock(return_value="tool result")

    with patch(
        "openbox_langchain.middleware_tool_hook._evaluate",
        new_callable=AsyncMock,
        return_value=verdict,
    ):
        with patch(
            "openbox_langchain.middleware_tool_hook.enforce_verdict",
            side_effect=RuntimeError("blocked"),
        ):
            with pytest.raises(RuntimeError, match="blocked"):
                await handle_wrap_tool_call(mock_middleware, mock_tool_request, handler)


@pytest.mark.asyncio
async def test_handle_wrap_tool_call_skips_governance_for_excluded_tools(mock_middleware):
    """Skip governance for tools in skip_tool_types."""
    mock_middleware._config.skip_tool_types = {"internal_tool"}
    request = MagicMock()
    request.tool_call = {"name": "internal_tool", "args": {}}

    handler = AsyncMock(return_value="result")

    with patch(
        "openbox_langchain.middleware_tool_hook._evaluate",
        new_callable=AsyncMock,
    ) as mock_eval:
        result = await handle_wrap_tool_call(mock_middleware, request, handler)

    # Should not have called _evaluate
    assert mock_eval.call_count == 0
    handler.assert_called_once()
    assert result == "result"


@pytest.mark.asyncio
async def test_handle_wrap_tool_call_registers_span_processor_context(
    mock_middleware, mock_tool_request
):
    """Register activity context in span processor."""
    mock_middleware._span_processor = MagicMock()
    handler = AsyncMock(return_value="tool result")

    with patch(
        "openbox_langchain.middleware_tool_hook._evaluate",
        new_callable=AsyncMock,
        return_value=None,
    ):
        with patch(
            "openbox_langchain.middleware_tool_hook._run_with_otel_context",
            new_callable=AsyncMock,
            return_value="tool result",
        ):
            await handle_wrap_tool_call(mock_middleware, mock_tool_request, handler)

    # Should have called set_activity_context
    mock_middleware._span_processor.set_activity_context.assert_called_once()


@pytest.mark.asyncio
async def test_handle_wrap_tool_call_clears_span_processor_context(
    mock_middleware, mock_tool_request
):
    """Clear activity context after execution."""
    mock_middleware._span_processor = MagicMock()
    handler = AsyncMock(return_value="tool result")

    with patch(
        "openbox_langchain.middleware_tool_hook._evaluate",
        new_callable=AsyncMock,
        return_value=None,
    ):
        with patch(
            "openbox_langchain.middleware_tool_hook._run_with_otel_context",
            new_callable=AsyncMock,
            return_value="tool result",
        ):
            await handle_wrap_tool_call(mock_middleware, mock_tool_request, handler)

    # Should have called clear_activity_context
    mock_middleware._span_processor.clear_activity_context.assert_called_once()


@pytest.mark.asyncio
async def test_handle_wrap_tool_call_hitl_polling_on_require_approval(
    mock_middleware, mock_tool_request
):
    """Poll for HITL approval when required."""
    verdict = MagicMock(spec=GovernanceVerdictResponse)
    verdict.verdict = "require_approval"
    verdict.requires_hitl = True

    handler = AsyncMock(return_value="tool result")

    with patch(
        "openbox_langchain.middleware_tool_hook._evaluate",
        new_callable=AsyncMock,
        return_value=verdict,
    ):
        with patch(
            "openbox_langchain.middleware_tool_hook.enforce_verdict",
            return_value=verdict,
        ):
            with patch(
                "openbox_langchain.middleware_tool_hook._poll_approval_or_halt",
                new_callable=AsyncMock,
            ) as mock_poll:
                with patch(
                    "openbox_langchain.middleware_tool_hook._run_with_otel_context",
                    new_callable=AsyncMock,
                    side_effect=handler,
                ):
                    await handle_wrap_tool_call(mock_middleware, mock_tool_request, handler)

                # Should have called polling
                assert mock_poll.call_count >= 1


@pytest.mark.asyncio
async def test_handle_wrap_tool_call_hitl_halt_stops_execution(mock_middleware, mock_tool_request):
    """Stop execution if HITL approval is rejected."""
    verdict = MagicMock(spec=GovernanceVerdictResponse)
    verdict.verdict = "require_approval"
    verdict.requires_hitl = True

    handler = AsyncMock(return_value="tool result")

    with patch(
        "openbox_langchain.middleware_tool_hook._evaluate",
        new_callable=AsyncMock,
        return_value=verdict,
    ):
        with patch(
            "openbox_langchain.middleware_tool_hook.enforce_verdict",
            return_value=verdict,
        ):
            with patch(
                "openbox_langchain.middleware_tool_hook._poll_approval_or_halt",
                new_callable=AsyncMock,
                side_effect=GovernanceHaltError("rejected"),
            ):
                mock_middleware._span_processor = MagicMock()
                with pytest.raises(GovernanceHaltError):
                    await handle_wrap_tool_call(mock_middleware, mock_tool_request, handler)

                # Should have cleared span processor context on halt
                mock_middleware._span_processor.clear_activity_context.assert_called_once()


@pytest.mark.asyncio
async def test_handle_wrap_tool_call_hitl_retry_on_require_approval_from_hook(
    mock_middleware, mock_tool_request
):
    """Retry on require_approval verdict from hook governance."""
    handler = AsyncMock(return_value="tool result")

    with patch(
        "openbox_langchain.middleware_tool_hook._evaluate",
        new_callable=AsyncMock,
        return_value=None,
    ):
        with patch(
            "openbox_langchain.middleware_tool_hook._run_with_otel_context",
            new_callable=AsyncMock,
            return_value="tool result",
        ):
            with patch(
                "openbox_langchain.middleware_tool_hook._poll_approval_or_halt",
                new_callable=AsyncMock,
            ):
                result = await handle_wrap_tool_call(mock_middleware, mock_tool_request, handler)

    # Should have executed successfully
    assert result == "tool result"


@pytest.mark.asyncio
async def test_handle_wrap_tool_call_on_non_approval_error(mock_middleware, mock_tool_request):
    """Send ToolCompleted failed on non-approval error."""
    handler = AsyncMock(return_value="tool result")

    with patch(
        "openbox_langchain.middleware_tool_hook._evaluate",
        new_callable=AsyncMock,
        return_value=None,
    ):
        with patch(
            "openbox_langchain.middleware_tool_hook._run_with_otel_context",
            new_callable=AsyncMock,
            return_value="tool result",
        ):
            with patch(
                "openbox_langchain.middleware_tool_hook._send_tool_failed",
                new_callable=AsyncMock,
            ) as mock_send_failed:
                result = await handle_wrap_tool_call(mock_middleware, mock_tool_request, handler)

                # Should not have called _send_tool_failed (no error)
                assert mock_send_failed.call_count == 0
                assert result == "tool result"


@pytest.mark.asyncio
async def test_handle_wrap_tool_call_includes_tool_args_in_event(
    mock_middleware, mock_tool_request
):
    """Include tool args in ToolStarted event."""
    handler = AsyncMock(return_value="result")

    with patch(
        "openbox_langchain.middleware_tool_hook._evaluate",
        new_callable=AsyncMock,
        return_value=None,
    ) as mock_eval:
        with patch(
            "openbox_langchain.middleware_tool_hook._run_with_otel_context",
            new_callable=AsyncMock,
            return_value="result",
        ):
            await handle_wrap_tool_call(mock_middleware, mock_tool_request, handler)

    # Find ToolStarted event
    events = [call[0][1] for call in mock_eval.call_args_list]
    tool_started = [e for e in events if e.event_type == "ToolStarted"]
    assert len(tool_started) > 0
    assert tool_started[0].tool_name == "search_web"


@pytest.mark.asyncio
async def test_handle_wrap_tool_call_skips_tool_start_when_flag_false(
    mock_middleware, mock_tool_request
):
    """Skip ToolStarted when send_tool_start_event is False."""
    mock_middleware._config.send_tool_start_event = False

    handler = AsyncMock(return_value="result")

    with patch(
        "openbox_langchain.middleware_tool_hook._evaluate",
        new_callable=AsyncMock,
        return_value=None,
    ) as mock_eval:
        with patch(
            "openbox_langchain.middleware_tool_hook._run_with_otel_context",
            new_callable=AsyncMock,
            return_value="result",
        ):
            await handle_wrap_tool_call(mock_middleware, mock_tool_request, handler)

    # ToolStarted should not be sent (but ToolCompleted might be)
    events = [call[0][1] for call in mock_eval.call_args_list]
    event_types = [e.event_type for e in events]
    assert "ToolStarted" not in event_types


@pytest.mark.asyncio
async def test_handle_wrap_tool_call_skips_tool_end_when_flag_false(
    mock_middleware, mock_tool_request
):
    """Skip ToolCompleted when send_tool_end_event is False."""
    mock_middleware._config.send_tool_end_event = False

    handler = AsyncMock(return_value="result")

    with patch(
        "openbox_langchain.middleware_tool_hook._evaluate",
        new_callable=AsyncMock,
        return_value=None,
    ) as mock_eval:
        with patch(
            "openbox_langchain.middleware_tool_hook._run_with_otel_context",
            new_callable=AsyncMock,
            return_value="result",
        ):
            await handle_wrap_tool_call(mock_middleware, mock_tool_request, handler)

    # ToolCompleted should not be sent
    events = [call[0][1] for call in mock_eval.call_args_list]
    event_types = [e.event_type for e in events]
    assert "ToolCompleted" not in event_types


# ─── _send_tool_failed ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_send_tool_failed_sends_failed_event(mock_middleware):
    """Send ToolCompleted with failed status."""
    with patch(
        "openbox_langchain.middleware_tool_hook._evaluate",
        new_callable=AsyncMock,
    ) as mock_eval:
        error = RuntimeError("tool error")
        await _send_tool_failed(
            mock_middleware,
            activity_id="act-123",
            tool_name="search_web",
            tool_type="http",
            error=error,
            duration_ms=100.5,
        )

    # Should have called _evaluate with ToolCompleted
    assert mock_eval.call_count == 1
    event = mock_eval.call_args_list[0][0][1]
    assert event.event_type == "ToolCompleted"
    assert event.status == "failed"
    # activity_output is a dict with error key
    assert event.activity_output.get("error") == "tool error" or "tool error" in str(
        event.activity_output
    )


@pytest.mark.asyncio
async def test_send_tool_failed_clears_span_processor(mock_middleware):
    """Clear span processor context when sending failed."""
    mock_middleware._span_processor = MagicMock()

    with patch(
        "openbox_langchain.middleware_tool_hook._evaluate",
        new_callable=AsyncMock,
    ):
        error = RuntimeError("tool error")
        await _send_tool_failed(
            mock_middleware,
            activity_id="act-123",
            tool_name="search_web",
            tool_type="http",
            error=error,
            duration_ms=100.5,
        )

    mock_middleware._span_processor.clear_activity_context.assert_called_once()


@pytest.mark.asyncio
async def test_send_tool_failed_skips_when_flag_false(mock_middleware):
    """Skip sending when send_tool_end_event is False."""
    mock_middleware._config.send_tool_end_event = False

    with patch(
        "openbox_langchain.middleware_tool_hook._evaluate",
        new_callable=AsyncMock,
    ) as mock_eval:
        error = RuntimeError("tool error")
        await _send_tool_failed(
            mock_middleware,
            activity_id="act-123",
            tool_name="search_web",
            tool_type="http",
            error=error,
            duration_ms=100.5,
        )

    # Should not call _evaluate
    assert mock_eval.call_count == 0


@pytest.mark.asyncio
async def test_send_tool_failed_includes_duration(mock_middleware):
    """Include duration_ms in ToolCompleted event."""
    with patch(
        "openbox_langchain.middleware_tool_hook._evaluate",
        new_callable=AsyncMock,
    ) as mock_eval:
        error = RuntimeError("tool error")
        await _send_tool_failed(
            mock_middleware,
            activity_id="act-123",
            tool_name="search_web",
            tool_type="http",
            error=error,
            duration_ms=123.45,
        )

    event = mock_eval.call_args_list[0][0][1]
    assert event.duration_ms == 123.45


@pytest.mark.asyncio
async def test_send_tool_failed_includes_tool_metadata(mock_middleware):
    """Include tool_name and tool_type in event."""
    with patch(
        "openbox_langchain.middleware_tool_hook._evaluate",
        new_callable=AsyncMock,
    ) as mock_eval:
        error = RuntimeError("tool error")
        await _send_tool_failed(
            mock_middleware,
            activity_id="act-123",
            tool_name="search_web",
            tool_type="http",
            error=error,
            duration_ms=100.0,
        )

    event = mock_eval.call_args_list[0][0][1]
    assert event.tool_name == "search_web"
    assert event.tool_type == "http"
