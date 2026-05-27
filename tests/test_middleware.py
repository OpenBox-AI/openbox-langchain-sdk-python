"""Tests for OpenBoxLangChainMiddleware wrapper behavior."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from openbox_langchain.middleware import (
    OpenBoxLangChainMiddleware,
    OpenBoxLangChainMiddlewareOptions,
)


@pytest.fixture
def middleware():
    """Create middleware with external OpenBox dependencies mocked."""
    with patch("openbox_langchain.middleware.get_global_config") as mock_gc:
        mock_gc.return_value = MagicMock(
            api_url="https://test.openbox.ai",
            api_key="obx_test_123",
            governance_timeout=30.0,
            agent_did=None,
            agent_private_key=None,
        )
        with patch("openbox_langchain.middleware.GovernanceClient"):
            with patch("openbox_langchain.middleware.merge_config") as mock_merge:
                config = MagicMock()
                config.on_api_error = "fail_open"
                config.skip_tool_types = set()
                config.tool_type_map = {"search": "http"}
                mock_merge.return_value = config
                with patch("openbox_langgraph.otel_setup.setup_opentelemetry_for_governance"):
                    mw = OpenBoxLangChainMiddleware(
                        OpenBoxLangChainMiddlewareOptions(
                            agent_name="TestAgent",
                            tool_type_map={"search": "http"},
                        )
                    )

    span_processor = MagicMock()
    span_processor.set_sync_mode = MagicMock()
    mw._span_processor = span_processor
    return mw


@pytest.fixture
def runtime():
    """Runtime-like object with a stable thread id."""
    return MagicMock(config={"configurable": {"thread_id": "thread-123"}})


@pytest.fixture
def state():
    """Minimal LangGraph state with a user message."""
    return {"messages": [{"role": "user", "content": "hello"}]}


async def _async_value(value):
    return value


def test_close_shuts_down_sync_executor(middleware):
    """Close releases the sync bridge executor and is idempotent."""
    executor = MagicMock()
    middleware._sync_executor = executor

    middleware.close()
    middleware.close()

    executor.shutdown.assert_called_once_with(wait=False)
    assert middleware._sync_executor is None


def test_run_async_without_running_loop(middleware):
    """Sync callers without an active event loop use asyncio.run."""
    assert middleware._run_async(_async_value("ok")) == "ok"


@pytest.mark.asyncio
async def test_run_async_inside_running_loop_uses_executor(middleware):
    """Sync callers inside an active loop use the thread bridge."""
    try:
        result = middleware._run_async(_async_value("thread-ok"))
        assert result == "thread-ok"
        assert middleware._sync_executor is not None
    finally:
        middleware.close()


def test_before_agent_delegates_to_async_handler(middleware, state, runtime):
    """Sync before_agent sets sync mode and delegates to the async handler."""

    def run_async(coro):
        coro.close()
        return {"status": "started"}

    middleware._run_async = MagicMock(side_effect=run_async)

    result = middleware.before_agent(state, runtime)

    assert result == {"status": "started"}
    assert middleware._sync_mode is True
    middleware._span_processor.set_sync_mode.assert_called_once_with(True)
    middleware._run_async.assert_called_once()


def test_after_agent_delegates_to_async_handler(middleware, state, runtime):
    """Sync after_agent delegates to the async handler and returns None."""

    def run_async(coro):
        coro.close()
        return {"ignored": True}

    middleware._run_async = MagicMock(side_effect=run_async)

    assert middleware.after_agent(state, runtime) is None
    middleware._run_async.assert_called_once()


def test_wrap_model_call_creates_sync_span_and_delegates(middleware):
    """Sync model wrapper creates an OTel span around the async handler."""
    request = MagicMock()
    handler = MagicMock(return_value="handler-result")
    span = MagicMock()
    tracer = MagicMock()
    tracer.start_span.return_value = span

    async def fake_handle(mw, req, async_handler):
        assert mw is middleware
        assert req is request
        assert await async_handler(req) == "handler-result"
        return "model-result"

    with patch(
        "openbox_langchain.middleware_hook_handlers.handle_wrap_model_call",
        side_effect=fake_handle,
    ) as mock_handle, patch(
        "opentelemetry.trace.get_tracer",
        return_value=tracer,
    ), patch(
        "opentelemetry.trace.set_span_in_context",
        return_value="span-context",
    ), patch(
        "opentelemetry.context.attach",
        return_value="token",
    ) as attach, patch(
        "opentelemetry.context.detach",
    ) as detach:
        result = middleware.wrap_model_call(request, handler)

    assert result == "model-result"
    mock_handle.assert_called_once()
    attach.assert_called_once_with("span-context")
    detach.assert_called_once_with("token")
    span.end.assert_called_once()


def test_wrap_tool_call_creates_sync_span_and_delegates(middleware):
    """Sync tool wrapper names the OTel span from the tool call."""
    request = MagicMock()
    request.tool_call = {"name": "search", "args": {"query": "ai"}}
    handler = MagicMock(return_value="handler-result")
    span = MagicMock()
    tracer = MagicMock()
    tracer.start_span.return_value = span

    async def fake_handle(mw, req, async_handler):
        assert mw is middleware
        assert req is request
        assert await async_handler(req) == "handler-result"
        return "tool-result"

    with patch(
        "openbox_langchain.middleware_tool_hook.handle_wrap_tool_call",
        side_effect=fake_handle,
    ) as mock_handle, patch(
        "opentelemetry.trace.get_tracer",
        return_value=tracer,
    ), patch(
        "opentelemetry.trace.set_span_in_context",
        return_value="span-context",
    ), patch(
        "opentelemetry.context.attach",
        return_value="token",
    ), patch(
        "opentelemetry.context.detach",
    ) as detach:
        result = middleware.wrap_tool_call(request, handler)

    assert result == "tool-result"
    tracer.start_span.assert_called_once()
    assert tracer.start_span.call_args.args[0] == "tool.search.sync"
    mock_handle.assert_called_once()
    detach.assert_called_once_with("token")
    span.end.assert_called_once()


def test_wrap_tool_call_uses_default_tool_name_when_missing(middleware):
    """Tool wrapper falls back to a generic span name when no tool call exists."""
    request = MagicMock(spec=[])
    handler = MagicMock(return_value="handler-result")
    span = MagicMock()
    tracer = MagicMock()
    tracer.start_span.return_value = span

    async def fake_handle(_mw, _req, async_handler):
        assert await async_handler(_req) == "handler-result"
        return "tool-result"

    with patch(
        "openbox_langchain.middleware_tool_hook.handle_wrap_tool_call",
        side_effect=fake_handle,
    ), patch(
        "opentelemetry.trace.get_tracer",
        return_value=tracer,
    ), patch(
        "opentelemetry.trace.set_span_in_context",
        return_value="span-context",
    ), patch(
        "opentelemetry.context.attach",
        return_value="token",
    ), patch(
        "opentelemetry.context.detach",
    ):
        assert middleware.wrap_tool_call(request, handler) == "tool-result"

    assert tracer.start_span.call_args.args[0] == "tool.tool.sync"


@pytest.mark.asyncio
async def test_async_entrypoints_delegate_to_handler_functions(middleware, state, runtime):
    """Async wrappers delegate directly to handler modules."""
    model_request = MagicMock()
    tool_request = MagicMock()
    handler = AsyncMock()

    with patch(
        "openbox_langchain.middleware_hook_handlers.handle_before_agent",
        new_callable=AsyncMock,
        return_value={"before": True},
    ) as before, patch(
        "openbox_langchain.middleware_hook_handlers.handle_after_agent",
        new_callable=AsyncMock,
        return_value={"after": True},
    ) as after, patch(
        "openbox_langchain.middleware_hook_handlers.handle_wrap_model_call",
        new_callable=AsyncMock,
        return_value="model",
    ) as model, patch(
        "openbox_langchain.middleware_tool_hook.handle_wrap_tool_call",
        new_callable=AsyncMock,
        return_value="tool",
    ) as tool:
        assert await middleware.abefore_agent(state, runtime) == {"before": True}
        assert middleware._sync_mode is False
        middleware._span_processor.set_sync_mode.assert_called_with(False)
        assert await middleware.aafter_agent(state, runtime) == {"after": True}
        assert await middleware.awrap_model_call(model_request, handler) == "model"
        assert await middleware.awrap_tool_call(tool_request, handler) == "tool"

    before.assert_called_once_with(middleware, state, runtime)
    after.assert_called_once_with(middleware, state, runtime)
    model.assert_called_once_with(middleware, model_request, handler)
    tool.assert_called_once_with(middleware, tool_request, handler)


def test_init_disables_otel_when_setup_fails():
    """OTel setup failures leave middleware usable with Layer 1 governance."""
    with patch("openbox_langchain.middleware.get_global_config") as mock_gc:
        mock_gc.return_value = MagicMock(
            api_url="https://test.openbox.ai",
            api_key="obx_test_123",
            governance_timeout=30.0,
            agent_did=None,
            agent_private_key=None,
        )
        with patch("openbox_langchain.middleware.GovernanceClient"):
            with patch("openbox_langchain.middleware.merge_config") as mock_merge:
                mock_merge.return_value = MagicMock(on_api_error="fail_open")
                with patch(
                    "openbox_langgraph.otel_setup.setup_opentelemetry_for_governance",
                    side_effect=RuntimeError("otel failed"),
                ):
                    mw = OpenBoxLangChainMiddleware()

    assert mw._span_processor is not None
