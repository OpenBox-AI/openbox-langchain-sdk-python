"""OpenBox governance middleware for LangChain agents.

Subclasses AgentMiddleware to intercept agent lifecycle and enforce governance:
    before_agent  → WorkflowStarted + SignalReceived + pre-screen guardrails
    wrap_model_call → LLMStarted (PII redaction) → Model → LLMCompleted
    wrap_tool_call  → ToolStarted → Tool (OTel spans) → ToolCompleted
    after_agent   → WorkflowCompleted + cleanup

Usage:
    from openbox_langchain import create_openbox_langchain_middleware
    middleware = create_openbox_langchain_middleware(api_url=..., api_key=...)
    agent = create_agent(model=..., tools=[...], middleware=[middleware])
    result = agent.invoke({"messages": [("user", "Hello")]})
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from langchain.agents.middleware.types import AgentMiddleware, ModelRequest
from langgraph.prebuilt.tool_node import ToolCallRequest
from openbox_langgraph.client import GovernanceClient
from openbox_langgraph.config import GovernanceConfig, get_global_config, merge_config
from openbox_langgraph.types import GovernanceVerdictResponse

if TYPE_CHECKING:
    from openbox_langgraph.span_processor import WorkflowSpanProcessor

_logger = logging.getLogger("openbox_langchain")


@dataclass
class OpenBoxLangChainMiddlewareOptions:
    """Configuration for OpenBoxLangChainMiddleware."""

    agent_name: str | None = None
    session_id: str | None = None
    task_queue: str = "langchain"
    on_api_error: str = "fail_open"
    governance_timeout: float = 30.0
    tool_type_map: dict[str, str] = field(default_factory=dict)
    skip_tool_types: set[str] = field(default_factory=set)
    sqlalchemy_engine: Any = None
    send_chain_start_event: bool = True
    send_chain_end_event: bool = True
    send_llm_start_event: bool = True
    send_llm_end_event: bool = True
    send_tool_start_event: bool = True
    send_tool_end_event: bool = True


class OpenBoxLangChainMiddleware(AgentMiddleware):
    """AgentMiddleware implementing OpenBox governance for LangChain agents.

    Hooks map directly to the governance event lifecycle:
    - before_agent: session setup (WorkflowStarted, SignalReceived, pre-screen)
    - wrap_model_call: LLM governance (LLMStarted/Completed, PII redaction)
    - wrap_tool_call: tool governance (ToolStarted/Completed, SpanProcessor ctx)
    - after_agent: session close (WorkflowCompleted, cleanup)
    """

    def __init__(self, options: OpenBoxLangChainMiddlewareOptions | None = None) -> None:
        opts = options or OpenBoxLangChainMiddlewareOptions()
        self._options = opts

        self._config: GovernanceConfig = merge_config({
            "on_api_error": opts.on_api_error,
            "api_timeout": opts.governance_timeout,
            "send_chain_start_event": opts.send_chain_start_event,
            "send_chain_end_event": opts.send_chain_end_event,
            "send_tool_start_event": opts.send_tool_start_event,
            "send_tool_end_event": opts.send_tool_end_event,
            "send_llm_start_event": opts.send_llm_start_event,
            "send_llm_end_event": opts.send_llm_end_event,
            "skip_tool_types": opts.skip_tool_types,
            "session_id": opts.session_id,
            "agent_name": opts.agent_name,
            "task_queue": opts.task_queue,
            "tool_type_map": opts.tool_type_map,
        })

        gc = get_global_config()
        self._client = GovernanceClient(
            api_url=gc.api_url,
            api_key=gc.api_key,
            timeout=gc.governance_timeout,
            on_api_error=self._config.on_api_error,
            agent_did=gc.agent_did,
            agent_private_key=gc.agent_private_key,
        )

        # OTel span processor for hook-level governance (Layer 2/3)
        self._span_processor: WorkflowSpanProcessor | None = None
        if gc.api_url and gc.api_key:
            try:
                from openbox_langgraph.otel_setup import setup_opentelemetry_for_governance
                from openbox_langgraph.span_processor import WorkflowSpanProcessor as WSP

                self._span_processor = WSP()
                setup_opentelemetry_for_governance(
                    span_processor=self._span_processor,
                    api_url=gc.api_url,
                    api_key=gc.api_key,
                    ignored_urls=[gc.api_url],
                    api_timeout=gc.governance_timeout,
                    on_api_error=self._config.on_api_error,
                    sqlalchemy_engine=opts.sqlalchemy_engine,
                    agent_did=gc.agent_did,
                    agent_private_key=gc.agent_private_key,
                )
                logging.getLogger("opentelemetry.context").setLevel(logging.CRITICAL)
            except Exception:
                _logger.warning("Failed to initialize OTel hooks; Layer 2/3 disabled")

        # Reusable thread pool for sync-to-async bridge (shut down via close())
        self._sync_executor: concurrent.futures.ThreadPoolExecutor | None = None

        # Per-invocation state (reset in before_agent)
        self._sync_mode: bool = False
        self._workflow_id: str = ""
        self._run_id: str = ""
        self._pre_screen_response: GovernanceVerdictResponse | None = None
        self._first_llm_call: bool = True
        self._workflow_type: str = opts.agent_name or "LangChainRun"

    def close(self) -> None:
        """Release resources (thread pool). Safe to call multiple times."""
        if self._sync_executor is not None:
            self._sync_executor.shutdown(wait=False)
            self._sync_executor = None

    # ─── Async-to-sync bridge ──────────────────────────────────────

    def _run_async(self, coro):
        """Run async coroutine from sync context with OTel context propagation."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop and loop.is_running():
            from opentelemetry import context as otel_context
            ctx = otel_context.get_current()

            def _run_with_ctx():
                token = otel_context.attach(ctx)
                try:
                    return asyncio.run(coro)
                finally:
                    try:
                        otel_context.detach(token)
                    except Exception:
                        pass

            if self._sync_executor is None:
                self._sync_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            return self._sync_executor.submit(_run_with_ctx).result()
        return asyncio.run(coro)

    # ─── Sync hooks (for invoke/stream) ────────────────────────────

    def before_agent(self, state, runtime) -> dict[str, Any] | None:
        self._sync_mode = True
        if self._span_processor:
            self._span_processor.set_sync_mode(True)
        from openbox_langchain.middleware_hook_handlers import handle_before_agent
        return self._run_async(handle_before_agent(self, state, runtime))

    def after_agent(self, state, runtime) -> dict[str, Any] | None:
        from openbox_langchain.middleware_hook_handlers import handle_after_agent
        self._run_async(handle_after_agent(self, state, runtime))
        return None

    def wrap_model_call(self, request: ModelRequest, handler) -> Any:
        from opentelemetry import context as otel_ctx
        from opentelemetry import trace as otel_tr

        from openbox_langchain.middleware_hook_handlers import handle_wrap_model_call

        tracer = otel_tr.get_tracer("openbox-langchain")
        span = tracer.start_span("llm.call.sync", kind=otel_tr.SpanKind.INTERNAL)
        token = otel_ctx.attach(otel_tr.set_span_in_context(span))

        try:
            async def async_handler(req):
                return handler(req)

            return self._run_async(handle_wrap_model_call(self, request, async_handler))
        finally:
            span.end()
            try:
                otel_ctx.detach(token)
            except Exception:
                pass

    def wrap_tool_call(self, request: ToolCallRequest, handler) -> Any:
        from opentelemetry import context as otel_ctx
        from opentelemetry import trace as otel_tr

        from openbox_langchain.middleware_tool_hook import handle_wrap_tool_call

        tool_name = (
            request.tool_call.get("name", "tool") if hasattr(request, "tool_call") else "tool"
        )
        tracer = otel_tr.get_tracer("openbox-langchain")
        span = tracer.start_span(f"tool.{tool_name}.sync", kind=otel_tr.SpanKind.INTERNAL)
        token = otel_ctx.attach(otel_tr.set_span_in_context(span))

        try:
            async def async_handler(req):
                return handler(req)

            return self._run_async(handle_wrap_tool_call(self, request, async_handler))
        finally:
            span.end()
            try:
                otel_ctx.detach(token)
            except Exception:
                pass

    # ─── Async hooks (for ainvoke/astream) ─────────────────────────

    async def abefore_agent(self, state, runtime) -> dict[str, Any] | None:
        self._sync_mode = False
        if self._span_processor:
            self._span_processor.set_sync_mode(False)
        from openbox_langchain.middleware_hook_handlers import handle_before_agent
        return await handle_before_agent(self, state, runtime)

    async def aafter_agent(self, state, runtime) -> dict[str, Any] | None:
        from openbox_langchain.middleware_hook_handlers import handle_after_agent
        return await handle_after_agent(self, state, runtime)

    async def awrap_model_call(self, request: ModelRequest, handler) -> Any:
        from openbox_langchain.middleware_hook_handlers import handle_wrap_model_call
        return await handle_wrap_model_call(self, request, handler)

    async def awrap_tool_call(self, request: ToolCallRequest, handler) -> Any:
        from openbox_langchain.middleware_tool_hook import handle_wrap_tool_call
        return await handle_wrap_tool_call(self, request, handler)
