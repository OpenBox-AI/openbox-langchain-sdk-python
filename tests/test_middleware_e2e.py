"""End-to-end turn-state invariant tests for OpenBoxLangChainMiddleware.

The M17 per-run turn state lives in a ContextVar set by ``before_agent`` and
read by the wrap hooks. The unit tests call the hooks directly; these tests
drive a REAL ``create_agent(middleware=[mw]).invoke()/.ainvoke()`` so the
invariant "the wrap hooks run in the contextvars context ``before_agent`` set"
is proven against the actual LangGraph dispatch (sync AND async), and the
concurrency isolation is proven through the real graph — not a hand-called
hook sequence.

No network: the middleware's gate is patched to ALLOW.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import StructuredTool
from openbox_core.contracts.results import EvaluationResult, Verdict

from openbox_langchain.middleware import (
    OpenBoxLangChainMiddleware,
    OpenBoxLangChainMiddlewareOptions,
)

pytest.importorskip("langchain.agents", reason="middleware e2e needs the [agent] extra")
from langchain.agents import create_agent

API_URL = "https://test.openbox.ai"
API_KEY = "obx_test_123"


class FakeToolCallingModel(BaseChatModel):
    """Emits one tool call, then a final answer once a ToolMessage is present."""

    tool_name: str = "echo_tool"

    @property
    def _llm_type(self) -> str:
        return "fake-tool-calling"

    def bind_tools(self, tools: Any, **kwargs: Any) -> FakeToolCallingModel:
        return self

    def _make_result(self, messages: list[BaseMessage]) -> ChatResult:
        has_tool_msg = any(isinstance(m, ToolMessage) for m in messages)
        if has_tool_msg:
            msg = AIMessage(content="final answer")
        else:
            msg = AIMessage(
                content="",
                tool_calls=[{"name": self.tool_name, "args": {"text": "hi"}, "id": "call_1"}],
            )
        return ChatResult(generations=[ChatGeneration(message=msg)])

    def _generate(self, messages: list[BaseMessage], stop: Any = None,
                  run_manager: Any = None, **kwargs: Any) -> ChatResult:
        return self._make_result(messages)

    async def _agenerate(self, messages: list[BaseMessage], stop: Any = None,
                         run_manager: Any = None, **kwargs: Any) -> ChatResult:
        return self._make_result(messages)


def _echo_tool() -> StructuredTool:
    calls: dict[str, int] = {"n": 0}

    def _echo(text: str = "") -> str:
        calls["n"] += 1
        return f"echoed: {text}"

    tool = StructuredTool.from_function(func=_echo, name="echo_tool", description="echo")
    tool.__dict__["_calls"] = calls
    return tool


def _allow_middleware() -> OpenBoxLangChainMiddleware:
    """A real middleware whose gate always ALLOWs (no network)."""
    mw = OpenBoxLangChainMiddleware(
        OpenBoxLangChainMiddlewareOptions(api_url=API_URL, api_key=API_KEY)
    )
    allow = EvaluationResult(verdict=Verdict.ALLOW)
    mw._runtime.gate.aevaluate = AsyncMock(return_value=allow)  # type: ignore[method-assign]
    mw._runtime.gate.evaluate = MagicMock(return_value=allow)  # type: ignore[method-assign]
    return mw


def _build_agent(mw: OpenBoxLangChainMiddleware):
    return create_agent(
        model=FakeToolCallingModel(),
        tools=[_echo_tool()],
        middleware=[mw],
    )


def _final_text(result: dict) -> str:
    msgs = result["messages"]
    return msgs[-1].content if msgs else ""


def test_sync_invoke_binds_turn_state_in_wrap_hooks() -> None:
    """H1: through the REAL sync graph, the wrap hooks must see the turn state
    bound by before_agent — no `no turn state bound` RuntimeError — and the
    tool body must run under an ALLOW verdict."""
    mw = _allow_middleware()
    try:
        agent = _build_agent(mw)
        result = agent.invoke({"messages": [{"role": "user", "content": "hello"}]})
        assert _final_text(result) == "final answer"
        # Gate saw the sync lifecycle sends (proves wrap hooks ran with turn state).
        assert mw._runtime.gate.evaluate.called
    finally:
        mw.close()


async def test_async_invoke_binds_turn_state_in_wrap_hooks() -> None:
    """H1 async: same invariant through the REAL async graph."""
    mw = _allow_middleware()
    try:
        agent = _build_agent(mw)
        result = await agent.ainvoke({"messages": [{"role": "user", "content": "hello"}]})
        assert _final_text(result) == "final answer"
        assert mw._runtime.gate.aevaluate.called
    finally:
        mw.close()


async def test_concurrent_ainvoke_through_real_graph_no_cross_contamination() -> None:
    """H1 + M17: two concurrent .ainvoke() on ONE middleware instance both
    complete correctly through the real graph (turn state isolated per run)."""
    mw = _allow_middleware()
    try:
        agent = _build_agent(mw)
        results = await asyncio.gather(
            agent.ainvoke({"messages": [{"role": "user", "content": "A"}]}),
            agent.ainvoke({"messages": [{"role": "user", "content": "B"}]}),
        )
        assert all(_final_text(r) == "final answer" for r in results)
    finally:
        mw.close()
