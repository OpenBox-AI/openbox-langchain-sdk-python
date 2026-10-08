"""BLOCK recovers at the tool boundary; HALT escapes the real agent graph."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import ToolMessage
from langchain_core.tools import StructuredTool
from openbox_core.contracts.results import EvaluationResult, Verdict
from openbox_core.errors import GovernanceBlockedError, GovernanceHaltError

from tests.test_middleware_e2e import FakeToolCallingModel, _allow_middleware, create_agent


@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize("inside_tool", [False, True])
@pytest.mark.parametrize("verdict", [Verdict.BLOCK, Verdict.HALT])
async def test_tool_verdict_through_graph(async_mode, inside_tool, verdict):
    mw = _allow_middleware()
    events = []
    steps = []

    def evaluate(event):
        events.append(event)
        if (not inside_tool and event.event_type.value == "ActivityStarted"
                and event.activity_type == "echo_tool"):
            return EvaluationResult(verdict=verdict, reason="private policy detail")
        return EvaluationResult(verdict=Verdict.ALLOW)

    def governed_tool(text: str) -> str:
        steps.append("entered")
        if inside_tool:
            if verdict is Verdict.HALT:
                raise GovernanceHaltError("private policy detail")
            raise GovernanceBlockedError(verdict, "private policy detail")
        steps.append("side effect")
        return text

    mw._runtime.gate.evaluate = MagicMock(side_effect=evaluate)
    mw._runtime.gate.aevaluate = AsyncMock(side_effect=evaluate)
    tool = StructuredTool.from_function(
        func=governed_tool, name="echo_tool", description="Governed action",
    )
    agent = create_agent(model=FakeToolCallingModel(), tools=[tool], middleware=[mw])
    inputs = {"messages": [{"role": "user", "content": "hello"}]}

    async def invoke():
        return await agent.ainvoke(inputs) if async_mode else agent.invoke(inputs)

    try:
        if verdict is Verdict.HALT:
            with pytest.raises(GovernanceHaltError):
                await invoke()
        else:
            result = await invoke()
            assert result["messages"][-1].content == "final answer"
            messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
            assert len(messages) == 1
            assert messages[0].tool_call_id == "call_1"
            assert messages[0].status == "error"
            assert "private policy detail" not in messages[0].content
        assert steps == (["entered"] if inside_tool else [])
        tool_events = [e for e in events if e.activity_type == "echo_tool"]
        assert [e.event_type.value for e in tool_events] == [
            "ActivityStarted", "ActivityCompleted",
        ]
        assert tool_events[0].activity_id == tool_events[1].activity_id
        assert tool_events[1].payload["error"]
    finally:
        mw.close()
