"""Exposure barriers on LangChain tool calls (PROD-1088).

Drives a real ``create_agent`` graph, and the core callback handler, against
a fake Core on localhost, so every event goes through the base SDK's
evaluate client exactly as in production. The fake Core records each body,
issues a verdict token for a hashed start, and refuses an admission (a
completed event carrying resources) when it is told to.
"""

from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import StructuredTool
from openbox_core import exposure_scope
from openbox_core.errors import GovernanceHaltError
from openbox_core.exposure import Participant, Resource, hash_payload

from openbox_langchain import (
    ActivityBridge,
    OpenBoxLangChainCoreCallbackOptions,
    OpenBoxLangChainCoreSyncCallbackHandler,
    ToolExposure,
    current_tool_call,
    current_verdict_token,
    exposure_tool,
    report_returned,
)

pytest.importorskip("langchain.agents", reason="middleware e2e needs the [agent] extra")
from langchain.agents import create_agent

from openbox_langchain.middleware import (
    OpenBoxLangChainMiddleware,
    OpenBoxLangChainMiddlewareOptions,
)

DOC = Resource("dms://coke/pricing.pdf", ["client:coke"])
DMS = Participant("destination", "dms:main")


class FakeCore:
    def __init__(self) -> None:
        self.bodies: list[dict[str, Any]] = []
        self.refuse_admission: str | None = None  # "block" | "halt"
        core = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a: Any) -> None:
                pass

            def _reply(self, obj: dict[str, Any]) -> None:
                raw = json.dumps(obj).encode()
                self.send_response(200)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def do_GET(self) -> None:
                self._reply({"valid": True})

            def do_POST(self) -> None:
                n = int(self.headers.get("content-length", 0))
                body = json.loads(self.rfile.read(n) or b"{}")
                if "evaluate" not in self.path:
                    self._reply({"valid": True})
                    return
                core.bodies.append(body)
                out: dict[str, Any] = {"verdict": "allow"}
                if body.get("event_type") == "ActivityStarted" and body.get("payload_hash"):
                    out["verdict_token"] = "vt-" + body["payload_hash"][-6:]
                if (body.get("event_type") == "ActivityCompleted" and body.get("resources")
                        and core.refuse_admission):
                    out = {"verdict": core.refuse_admission, "reason": "barrier",
                           "barrier": {"reason_class": "barrier", "remedy": "spawn_clean_instance",
                                       "detail": None}}
                self._reply(out)

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def tool_events(self, tool: str) -> dict[str, dict[str, Any]]:
        return {b["event_type"]: b for b in self.bodies if b.get("activity_type") == tool}


@pytest.fixture
def core() -> Iterator[FakeCore]:
    c = FakeCore()
    yield c
    c.server.shutdown()


class Model(BaseChatModel):
    tool_name: str = "search_docs"

    @property
    def _llm_type(self) -> str:
        return "fake"

    def bind_tools(self, tools: Any, **kwargs: Any) -> Model:
        return self

    def _res(self, messages: list[BaseMessage]) -> ChatResult:
        if any(isinstance(m, ToolMessage) for m in messages):
            msg = AIMessage(content="done")
        else:
            msg = AIMessage(content="", tool_calls=[
                {"name": self.tool_name, "args": {"q": "pricing"}, "id": "c1"}])
        return ChatResult(generations=[ChatGeneration(message=msg)])

    def _generate(self, messages: Any, stop: Any = None, run_manager: Any = None,
                  **kw: Any) -> ChatResult:
        return self._res(messages)

    async def _agenerate(self, messages: Any, stop: Any = None, run_manager: Any = None,
                         **kw: Any) -> ChatResult:
        return self._res(messages)


def _search_tool(seen: dict[str, Any]) -> Any:
    def search_docs(q: str = "") -> tuple[str, list[Resource]]:
        call = current_tool_call()
        seen["token"] = current_verdict_token()
        seen["payload_hash"] = call.payload_hash if call else None
        return "the pricing doc", [DOC]

    tool = StructuredTool.from_function(search_docs, name="search_docs", description="d",
                                        response_format="content_and_artifact")
    return exposure_tool(tool, ToolExposure(
        action_class="retrieve", participants=[DMS],
        payload=lambda args: json.dumps(args, sort_keys=True),
    ))


def _run(core: FakeCore, tools: list[Any], mode: str, **opts: Any) -> dict[str, Any]:
    mw = OpenBoxLangChainMiddleware(OpenBoxLangChainMiddlewareOptions(
        api_url=core.url, api_key="obx_test_exposure_123", **opts))
    agent = create_agent(model=Model(), tools=tools, middleware=[mw])
    msg = {"messages": [{"role": "user", "content": "find pricing"}]}
    try:
        with exposure_scope("matter:coke-2026", purpose="pricing_review"):
            if mode == "async":
                return asyncio.run(agent.ainvoke(msg))
            return agent.invoke(msg)
    finally:
        mw.close()


@pytest.mark.parametrize("mode", ["sync", "async"])
def test_middleware_declares_the_call_hands_back_the_token_and_reports_results(
    core: FakeCore, mode: str
) -> None:
    seen: dict[str, Any] = {}
    _run(core, [_search_tool(seen)], mode)
    events = core.tool_events("search_docs")
    started, completed = events["ActivityStarted"], events["ActivityCompleted"]

    want_hash = hash_payload(json.dumps({"q": "pricing"}, sort_keys=True))
    assert started["scope"] == "matter:coke-2026"
    assert started["action_class"] == "retrieve"
    assert started["participants"] == [{"kind": "destination", "id": "dms:main"}]
    assert started["payload_hash"] == want_hash
    assert "resources" not in started  # nothing declared up front

    assert seen == {"token": "vt-" + want_hash[-6:], "payload_hash": want_hash}

    assert completed["action_class"] == "retrieve"
    assert completed["resources"] == [
        {"uri": DOC.uri, "label_source": "sdk_declared", "labels": ["client:coke"]}]
    assert "payload_hash" not in completed
    assert current_tool_call() is None  # nothing leaks past the call


@pytest.mark.parametrize("mode", ["sync", "async"])
def test_a_refused_admission_never_reaches_the_model(core: FakeCore, mode: str) -> None:
    core.refuse_admission = "block"
    result = _run(core, [_search_tool({})], mode)
    tool_msgs = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_msgs) == 1
    msg = tool_msgs[0]
    assert msg.status == "error"
    assert msg.content == "Refused by an exposure barrier (remedy: spawn_clean_instance)."
    assert "pricing doc" not in str(msg.content) and "client:coke" not in str(msg.content)
    assert not msg.artifact


def test_a_halted_admission_stops_the_run(core: FakeCore) -> None:
    core.refuse_admission = "halt"
    with pytest.raises(GovernanceHaltError):
        _run(core, [_search_tool({})], "sync")


def test_the_options_registry_and_report_returned(core: FakeCore) -> None:
    def search_docs(q: str = "") -> str:
        report_returned(DOC)
        return "the pricing doc"

    tool = StructuredTool.from_function(search_docs, name="search_docs", description="d")
    _run(core, [tool], "sync", tool_exposure={
        "search_docs": ToolExposure(action_class="read", resources=lambda a: [DOC])})
    events = core.tool_events("search_docs")
    assert events["ActivityStarted"]["action_class"] == "read"
    assert events["ActivityStarted"]["resources"][0]["uri"] == DOC.uri
    assert "payload_hash" not in events["ActivityStarted"]
    assert events["ActivityCompleted"]["resources"][0]["labels"] == ["client:coke"]


def test_an_undeclared_tool_is_unchanged(core: FakeCore) -> None:
    tool = StructuredTool.from_function(lambda q="": "x", name="search_docs", description="d")
    _run(core, [tool], "sync")
    events = core.tool_events("search_docs")
    for e in events.values():
        assert e["scope"] == "matter:coke-2026"
        assert not {"action_class", "resources", "payload_hash"} & e.keys()


def test_report_returned_outside_a_tool_call_raises() -> None:
    with pytest.raises(RuntimeError):
        report_returned(DOC)


def test_unknown_action_class_is_refused() -> None:
    with pytest.raises(ValueError):
        ToolExposure(action_class="teleport")


def test_callback_handler_declares_binds_and_reports(core: FakeCore) -> None:
    mw = OpenBoxLangChainMiddleware(OpenBoxLangChainMiddlewareOptions(
        api_url=core.url, api_key="obx_test_exposure_123"))
    seen: dict[str, Any] = {}
    try:
        options = OpenBoxLangChainCoreCallbackOptions(
            runtime=mw._runtime, bridge=ActivityBridge(), workflow_id="wf", run_id="run",
            workflow_type="t", tool_exposure={"lookup": ToolExposure(
                action_class="retrieve", participants=[DMS],
                payload=lambda a: json.dumps(a, sort_keys=True))},
        )

        def lookup(q: str = "") -> str:
            seen["token"] = current_verdict_token()
            report_returned(DOC)
            return "found"

        tool = StructuredTool.from_function(lookup, name="lookup", description="d")
        with exposure_scope("matter:coke-2026"):
            tool.run({"q": "pricing"}, callbacks=[OpenBoxLangChainCoreSyncCallbackHandler(options)])
    finally:
        mw.close()
    events = core.tool_events("lookup")
    want_hash = hash_payload(json.dumps({"q": "pricing"}, sort_keys=True))
    assert events["ActivityStarted"]["payload_hash"] == want_hash
    assert events["ActivityStarted"]["scope"] == "matter:coke-2026"
    assert seen["token"] == "vt-" + want_hash[-6:]
    assert events["ActivityCompleted"]["resources"][0]["uri"] == DOC.uri
    assert options.tool_calls == {} and current_tool_call() is None
