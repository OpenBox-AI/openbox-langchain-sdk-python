"""Tests for middleware.py — OpenBoxLangChainMiddleware class itself.

Covers construction, the M17 per-run turn-state mechanism (turn identity is
now carried BY VALUE in the LangGraph graph state — ``OpenBoxAgentState.ob_turn``
— rather than a ``ContextVar``, since a ContextVar set in ``before_agent`` is
invisible to the wrap hooks through a real graph dispatch: LangGraph runs each
node as a separate task whose contextvars context is copied from the
graph-invocation parent, not from ``before_agent``'s task. The concurrency
regression this state-carriage mechanism fixes is the headline test here —
see ``tests/test_middleware_e2e.py`` for the full real-graph proof), and
close()/aclose().
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from openbox_core.contracts.results import EvaluationResult, Verdict

from openbox_langchain.middleware import (
    OpenBoxLangChainMiddleware,
    OpenBoxLangChainMiddlewareOptions,
)
from openbox_langchain.middleware_turn_state import (
    require_turn_state,
    require_turn_state_from_request,
)
from openbox_langchain.sdk_metadata import SDK_ENGINE, SDK_LANGUAGE, SDK_PACKAGE_VERSION

API_URL = "https://test.openbox.ai"
API_KEY = "obx_test_123"


def make_middleware(**option_overrides) -> OpenBoxLangChainMiddleware:
    options = OpenBoxLangChainMiddlewareOptions(
        api_url=API_URL, api_key=API_KEY, **option_overrides
    )
    return OpenBoxLangChainMiddleware(options)


def state_with_prompt(text: str) -> dict:
    return {"messages": [{"role": "user", "content": text}]}


# ─── Construction ───────────────────────────────────────────────────────


def test_construction_builds_own_runtime_and_workflow_type():
    mw = make_middleware(agent_name="MyAgent")
    try:
        assert mw._workflow_type == "MyAgent"
        assert mw._runtime.config.api_url == API_URL
        assert mw._runtime.config.api_key == API_KEY
        assert mw._runtime.config.sdk_version == SDK_PACKAGE_VERSION
        assert mw._runtime.config.sdk_engine == SDK_ENGINE
        assert mw._runtime.config.sdk_language == SDK_LANGUAGE
    finally:
        mw.close()


def test_construction_default_workflow_type_without_agent_name():
    mw = make_middleware()
    try:
        assert mw._workflow_type == "LangChainRun"
    finally:
        mw.close()


def test_two_instances_have_independent_context_stores():
    """Each middleware instance owns its OWN ContextStore — never a shared global."""
    mw1 = make_middleware()
    mw2 = make_middleware()
    try:
        assert mw1._runtime.context_store is not mw2._runtime.context_store
    finally:
        mw1.close()
        mw2.close()


# ─── require_turn_state / require_turn_state_from_request ────────────────


def test_require_turn_state_raises_when_before_agent_state_has_no_ob_turn():
    """A ``state`` dict that never went through before_agent (no ``ob_turn``
    key) raises — proves the missing-turn-state guard still fires without a
    ContextVar backing it."""
    with pytest.raises(RuntimeError, match="no turn state bound"):
        require_turn_state({"messages": []})


def test_require_turn_state_from_request_raises_when_request_state_has_no_ob_turn():
    """Same guard, but through the wrap-hook path: a ``request.state`` that
    never carried ``ob_turn`` (before_agent never ran for this invocation)."""
    request = MagicMock()
    request.state = {"messages": []}
    with pytest.raises(RuntimeError, match="no turn state bound"):
        require_turn_state_from_request(request)


def test_before_agent_state_update_carries_ob_turn_readable_by_require_turn_state():
    """The dict before_agent/abefore_agent RETURNS is exactly what
    require_turn_state (and, via request.state, require_turn_state_from_request)
    must read back — proving the state-carriage contract end to end without a
    real graph (the real-graph proof lives in test_middleware_e2e.py)."""
    mw = make_middleware()
    try:
        allow = EvaluationResult(verdict=Verdict.ALLOW)
        with patch.object(mw._runtime.gate, "evaluate", return_value=allow):
            state_update = mw.before_agent(state_with_prompt("hello"), object())
            assert state_update is not None
            turn = require_turn_state(state_update)
            assert turn.workflow_id and turn.run_id

            request = MagicMock()
            request.state = state_update
            assert require_turn_state_from_request(request) is turn
    finally:
        mw.close()


# ─── close/aclose ────────────────────────────────────────────────────────


def test_close_is_idempotent():
    mw = make_middleware()
    mw.close()
    mw.close()  # must not raise


async def test_aclose_is_idempotent():
    mw = make_middleware()
    await mw.aclose()
    await mw.aclose()  # must not raise


# ─── M17: per-run turn state, concurrency isolation ──────────────────────
#
# These tests simulate what LangGraph does per-invocation: each "run" owns
# its OWN local `state` dict, updated by abefore_agent's return value and
# threaded explicitly into the later calls of THAT run only — proving turn
# identity/pre-screen verdicts never leak across concurrent runs on ONE
# middleware instance even without a ContextVar or any shared mutable
# instance attribute involved. The full real-graph proof (LangGraph actually
# doing this threading itself, concurrently) lives in test_middleware_e2e.py.


async def test_concurrent_ainvoke_turn_state_does_not_cross_contaminate():
    """Two concurrent abefore_agent/aafter_agent sequences on ONE middleware
    instance must never share workflow_id/run_id — the headline M17
    regression this phase fixes."""
    mw = make_middleware()
    try:
        allow = EvaluationResult(verdict=Verdict.ALLOW)
        with patch.object(mw._runtime.gate, "aevaluate", new=AsyncMock(return_value=allow)):
            seen_workflow_ids: list[str] = []

            async def one_run(prompt: str) -> None:
                state = state_with_prompt(prompt)
                state_update = await mw.abefore_agent(state, object())
                assert state_update is not None
                run_state = {**state, **state_update}
                turn = require_turn_state(run_state)
                seen_workflow_ids.append(turn.workflow_id)
                # Yield control so the two concurrent runs interleave.
                await asyncio.sleep(0)
                # THIS run's local state must still resolve to THIS run's
                # turn identity after the yield — never the sibling run's.
                turn_after_yield = require_turn_state(run_state)
                assert turn_after_yield.workflow_id == turn.workflow_id
                await mw.aafter_agent(run_state, object())

            await asyncio.gather(one_run("run-a"), one_run("run-b"))

            assert len(seen_workflow_ids) == 2
            assert seen_workflow_ids[0] != seen_workflow_ids[1]
    finally:
        mw.close()


async def test_concurrent_ainvoke_pre_screen_verdict_not_shared():
    """Each concurrent run's pre-screen verdict lives on ITS OWN state update
    only — not visible to a sibling run's turn state."""
    mw = make_middleware()
    try:
        allow = EvaluationResult(verdict=Verdict.ALLOW)
        with patch.object(mw._runtime.gate, "aevaluate", new=AsyncMock(return_value=allow)):
            update_a = await mw.abefore_agent(state_with_prompt("hello A"), object())
            assert update_a is not None
            turn_a = require_turn_state(update_a)
            assert turn_a.pre_screen_response is not None

            # A second concurrent "run" binds its OWN turn state via a fresh
            # abefore_agent call (simulating a second concurrent ainvoke) —
            # this must not see or consume run A's pre-screen verdict.
            update_b = await mw.abefore_agent(state_with_prompt("hello B"), object())
            assert update_b is not None
            turn_b = require_turn_state(update_b)
            assert turn_b is not turn_a
            assert turn_b.pre_screen_response is not None
            assert turn_b.workflow_id != turn_a.workflow_id

            # run A's own state update is unaffected by run B ever happening.
            assert require_turn_state(update_a).workflow_id == turn_a.workflow_id
    finally:
        mw.close()
