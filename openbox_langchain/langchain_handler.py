"""OpenBox governance callback handler for LangChain agents.

Extends AsyncCallbackHandler to intercept LangChain lifecycle events,
map them to OpenBox governance events, and enforce verdicts (block/halt/
require_approval). Reuses all governance infrastructure from the shared
modules (client, verdict_handler, hook_governance, etc.).
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.messages import BaseMessage

from openbox_langchain.client import GovernanceClient
from openbox_langchain.config import (
    GovernanceConfig,
    get_global_config,
    merge_config,
)
from openbox_langchain.hitl import HITLPollParams, poll_until_decision
from openbox_langchain.types import (
    GovernanceVerdictResponse,
    HITLConfig,
    LangChainGovernanceEvent,
    rfc3339_now,
    safe_serialize,
)
from openbox_langchain.verdict_handler import enforce_verdict

try:
    from opentelemetry import context as otel_context
    from opentelemetry import trace as otel_trace

    _HAS_OTEL = True
except ImportError:
    _HAS_OTEL = False

logger = logging.getLogger("openbox_langchain")


# ═══════════════════════════════════════════════════════════════════
# Internal state tracking
# ═══════════════════════════════════════════════════════════════════


@dataclass
class _RunBuffer:
    """Tracks in-flight run metadata for duration calculation."""

    run_id: str
    run_type: str  # "chain" | "tool" | "llm" | "agent"
    name: str
    start_time_ms: float = field(default_factory=lambda: time.monotonic() * 1000)
    otel_span: Any = None
    otel_token: Any = None


class _RunBufferManager:
    """Registry of in-flight runs keyed by run_id string."""

    def __init__(self) -> None:
        self._runs: dict[str, _RunBuffer] = {}

    def register(self, run_id: str, run_type: str, name: str) -> _RunBuffer:
        buf = _RunBuffer(run_id=run_id, run_type=run_type, name=name)
        self._runs[run_id] = buf
        return buf

    def get(self, run_id: str) -> _RunBuffer | None:
        return self._runs.get(run_id)

    def remove(self, run_id: str) -> None:
        self._runs.pop(run_id, None)

    def duration_ms(self, run_id: str) -> float | None:
        buf = self._runs.get(run_id)
        if buf is None:
            return None
        return time.monotonic() * 1000 - buf.start_time_ms

    def clear(self) -> None:
        self._runs.clear()


# ═══════════════════════════════════════════════════════════════════
# Configuration dataclass
# ═══════════════════════════════════════════════════════════════════


@dataclass
class OpenBoxLangChainHandlerOptions:
    """Configuration options for OpenBoxGovernanceCallbackHandler."""

    client: GovernanceClient | None = None
    on_api_error: str = "fail_open"
    api_timeout: int = 30_000
    send_chain_start_event: bool = True
    send_chain_end_event: bool = True
    send_tool_start_event: bool = True
    send_tool_end_event: bool = True
    send_llm_start_event: bool = True
    send_llm_end_event: bool = True
    skip_chain_types: set[str] = field(default_factory=set)
    skip_tool_types: set[str] = field(default_factory=set)
    hitl: HITLConfig | dict | None = None
    session_id: str | None = None
    agent_name: str | None = None
    task_queue: str = "langchain"
    tool_type_map: dict[str, str] | None = None
    sqlalchemy_engine: Any = None


# ═══════════════════════════════════════════════════════════════════
# Main callback handler
# ═══════════════════════════════════════════════════════════════════


class OpenBoxGovernanceCallbackHandler(AsyncCallbackHandler):
    """OpenBox governance callback handler for LangChain agents.

    Intercepts chain/tool/LLM/agent lifecycle events, maps them to
    OpenBox governance events, and enforces verdicts. Attach via:

        result = agent.invoke(input, config={"callbacks": [handler]})
    """

    raise_error = True  # Propagate exceptions through AgentExecutor

    def __init__(self, options: OpenBoxLangChainHandlerOptions | None = None) -> None:
        super().__init__()
        opts = options or OpenBoxLangChainHandlerOptions()

        # Build GovernanceConfig from options
        self._config: GovernanceConfig = merge_config(
            {
                "on_api_error": opts.on_api_error,
                "api_timeout": opts.api_timeout,
                "send_chain_start_event": opts.send_chain_start_event,
                "send_chain_end_event": opts.send_chain_end_event,
                "send_tool_start_event": opts.send_tool_start_event,
                "send_tool_end_event": opts.send_tool_end_event,
                "send_llm_start_event": opts.send_llm_start_event,
                "send_llm_end_event": opts.send_llm_end_event,
                "skip_chain_types": opts.skip_chain_types,
                "skip_tool_types": opts.skip_tool_types,
                "hitl": opts.hitl,
                "session_id": opts.session_id,
                "agent_name": opts.agent_name,
                "task_queue": opts.task_queue,
                "tool_type_map": opts.tool_type_map or {},
            }
        )

        # Client: use provided or build from global config
        if opts.client:
            self._client = opts.client
        else:
            gc = get_global_config()
            self._client = GovernanceClient(
                api_url=gc.api_url,
                api_key=gc.api_key,
                timeout=gc.governance_timeout,
                on_api_error=self._config.on_api_error,
            )

        self._sqlalchemy_engine = opts.sqlalchemy_engine

        # Workflow state — reset between invocations
        self._reset_state()

        # OTel setup for Layer 2/3 hooks
        self._span_processor = None
        self._setup_otel_hooks()

    # ─── OTel / Hook Governance Setup ───────────────────────────────

    def _setup_otel_hooks(self) -> None:
        """Initialize OpenTelemetry instrumentation for Layer 2/3 hook governance."""
        gc = get_global_config()
        if not gc or not gc.api_url or not gc.api_key:
            return
        try:
            from openbox_langchain.otel_setup import setup_opentelemetry_for_governance
            from openbox_langchain.span_processor import WorkflowSpanProcessor

            self._span_processor = WorkflowSpanProcessor()
            setup_opentelemetry_for_governance(
                span_processor=self._span_processor,
                api_url=gc.api_url,
                api_key=gc.api_key,
                ignored_urls=[gc.api_url],
                api_timeout=gc.governance_timeout,
                on_api_error=self._config.on_api_error,
                sqlalchemy_engine=self._sqlalchemy_engine,
            )
        except Exception:
            logger.warning("Failed to initialize OTel hooks; Layer 2/3 governance disabled")

    # ─── State Management ───────────────────────────────────────────

    def _reset_state(self) -> None:
        """Reset handler state for reuse across multiple invocations."""
        self._root_run_id: str | None = None
        self._workflow_id: str | None = None
        self._run_id: str | None = None
        self._workflow_started: bool = False
        self._pre_screen_response: GovernanceVerdictResponse | None = None
        self._pre_screen_activity_id: str | None = None
        self._llm_activity_map: dict[str, str] = {}
        self._seen_tool_run_ids: set[str] = set()
        self._buffer = _RunBufferManager()

    # ─── Shared verdict enforcement + HITL polling ──────────────────

    async def _enforce_and_poll(
        self,
        response: GovernanceVerdictResponse,
        context: str,
        activity_id: str,
        activity_type: str,
    ) -> None:
        """Enforce verdict and poll for HITL approval if required."""
        result = enforce_verdict(response, context)
        if result.requires_hitl:
            await poll_until_decision(
                self._client,
                HITLPollParams(
                    workflow_id=self._workflow_id or "",
                    run_id=self._run_id or "",
                    activity_id=activity_id,
                    activity_type=activity_type,
                ),
                self._config.hitl,
            )

    # ─── Pre-Screen ───────────────────────────────────────────────

    async def _do_pre_screen(self, input_data: dict[str, Any]) -> None:
        """Shared pre-screen logic: SignalReceived + WorkflowStarted + LLMStarted."""
        wf_uuid = uuid.uuid4().hex[:8]
        self._workflow_id = f"lc-{wf_uuid}"
        self._run_id = f"lc-{wf_uuid}-run-{uuid.uuid4().hex[:8]}"

        user_prompt = _extract_user_prompt(input_data)

        if user_prompt:
            sig_event = self._build_event(
                event_type="SignalReceived",
                activity_id=f"{self._run_id}-sig",
                activity_type="user_prompt",
                signal_name="user_prompt",
                signal_args=[user_prompt],
            )
            await self._client.evaluate_event(sig_event)

        wf_event = self._build_event(
            event_type="WorkflowStarted",
            activity_id=f"{self._run_id}-wf",
            activity_type=self._config.agent_name or "LangChainRun",
            activity_input=[safe_serialize(input_data)],
        )
        await self._client.evaluate_event(wf_event)
        self._workflow_started = True

        if self._config.send_llm_start_event and user_prompt:
            pre_id = f"{self._run_id}-pre"
            pre_event = self._build_event(
                event_type="LLMStarted",
                activity_id=pre_id,
                activity_type="llm_call",
                activity_input=[{"prompt": user_prompt}],
                prompt=user_prompt,
            )
            response = await self._client.evaluate_event(pre_event)
            if response:
                try:
                    await self._enforce_and_poll(
                        response, "llm_start", pre_id, "llm_call"
                    )
                except Exception as exc:
                    if self._workflow_started:
                        await self._send_workflow_completed(
                            status="failed", error=str(exc)
                        )
                    raise
                self._pre_screen_response = response
                self._pre_screen_activity_id = pre_id

    async def pre_screen(self, input_data: dict[str, Any]) -> None:
        """Pre-screen user input BEFORE calling agent.invoke().

        Sends SignalReceived + WorkflowStarted + LLMStarted to Core and
        enforces verdicts. Call this for reliable blocking/HITL before
        the agent starts — callback exceptions may not propagate through
        AgentExecutor.

        Args:
            input_data: The same input dict you'll pass to agent.invoke().

        Raises:
            GovernanceBlockedError: If policy blocks the input.
            GovernanceHaltError: If policy halts the workflow.
            GuardrailsValidationError: If guardrails reject the input.
        """
        await self._do_pre_screen(input_data)

    # ─── Chain Callbacks ────────────────────────────────────────────

    async def on_chain_start(
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        str_run_id = str(run_id)
        chain_name = _extract_name(serialized)

        # Root detection: first chain_start with no parent
        if parent_run_id is None and self._root_run_id is None:
            self._root_run_id = str_run_id
            self._buffer.register(str_run_id, "chain", chain_name)

            # If pre_screen() wasn't called, do inline pre-screen
            if not self._workflow_started:
                await self._do_pre_screen(inputs)
            return

        # Nested chain (non-root)
        if not self._config.send_chain_start_event:
            return
        if chain_name in self._config.skip_chain_types:
            return

        self._buffer.register(str_run_id, "chain", chain_name)
        gov = self._build_event(
            event_type="ChainStarted",
            activity_id=str_run_id,
            activity_type=chain_name,
            activity_input=[safe_serialize(inputs)] if inputs else None,
        )
        response = await self._client.evaluate_event(gov)
        if response:
            enforce_verdict(response, "chain_start")

    async def on_chain_end(
        self,
        outputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        str_run_id = str(run_id)
        is_root = str_run_id == self._root_run_id
        dur = self._buffer.duration_ms(str_run_id)
        self._buffer.remove(str_run_id)

        if is_root:
            if self._config.send_chain_end_event and self._workflow_started:
                await self._send_workflow_completed(
                    status="completed", output=outputs, duration_ms=dur
                )
            self._reset_state()
            return

        # Nested chain — observation only
        if not self._config.send_chain_end_event:
            return
        gov = self._build_event(
            event_type="ChainCompleted",
            activity_id=str_run_id,
            activity_output=safe_serialize(outputs),
            status="completed",
            duration_ms=dur,
        )
        await self._client.evaluate_event(gov)

    async def on_chain_error(
        self, error: BaseException, *, run_id: UUID, **kwargs: Any
    ) -> None:
        str_run_id = str(run_id)
        if str_run_id == self._root_run_id and self._workflow_started:
            await self._send_workflow_completed(status="failed", error=str(error))
            self._reset_state()
        self._buffer.remove(str_run_id)

    # ─── Tool Callbacks ─────────────────────────────────────────────

    async def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        if not self._config.send_tool_start_event:
            return
        str_run_id = str(run_id)
        tool_name = _extract_name(serialized)
        if tool_name in self._config.skip_tool_types:
            return

        # Dedup: skip governance if already seen via on_agent_action (same parent run_id)
        parent_str = str(parent_run_id) if parent_run_id else None
        if parent_str and parent_str in self._seen_tool_run_ids:
            self._seen_tool_run_ids.discard(parent_str)
            # Still register buffer for duration tracking
            self._buffer.register(str_run_id, "tool", tool_name)
            return

        self._buffer.register(str_run_id, "tool", tool_name)

        # Register activity context with SpanProcessor for Layer 2 hooks
        if self._span_processor is not None:
            activity_context = {
                "source": "workflow-telemetry",
                "event_type": "ActivityStarted",
                "workflow_id": self._workflow_id,
                "run_id": self._run_id,
                "workflow_type": self._config.agent_name or "LangChainRun",
                "task_queue": self._config.task_queue or "langchain",
                "activity_id": str_run_id,
                "activity_type": tool_name,
            }
            self._span_processor.set_activity_context(
                self._workflow_id, str_run_id, activity_context
            )

        # Create OTel span for trace context propagation
        self._create_otel_span_for_tool(str_run_id, tool_name)

        # Resolve tool_type from tool_type_map
        tool_type = self._config.tool_type_map.get(tool_name)
        tool_input = _unwrap_tool_input(input_str)

        gov = self._build_event(
            event_type="ToolStarted",
            activity_id=str_run_id,
            activity_type=tool_name,
            activity_input=[safe_serialize(tool_input)],
            tool_name=tool_name,
            tool_type=tool_type,
        )
        response = await self._client.evaluate_event(gov)
        if response:
            await self._enforce_and_poll(
                response, "tool_start", str_run_id, tool_name
            )

    async def on_tool_end(
        self, output: str, *, run_id: UUID, **kwargs: Any
    ) -> None:
        if not self._config.send_tool_end_event:
            return
        str_run_id = str(run_id)
        buf = self._buffer.get(str_run_id)
        dur = self._buffer.duration_ms(str_run_id)

        # Clean up OTel span
        self._cleanup_otel_span(str_run_id)

        tool_name = buf.name if buf else "Tool"
        self._buffer.remove(str_run_id)

        gov = self._build_event(
            event_type="ToolCompleted",
            activity_id=str_run_id,
            activity_type=tool_name,
            activity_output=safe_serialize(output),
            status="completed",
            duration_ms=dur,
        )
        response = await self._client.evaluate_event(gov)
        if response:
            await self._enforce_and_poll(
                response, "tool_end", str_run_id, tool_name
            )

    async def on_tool_error(
        self, error: BaseException, *, run_id: UUID, **kwargs: Any
    ) -> None:
        str_run_id = str(run_id)
        self._cleanup_otel_span(str_run_id)
        self._buffer.remove(str_run_id)

    # ─── LLM Callbacks ──────────────────────────────────────────────

    async def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        if not self._config.send_llm_start_event:
            return

        prompt_text = _extract_prompt_from_messages(messages)
        if not prompt_text.strip():
            return

        str_run_id = str(run_id)
        model_name = _extract_name(serialized)

        # Reuse pre-screen response if available (first LLM call after root chain)
        if self._pre_screen_response is not None:
            response = self._pre_screen_response
            self._pre_screen_response = None
            if self._pre_screen_activity_id:
                self._llm_activity_map[str_run_id] = self._pre_screen_activity_id
        else:
            # Subsequent LLM calls — send new LLMStarted event
            gov = self._build_event(
                event_type="LLMStarted",
                activity_id=str_run_id,
                activity_type="llm_call",
                activity_input=[{"prompt": prompt_text}],
                llm_model=model_name,
                prompt=prompt_text,
            )
            response = await self._client.evaluate_event(gov)
            self._llm_activity_map[str_run_id] = str_run_id

        if response is None:
            return

        # PII redaction: mutate messages in-place before LLM sees content.
        # Verdict enforcement is NOT done here — pre_screen() handles blocking.
        # This callback's job is PII redaction only (same as LangGraph SDK).
        _apply_pii_redaction(messages, response)

    async def on_llm_end(self, response: Any, *, run_id: UUID, **kwargs: Any) -> None:
        if not self._config.send_llm_end_event:
            return
        str_run_id = str(run_id)
        llm_activity_id = self._llm_activity_map.pop(str_run_id, str_run_id)

        output_text, input_tokens, output_tokens, total_tokens = _extract_llm_output(
            response
        )

        completed_event = self._build_event(
            event_type="LLMCompleted",
            activity_id=f"{llm_activity_id}-c",
            activity_type="llm_call",
            activity_output=safe_serialize(output_text),
            status="completed",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            completion=output_text,
        )
        # Observation only — no verdict enforcement
        await self._client.evaluate_event(completed_event)

    async def on_llm_error(
        self, error: BaseException, *, run_id: UUID, **kwargs: Any
    ) -> None:
        self._llm_activity_map.pop(str(run_id), None)

    # ─── Agent Callbacks ────────────────────────────────────────────

    async def on_agent_action(
        self, action: Any, *, run_id: UUID, **kwargs: Any
    ) -> None:
        """Fired by AgentExecutor before tool dispatch. Enforce verdict."""
        tool_name = getattr(action, "tool", "Tool")
        tool_input = getattr(action, "tool_input", {})
        str_run_id = str(run_id)

        # Mark this run_id to prevent duplicate in on_tool_start
        self._seen_tool_run_ids.add(str_run_id)

        gov = self._build_event(
            event_type="ToolStarted",
            activity_id=f"{str_run_id}-action",
            activity_type=tool_name,
            activity_input=[safe_serialize(tool_input)],
            tool_name=tool_name,
        )
        response = await self._client.evaluate_event(gov)
        if response:
            await self._enforce_and_poll(
                response, "agent_action", f"{str_run_id}-action", tool_name
            )

    # ─── Event Builder ──────────────────────────────────────────────

    def _build_event(self, **fields: Any) -> LangChainGovernanceEvent:
        """Build a governance event with common fields pre-filled."""
        return LangChainGovernanceEvent(
            source="workflow-telemetry",
            workflow_id=self._workflow_id or "",
            run_id=self._run_id or "",
            workflow_type=self._config.agent_name or "LangChainRun",
            task_queue=self._config.task_queue,
            timestamp=rfc3339_now(),
            session_id=self._config.session_id,
            **fields,
        )

    async def _send_workflow_completed(
        self,
        *,
        status: str,
        output: Any = None,
        error: str | None = None,
        duration_ms: float | None = None,
    ) -> None:
        """Send WorkflowCompleted event (observation only)."""
        gov = self._build_event(
            event_type="WorkflowCompleted",
            activity_id=f"{self._run_id}-wf",
            activity_type=self._config.agent_name or "LangChainRun",
            activity_output=safe_serialize(output) if output else None,
            status=status,
            error={"message": error, "type": "error"} if error else None,
            duration_ms=duration_ms,
        )
        await self._client.evaluate_event(gov)

    # ─── OTel Span Helpers ──────────────────────────────────────────

    def _create_otel_span_for_tool(self, str_run_id: str, tool_name: str) -> None:
        """Create OTel span for tool execution to bridge trace context to hooks."""
        if self._span_processor is None or not _HAS_OTEL:
            return
        try:
            tracer = otel_trace.get_tracer("openbox_langchain")
            parent_ctx = otel_context.get_current()
            tool_span = tracer.start_span(f"tool.{tool_name}", context=parent_ctx)
            token = otel_context.attach(otel_trace.set_span_in_context(tool_span))

            trace_id = tool_span.get_span_context().trace_id
            if trace_id:
                self._span_processor.set_trace_mapping(
                    trace_id, self._workflow_id, str_run_id
                )

            buf = self._buffer.get(str_run_id)
            if buf:
                buf.otel_span = tool_span
                buf.otel_token = token
        except Exception:
            logger.debug("OTel span creation failed for tool %s", tool_name)

    def _cleanup_otel_span(self, str_run_id: str) -> None:
        """End OTel span and detach context for a tool run."""
        buf = self._buffer.get(str_run_id)
        if not buf:
            return
        if buf.otel_span:
            try:
                buf.otel_span.end()
            except Exception:
                logger.debug("OTel span end failed for %s", str_run_id)
        if buf.otel_token and _HAS_OTEL:
            try:
                otel_context.detach(buf.otel_token)
            except Exception:
                logger.debug("OTel context detach failed for %s", str_run_id)


# ═══════════════════════════════════════════════════════════════════
# Private utility functions
# ═══════════════════════════════════════════════════════════════════


def _extract_name(serialized: dict[str, Any]) -> str:
    """Extract human-readable name from LangChain serialized metadata."""
    return (
        serialized.get("name")
        or (serialized.get("id") or [None])[-1]
        or "Unknown"
    )


def _extract_user_prompt(inputs: dict[str, Any]) -> str | None:
    """Extract last human/user message from chain inputs."""
    # AgentExecutor pattern: {"input": "...", "chat_history": [...]}
    if isinstance(inputs.get("input"), str):
        return inputs["input"]
    # Messages pattern: {"messages": [HumanMessage(...)]}
    messages = inputs.get("messages")
    if isinstance(messages, list):
        for msg in reversed(messages):
            if hasattr(msg, "content") and getattr(msg, "type", None) in (
                "human",
                "user",
            ):
                return msg.content if isinstance(msg.content, str) else str(msg.content)
            if isinstance(msg, dict) and msg.get("role") in ("human", "user"):
                return str(msg.get("content", ""))
    return None


def _extract_prompt_from_messages(messages: list[list[BaseMessage]]) -> str:
    """Extract human/user message text from LLM messages (same as LangGraph SDK)."""
    prompt_parts: list[str] = []
    for group in messages:
        for msg in group:
            role = getattr(msg, "type", None) or getattr(msg, "role", None) or ""
            if role not in ("human", "user", "generic"):
                continue
            content = msg.content
            if isinstance(content, str):
                prompt_parts.append(content)
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        prompt_parts.append(part.get("text", ""))
    return "\n".join(prompt_parts)


def _apply_pii_redaction(
    messages: list[list[BaseMessage]], response: GovernanceVerdictResponse
) -> None:
    """Mutate messages in-place with PII-redacted content from Core."""
    gr = response.guardrails_result
    if not gr or gr.input_type != "activity_input" or gr.redacted_input is None:
        return

    redacted = gr.redacted_input
    redacted_text: str | None = None
    if isinstance(redacted, list) and redacted:
        first = redacted[0]
        if isinstance(first, dict):
            redacted_text = first.get("prompt")
        elif isinstance(first, str):
            redacted_text = first
    elif isinstance(redacted, str):
        redacted_text = redacted

    if not redacted_text:
        return

    # Replace the last human message in each message group
    for group in messages:
        for j in range(len(group) - 1, -1, -1):
            msg = group[j]
            if getattr(msg, "type", None) in ("human", "generic"):
                msg.content = redacted_text  # type: ignore[assignment]
                break


def _extract_llm_output(response: Any) -> tuple[str, int | None, int | None, int | None]:
    """Extract output text and token usage from LLMResult."""
    output_text = ""
    input_tokens = output_tokens = total_tokens = None
    generations = getattr(response, "generations", None)
    if generations and generations[0]:
        gen = generations[0][0]
        output_text = getattr(gen, "text", "") or ""
    llm_output = getattr(response, "llm_output", None) or {}
    usage = llm_output.get("token_usage") or llm_output.get("usage") or {}
    if usage:
        input_tokens = usage.get("prompt_tokens")
        output_tokens = usage.get("completion_tokens")
        total_tokens = usage.get("total_tokens")
    return output_text, input_tokens, output_tokens, total_tokens


def _unwrap_tool_input(raw: Any) -> Any:
    """Unwrap potentially double-encoded JSON tool input."""
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return raw
    return raw


# ═══════════════════════════════════════════════════════════════════
# Factory function
# ═══════════════════════════════════════════════════════════════════


def create_openbox_langchain_handler(
    *,
    api_url: str,
    api_key: str,
    governance_timeout: float = 30.0,
    validate: bool = True,
    enable_telemetry: bool = True,
    sqlalchemy_engine: Any = None,
    **handler_kwargs: Any,
) -> OpenBoxGovernanceCallbackHandler:
    """Create a configured OpenBoxGovernanceCallbackHandler.

    Validates credentials, initializes global config, returns handler
    ready for use via config={"callbacks": [handler]}.

    Args:
        api_url: Base URL of OpenBox Core instance.
        api_key: API key (obx_live_* or obx_test_* format).
        governance_timeout: HTTP timeout in seconds (default 30.0).
        validate: If True, validates API key against server on startup.
        enable_telemetry: If True, enables OTel hook governance (Layer 2/3).
        sqlalchemy_engine: Optional SQLAlchemy Engine for DB governance.
        **handler_kwargs: Forwarded to OpenBoxLangChainHandlerOptions.

    Returns:
        Configured OpenBoxGovernanceCallbackHandler.

    Example:
        >>> handler = create_openbox_langchain_handler(
        ...     api_url=os.environ["OPENBOX_URL"],
        ...     api_key=os.environ["OPENBOX_API_KEY"],
        ...     agent_name="MyAgent",
        ... )
        >>> result = agent.invoke(
        ...     {"input": "Hello"},
        ...     config={"callbacks": [handler]},
        ... )
    """
    from openbox_langchain.config import initialize

    initialize(
        api_url=api_url,
        api_key=api_key,
        governance_timeout=governance_timeout,
        validate=validate,
    )

    known_fields = set(OpenBoxLangChainHandlerOptions.__dataclass_fields__)
    filtered_kwargs = {k: v for k, v in handler_kwargs.items() if k in known_fields}

    options = OpenBoxLangChainHandlerOptions(
        api_timeout=int(governance_timeout * 1000),
        sqlalchemy_engine=sqlalchemy_engine,
        **filtered_kwargs,
    )
    return OpenBoxGovernanceCallbackHandler(options)
