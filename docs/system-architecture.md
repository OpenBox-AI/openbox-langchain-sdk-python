# System Architecture — OpenBox LangChain SDK

## Overview

Three-layer governance architecture for LangChain agents using `AgentMiddleware` (v1.0+) for lifecycle interception:

1. **Layer 1:** AgentMiddleware hooks intercept agent execution
2. **Layer 2:** Hook governance intercepts I/O at kernel boundary (HTTP/DB/File)
3. **Layer 3:** OTel context mapping attributes I/O to governance activities

## Layer 1: AgentMiddleware Governance

`OpenBoxLangChainMiddleware` extends LangChain's `AgentMiddleware` and intercepts:

| Hook | Governance Event | Enforced? | Purpose |
|------|------------------|-----------|---------|
| `before_agent` | SignalReceived, WorkflowStarted | No (auto-allow) | Session setup, trace context init |
| `before_agent` | LLMStarted (pre-screen) | **Yes** | Pre-screen guardrails before agent loop |
| `wrap_model_call` | LLMStarted | PII only | Intercept all LLM calls, redact PII |
| `wrap_model_call` | LLMCompleted | No (observation) | Track model responses |
| `wrap_tool_call` | ToolStarted | **Yes** | Governance verdict on tool execution |
| `wrap_tool_call` | ToolCompleted | **Yes** | Post-execution behavior rules |
| `after_agent` | WorkflowCompleted | No (observation) | Cleanup, session close |

**Hook Implementation:**

Both sync and async versions:
```python
# Async (straightforward)
async def abefore_agent(self, config: RunnableConfig) -> None: ...

# Sync (bridges to async with OTel context preservation)
def before_agent(self, config: RunnableConfig) -> None: ...
```

### Lifecycle Flow

```
agent.invoke({"messages": [("user", "query")]})
  │
  ├─ BEFORE_AGENT HOOK
  │   ├─ Create session & trace context
  │   ├─ Send SignalReceived + WorkflowStarted events
  │   └─ Run pre-screen: LLMStarted → Governance API → verdict
  │       (If BLOCK or HALT, raise exception, stop execution)
  │       (Response cached to avoid duplicate API call)
  │
  ├─ MODEL CALLS (may occur multiple times)
  │   └─ WRAP_MODEL_CALL HOOK
  │       ├─ Build + redact LLMStarted event
  │       ├─ (PII redaction only, not governance verdict)
  │       ├─ Call model
  │       └─ Build LLMCompleted event (observation)
  │
  ├─ TOOL EXECUTION (per tool call)
  │   └─ WRAP_TOOL_CALL HOOK
  │       ├─ Build ToolStarted event
  │       ├─ Send to Governance API → verdict
  │       ├─ Enforce verdict (ALLOW/CONSTRAIN/BLOCK/HALT/REQUIRE_APPROVAL)
  │       │   └─ If REQUIRE_APPROVAL: HITL polling loop
  │       ├─ Register OTel span (for Layer 2 hook attribution)
  │       ├─ Execute tool
  │       └─ Build ToolCompleted event
  │
  └─ AFTER_AGENT HOOK
      ├─ Send WorkflowCompleted event
      ├─ Close session
      └─ Reset trace state
```

## Layer 2: Hook Governance

Intercepts I/O at kernel boundary (reused from `openbox-langgraph-sdk-python`):

| Protocol | Coverage | Examples |
|----------|----------|----------|
| HTTP | Request + response | httpx, requests, urllib3, urllib |
| Database | Query + result | SQLAlchemy, asyncpg, psycopg2, pymongo, redis |
| File I/O | Open + read/write | builtins.open(), os.fdopen() |

**Execution:** Hooks registered during middleware construction via `setup_opentelemetry_for_governance()`.

**Verdict enforcement:** Hooks raise `GovernanceBlockedError` or `GovernanceHaltError` on BLOCK/HALT verdicts.

## Layer 3: Activity Context Mapping

`WorkflowSpanProcessor` (imported from langgraph SDK) maps OTel metadata to governance context:

**Problem:** Hook fires during tool execution, but OpenBox Core doesn't know which tool.

**Solution:**
1. Middleware registers OTel span when tool starts
2. Hook reads OTel context (trace_id, span_id)
3. `WorkflowSpanProcessor` maps trace_id → (workflow_id, activity_id)
4. Hook sends activity_id to governance API
5. API attributes the hook to the correct tool

**Implementation:**

```python
# In middleware_tool_hook.py
async def ahandle_wrap_tool_call(self, tool_input, config, run_manager):
    # 1. Build ToolStarted event
    event = build_tool_started_event(...)

    # 2. Get verdict
    verdict = await self.client.get_verdict(event)

    # 3. Register OTel span for hook attribution
    with self.span_processor.register_span(
        trace_id=trace_id,
        activity_id=activity_id,
    ):
        # 4. Execute tool (any Layer 2 hooks fire in this context)
        return await tool(tool_input, **kwargs)
```

## Verdict System

5-tier severity hierarchy (lowest to highest priority):

| Verdict | Action | Example |
|---------|--------|---------|
| **ALLOW** | Permit execution | Tool passes all rules |
| **CONSTRAIN** | Apply limits | Rate-limit, truncate response |
| **REQUIRE_APPROVAL** | HITL polling | Requires human review |
| **BLOCK** | Raise GovernanceBlockedError | Request violates policy |
| **HALT** | Raise GovernanceHaltError | Unrecoverable security event |

**Enforcement points:**
- Tool start (ToolStarted event) — decide if tool can execute
- Tool end (ToolCompleted event) — check tool result against behavior rules
- LLM start (LLMStarted pre-screen only) — pre-check before agent loop

**HITL Polling (REQUIRE_APPROVAL):**
```python
# In enforce_verdict (from langgraph SDK)
if verdict == Verdict.REQUIRE_APPROVAL:
    approval = await poll_until_decision(
        workflow_id=workflow_id,
        activity_id=activity_id,
        poll_interval_ms=5000,  # Check every 5s
        timeout_ms=300000,      # Max 5 minutes
    )
    # Returns ApprovalResponse or raises ApprovalTimeoutError
```

## Pre-Screen Guardrails

**Purpose:** Evaluate user input before agent loop starts (avoids wasted token usage).

**Implementation:**
1. `before_agent` hook invoked → session created
2. Build minimal LLMStarted event with user input
3. Send to Governance API → get verdict
4. If BLOCK/HALT → raise exception, stop
5. Cache response to avoid duplicate API call
6. Agent loop proceeds

**Cache reuse:** If first model call matches pre-screen input, reuse verdict.

## PII Redaction

Applied to all LLM calls before sending to governance API:

**Detection:** Regex patterns + ML models (in langgraph SDK).

**Redaction:** Replace PII tokens with placeholders (e.g., `[EMAIL]`, `[PHONE]`).

**Implementation:**
```python
# In middleware_hooks.py
event = build_llm_started_event(...)
redacted = redact_pii_from_event(event)
response = await self.client.send_event(redacted)
```

## Configuration & Initialization

**Single entry point:**

```python
middleware = create_openbox_langchain_middleware(
    api_url="https://core.openbox.ai",
    api_key="obx_live_...",
    agent_name="MyAgent",
    governance_timeout=30.0,
    validate=True,  # Validates API key on startup
    sqlalchemy_engine=engine,  # Optional DB governance
    tool_type_map={"search_web": "http"},  # Optional tool classification
)
```

**Initialization sequence:**
1. `create_openbox_langchain_middleware()` called
2. `initialize()` called → validates API key, sets global config
3. `OpenBoxLangChainMiddlewareOptions` created with user options
4. `OpenBoxLangChainMiddleware` constructed:
   - GovernanceClient initialized (HTTP client)
   - WorkflowSpanProcessor configured (OTel mapping)
   - Layer 2 hooks registered via `setup_opentelemetry_for_governance()`
5. Returned to user, passed to `create_agent(middleware=[...])`

**Global config:** Stored in module-level variable, accessed via `get_global_config()`. Avoids passing config through call stack.

## Error Handling

Exception hierarchy (from langgraph SDK):

```
OpenBoxError (base)
  ├─ OpenBoxAuthError — API key invalid
  ├─ OpenBoxNetworkError — Network timeout, DNS error, etc.
  ├─ GovernanceBlockedError — BLOCK verdict
  ├─ GovernanceHaltError — HALT verdict
  ├─ ApprovalRejectedError — User rejected in HITL
  ├─ ApprovalTimeoutError — HITL approval timed out
  └─ GuardrailsValidationError — Validation failed
```

**Graceful degradation:** If governance API unreachable and timeout, log warning and allow request (fail-open).

## Data Serialization

Events sent to Governance API as JSON:

```python
# Example LLMStarted event
{
  "type": "LLMStarted",
  "payload": {
    "model": "gpt-4",
    "prompt": "What is 2+2?",
    "metadata": {
      "agent_name": "MyAgent",
      "user_id": "user-123"
    }
  },
  "timestamp": "2026-03-30T15:49:00Z"
}
```

**Serialization helpers:** `serialize_for_governance()`, `safe_serialize()` (handles circular refs, non-serializable objects).

## Performance Characteristics

- **Latency:** Governance API calls ~200-500ms (default timeout 30s)
- **Async I/O:** Non-blocking via httpx AsyncClient
- **Pre-screen caching:** Avoids duplicate API calls for first LLM invocation
- **Parallel tool execution:** OTel span registration thread-safe
- **Memory:** ~1MB per active session (trace context + span buffer)

## References

- **Project Overview:** `./docs/project-overview-pdr.md`
- **Code Standards:** `./docs/code-standards.md`
- **Codebase Summary:** `./docs/codebase-summary.md`
- **LangChain Middleware:** https://github.com/langchain-ai/langgraph (AgentMiddleware base class)
- **LangGraph SDK:** `/Users/tino/code/openbox-langgraph-sdk-python/` (governance infrastructure)

