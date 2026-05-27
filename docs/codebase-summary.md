# Codebase Summary — OpenBox LangChain SDK

## Package Structure

```
openbox_langchain/
├── __init__.py                     # Public API — re-exports from langgraph SDK + middleware
├── middleware.py                   # OpenBoxLangChainMiddleware (AgentMiddleware subclass)
├── middleware_factory.py           # create_openbox_langchain_middleware() factory
├── middleware_hooks.py             # Shared helpers: event building, PII redaction, OTel
├── middleware_hook_handlers.py     # before_agent, after_agent, wrap_model_call handlers
└── middleware_tool_hook.py         # wrap_tool_call handler
```

## Code Statistics

| Module | Purpose | Lines | Type |
|--------|---------|-------|------|
| middleware.py | Middleware class + options | 239 | NEW |
| middleware_factory.py | Factory function | 67 | NEW |
| middleware_hooks.py | Event builders, PII, OTel helpers | 220 | NEW |
| middleware_hook_handlers.py | Lifecycle hooks (before/after agent, model) | 231 | NEW |
| middleware_tool_hook.py | Tool execution hook | 144 | NEW |
| __init__.py | Public API + re-exports | 124 | NEW |
| **Total (LangChain SDK)** | | **1,025** | NEW |

**Governance Infrastructure:** All imported from `openbox-langgraph-sdk-python` (not copied):
- `GovernanceClient` — HTTP client to OpenBox Core
- `GovernanceConfig` — Merged config initialization
- `WorkflowSpanProcessor` — OTel trace→activity mapper
- `enforce_verdict()`, `poll_until_decision()` — Verdict enforcement & HITL
- Exception hierarchy: `GovernanceBlockedError`, `GovernanceHaltError`, etc.
- Event types: `LangChainGovernanceEvent`, `WorkflowEventType`

## Key Classes

**LangChain SDK:**
- `OpenBoxLangChainMiddleware` (239 lines) — Main middleware, `AgentMiddleware` subclass with sync+async hooks
- `OpenBoxLangChainMiddlewareOptions` — Configuration dataclass (agent_name, session_id, governance_timeout, sqlalchemy_engine, etc.)

**Re-exported from langgraph SDK:**
- `GovernanceClient` — HTTP client for OpenBox Core API
- `GovernanceConfig` — Merged governance configuration
- `WorkflowSpanProcessor` — OTel span→activity mapper
- `Verdict` — Enum (ALLOW, CONSTRAIN, REQUIRE_APPROVAL, BLOCK, HALT)

## Key Functions

**LangChain SDK:**
- `create_openbox_langchain_middleware()` — Factory, primary entry point (validates API key, initializes global config, returns configured middleware)
- `handle_before_agent()` — Session setup, pre-screen guardrails, SignalReceived + WorkflowStarted events
- `handle_after_agent()` — Session cleanup, WorkflowCompleted event
- `handle_wrap_model_call()` — LLM interception, PII redaction, LLMStarted/LLMCompleted events
- `handle_wrap_tool_call()` — Tool governance, OTel span registration, ToolStarted/ToolCompleted events

**Re-exported from langgraph SDK:**
- `enforce_verdict()` — Raises on block/halt verdicts
- `poll_until_decision()` — HITL approval polling loop with retry
- `setup_opentelemetry_for_governance()` — Initializes OTel hooks
- `parse_governance_response()`, `safe_serialize()`, `traced()` — Utilities

## Test Structure

```
tests/
├── conftest.py                     # Shared fixtures (mock client, config, serialized data)
└── __init__.py

examples/
└── content-builder-agent/          # Full working example agent with middleware
```

**Test Status:** 116 tests, 100% pass rate, 89% package coverage.

## Module Dependencies

```
middleware.py
  ├─ imports: middleware_factory, middleware_hooks, middleware_hook_handlers, middleware_tool_hook
  ├─ from openbox_langgraph: GovernanceClient, GovernanceConfig, Verdict, etc.
  └─ from langgraph: AgentMiddleware, RunnableConfig

middleware_factory.py
  └─ imports: middleware.py, initialize() from openbox_langgraph

middleware_hooks.py
  ├─ Event builders: build_workflow_started, build_signal_received, build_llm_started, etc.
  ├─ PII redaction helpers
  └─ OTel span helpers

middleware_hook_handlers.py
  ├─ before_agent() / abefore_agent()
  ├─ after_agent() / aafter_agent()
  ├─ wrap_model_call() / awrap_model_call()
  └─ Imports from middleware_hooks.py

middleware_tool_hook.py
  ├─ wrap_tool_call() / awrap_tool_call()
  └─ Imports from middleware_hooks.py

__init__.py
  ├─ Re-exports entire langgraph SDK public surface
  ├─ Re-exports: OpenBoxLangChainMiddleware, create_openbox_langchain_middleware
  └─ Exports __version__ via importlib.metadata
```

## File Organization Rationale

**Split by responsibility:**
- `middleware.py` — Class definition + options dataclass (single responsibility: define the middleware)
- `middleware_factory.py` — Factory pattern (single responsibility: construct + configure middleware)
- `middleware_hooks.py` — Reusable helpers (single responsibility: helper functions for all hooks)
- `middleware_hook_handlers.py` — before/after agent + wrap_model_call (grouped by lifecycle stage)
- `middleware_tool_hook.py` — wrap_tool_call (tool governance is complex enough to warrant separate module)

**Rationale:** Clear separation of concerns, easy to locate code by responsibility. No file exceeds 240 lines (LangChain SDK best practice).
