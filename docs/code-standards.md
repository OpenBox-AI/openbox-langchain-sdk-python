# Code Standards — OpenBox LangChain SDK

## Architecture & Design Patterns

### Middleware Pattern (AgentMiddleware)

All agent interception via LangChain v1.0+ `AgentMiddleware` subclass:

```python
from langgraph.types import AgentMiddleware

class OpenBoxLangChainMiddleware(AgentMiddleware):
    def __init__(self, options: OpenBoxLangChainMiddlewareOptions) -> None: ...

    async def abefore_agent(self, state: RunnableConfig) -> None: ...
    def before_agent(self, state: RunnableConfig) -> None: ...

    async def aafter_agent(self, state: RunnableConfig) -> None: ...
    def after_agent(self, state: RunnableConfig) -> None: ...

    async def awrap_model_call(...) -> LLMOutput: ...
    def wrap_model_call(...) -> LLMOutput: ...

    async def awrap_tool_call(...) -> ToolOutput: ...
    def wrap_tool_call(...) -> ToolOutput: ...
```

**Why middleware over callbacks:**
- Callbacks are deprecated in LangChain v1.0+
- Middleware is the official interception pattern
- Cleaner integration with `create_agent(..., middleware=[...])`
- Proper type hints and lifecycle guarantees

### Async/Sync Bridge Pattern

Every hook has both async and sync versions. Sync hooks run async code via bridge:

```python
async def awrap_model_call(self, model_call: ...) -> LLMOutput:
    # Async version (straightforward)
    return await run_governance(...)

def wrap_model_call(self, model_call: ...) -> LLMOutput:
    # Sync version — bridge to async with OTel context propagation
    return _run_async(
        awrap_model_call(self, model_call),
        preserve_context=True,  # Maintain OTel context across thread
    )
```

**Implementation detail:** `_run_async()` helper in middleware creates event loop or uses running loop, properly propagates OTel context via contextvars.

### Factory Pattern

Single entry point for middleware creation:

```python
def create_openbox_langchain_middleware(
    *,
    api_url: str,
    api_key: str,
    agent_name: str | None = None,
    **kwargs,
) -> OpenBoxLangChainMiddleware:
    """Create configured middleware.

    1. Validate API key (if validate=True, default)
    2. Initialize global config via initialize()
    3. Return configured OpenBoxLangChainMiddleware
    """
    initialize(api_url=api_url, api_key=api_key, ...)
    options = OpenBoxLangChainMiddlewareOptions(...)
    return OpenBoxLangChainMiddleware(options)
```

**Purpose:**
- Centralize configuration logic
- Ensure global state initialized before middleware created
- Validate API key on startup (fail fast)
- Accept both required (api_url, api_key) and optional args (agent_name, sqlalchemy_engine)

### Deferred Imports

Imports from `openbox_langgraph` are deferred to avoid circular dependencies:

```python
# In middleware_factory.py
def create_openbox_langchain_middleware(...) -> OpenBoxLangChainMiddleware:
    from openbox_langgraph.config import initialize  # Deferred
    initialize(...)  # Ensures langgraph SDK initialized before middleware created
```

**Why:** Avoids import-time side effects, ensures correct initialization order.

## File Organization

### `middleware.py` (239 lines)

Defines two classes:

**`OpenBoxLangChainMiddlewareOptions`** (dataclass):
- Stores configuration options (agent_name, session_id, governance_timeout, sqlalchemy_engine, tool_type_map)
- Passed to middleware constructor
- No business logic, purely data structure

**`OpenBoxLangChainMiddleware`** (AgentMiddleware subclass):
- Implements all 6 hook methods (before/after agent, wrap model/tool calls)
- Stores global config, client, span processor
- Delegates to handler functions (in separate modules)

### `middleware_factory.py` (67 lines)

Single function: `create_openbox_langchain_middleware()`
- Primary entry point for users
- Validates API key, initializes global config
- Returns configured middleware

### `middleware_hooks.py` (220 lines)

Reusable helpers used by all handlers:

**Event builders:**
```python
def build_workflow_started_event(...) -> LangChainGovernanceEvent: ...
def build_signal_received_event(...) -> LangChainGovernanceEvent: ...
def build_llm_started_event(...) -> LangChainGovernanceEvent: ...
```

**PII redaction:**
```python
def redact_pii_from_event(event: LangChainGovernanceEvent) -> LangChainGovernanceEvent: ...
```

**OTel span helpers:**
```python
def create_span_from_tool_start(...) -> Span: ...
def set_span_attributes(...): ...
```

**Serialization & utilities:**
```python
def serialize_for_governance(obj: Any) -> dict: ...
def format_tool_result(result: Any) -> str: ...
```

### `middleware_hook_handlers.py` (231 lines)

Handlers for lifecycle hooks:

**`handle_before_agent()` / `ahandle_before_agent()`:**
- Create session + trace context
- Send SignalReceived + WorkflowStarted events
- Run pre-screen guardrails (first LLM call)
- Store response to avoid duplicate API call

**`handle_after_agent()` / `ahandle_after_agent()`:**
- Send WorkflowCompleted event
- Close session
- Cleanup trace state

**`handle_wrap_model_call()` / `ahandle_wrap_model_call()`:**
- Wrap model invocation
- Build + redact LLMStarted event
- Send to governance API
- Call model
- Track response for pre-screen reuse

### `middleware_tool_hook.py` (144 lines)

Handler for tool execution:

**`handle_wrap_tool_call()` / `ahandle_wrap_tool_call()`:**
- Build ToolStarted event
- Send to governance API, get verdict
- Enforce verdict (ALLOW/CONSTRAIN/BLOCK/HALT)
- Register OTel span for hook attribution
- Execute tool
- Build + send ToolCompleted event

## Coding Conventions

### Type Hints (Required)

All functions must have complete type hints:

```python
async def ahandle_before_agent(
    self,
    config: RunnableConfig,
) -> None:
    """Handle agent startup."""
    # ✓ Good: clear input/output types
```

**Tools used:** `mypy --strict` on all files.

### Docstrings (Minimal)

Use minimal docstrings (one-liner + args/returns if needed):

```python
def enforce_verdict(verdict: Verdict) -> None:
    """Raise exception if verdict requires it.

    Args:
        verdict: Governance decision from OpenBox Core

    Raises:
        GovernanceBlockedError: If verdict is BLOCK
        GovernanceHaltError: If verdict is HALT
    """
```

**Style:** Google-style docstrings, focus on **why** not **what**.

### Error Handling

Use exception hierarchy from `openbox_langgraph`:

```python
from openbox_langgraph import (
    GovernanceBlockedError,
    GovernanceHaltError,
    ApprovalRejectedError,
    ApprovalTimeoutError,
)

try:
    response = await governance_client.get_verdict(event)
except OpenBoxNetworkError as e:
    logger.warning("Governance API unreachable, allowing request", exc_info=e)
    # Graceful degradation: continue if API down
except GovernanceBlockedError:
    raise  # Re-raise blocking verdicts
```

### Logging

Use DEBUG level for governance decision points:

```python
import logging

logger = logging.getLogger(__name__)

logger.debug("Pre-screen verdict: %s", verdict)
logger.debug("Tool governance: tool=%s verdict=%s", tool_name, verdict)
```

**Convention:** DEBUG for decisions, INFO for errors/warnings.

### Testing Strategy

**Test organization:**
- Unit tests for each handler function
- Integration tests for full agent workflow
- Fixtures for mock client, config, serialized events

**Coverage:** Aim for 100% (currently 99 tests, 100% pass).

**Patterns:**
```python
@pytest.mark.asyncio
async def test_ahandle_before_agent_pre_screen():
    """Verify pre-screen guardrails executed in before_agent."""
    config = RunnableConfig(...)
    handler = create_middleware(...)

    # Mock client to return BLOCK verdict
    handler.client.get_verdict = mock_block_verdict

    # Expect GovernanceBlockedError
    with pytest.raises(GovernanceBlockedError):
        await handler.ahandle_before_agent(config)
```

## Module Dependencies

**Dependency graph (acyclic):**

```
middleware_factory.py
  └─ middleware.py
       ├─ middleware_hooks.py
       ├─ middleware_hook_handlers.py
       │   └─ middleware_hooks.py
       └─ middleware_tool_hook.py
            └─ middleware_hooks.py
```

**Rule:** Avoid circular imports. Deferred imports OK for initialization-time dependencies.

## Code Quality Standards

### Linting & Formatting

**Tool:** ruff (configured in `pyproject.toml`)

```bash
ruff check openbox_langchain/
```

**Rules enforced:**
- E (PEP 8 errors)
- F (undefined names)
- I (import sorting)
- UP (Python syntax upgrade)
- B (bugbear)
- C4 (comprehensions)
- PIE (pie)
- RUF (ruff-specific)

### Type Checking

**Tool:** mypy (strict mode)

```bash
mypy openbox_langchain/
```

**Rule:** All functions must pass strict type checking (no `Any` without justification).

### Line Length

**Limit:** 100 characters (per ruff config).

**Exceptions:** Long strings, URLs, type hints that naturally exceed 100 chars are OK.

### File Size

**Limit:** < 240 lines per file.

**Rationale:** Optimal for context window, encourages modularization.

## Async/Await Patterns

### Rule 1: Both Sync & Async Implementations

Every hook has sync and async versions:

```python
async def ahandle_before_agent(self, config: RunnableConfig) -> None: ...
def handle_before_agent(self, config: RunnableConfig) -> None: ...
```

### Rule 2: OTel Context Propagation

Sync hooks must preserve OTel context across async boundary:

```python
def wrap_model_call(self, model_call: ...) -> LLMOutput:
    # Use _run_async to properly propagate OTel context
    return _run_async(self.awrap_model_call(model_call))
```

### Rule 3: No Blocking I/O in Async Code

Async functions must not use blocking I/O:

```python
# ✗ Wrong: blocking
async def ahandle_before_agent(self, config: RunnableConfig) -> None:
    response = requests.post(url)  # BLOCKING!

# ✓ Correct: async
async def ahandle_before_agent(self, config: RunnableConfig) -> None:
    response = await self.client.post(url)  # httpx async client
```

## Event Building & Serialization

### Pattern: Immutable Events

Events are immutable dataclasses (from `openbox_langgraph`):

```python
@dataclass(frozen=True)
class LangChainGovernanceEvent:
    type: WorkflowEventType
    payload: dict[str, Any]
    timestamp: str
```

### Building Events

Use helper functions in `middleware_hooks.py`:

```python
# ✓ Good: Use builder
event = build_llm_started_event(
    model=model_name,
    prompt=prompt_text,
    metadata={"agent_name": self.options.agent_name},
)
```

### PII Redaction

Apply after building, before sending:

```python
event = build_llm_started_event(...)
redacted_event = redact_pii_from_event(event)
response = await self.client.send_event(redacted_event)
```

## Configuration Management

### Pattern: Global Config via openbox_langgraph

Configuration initialized once via `initialize()`:

```python
from openbox_langgraph import initialize

# In factory
initialize(api_url=..., api_key=..., governance_timeout=...)

# Then in middleware, retrieve global config
from openbox_langgraph import get_global_config
config = get_global_config()
```

**Why:** Matches langgraph SDK pattern, avoids passing config everywhere.

### Pattern: Options Dataclass

Middleware-specific options stored in dataclass:

```python
@dataclass
class OpenBoxLangChainMiddlewareOptions:
    agent_name: str | None = None
    session_id: str | None = None
    governance_timeout: float = 30.0
    sqlalchemy_engine: Any = None
    tool_type_map: dict[str, str] | None = None
```

## Performance Considerations

### Async I/O (Non-Blocking)

All HTTP calls use httpx async client:

```python
# In GovernanceClient (from langgraph SDK)
async def get_verdict(self, event: ...) -> GovernanceVerdictResponse:
    async with httpx.AsyncClient(...) as client:
        response = await client.post(url, json=...)
        return parse_governance_response(response.json())
```

### Parallel Execution

Multiple tools can execute concurrently. OTel span registration must be thread-safe:

```python
# In middleware_tool_hook.py
async def ahandle_wrap_tool_call(self, ...):
    # Register span before tool execution (OTel context holder)
    with self.span_processor.register_span(trace_id, activity_id):
        # Tool executes here, any hooks fire in correct OTel context
        return await tool(...)
```

### Caching (Pre-Screen)

Response from first LLM call cached to avoid duplicate governance check:

```python
# In handle_before_agent
if self._prescreen_response_cached:
    # Reuse response, don't call governance API again
    return self._prescreen_response_cached
```

## References

- **System Architecture:** `./docs/system-architecture.md` — Data flow, layer descriptions
- **Codebase Summary:** `./docs/codebase-summary.md` — Module organization, class/function lists
- **LangGraph Middleware:** `https://github.com/langchain-ai/langgraph` — AgentMiddleware base class
- **OpenBox LangGraph SDK:** `/Users/tino/code/openbox-langgraph-sdk-python/` — Governance infrastructure
