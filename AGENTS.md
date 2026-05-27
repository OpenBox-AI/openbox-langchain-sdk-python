# AGENTS.md — OpenBox LangChain SDK

## Project Overview

OpenBox governance SDK for LangChain agents. Intercepts agent execution via `AgentMiddleware` and enforces OpenBox policies, guardrails, HITL approvals, and behavior rules.

## Architecture

Three-layer governance:

- **Layer 1:** `OpenBoxLangChainMiddleware` (AgentMiddleware subclass) — Intercepts agent lifecycle (before/after agent, wrap model/tool calls)
- **Layer 2:** Hook governance (HTTP/DB/File I/O) — Imported from `openbox-langgraph-sdk-python`
- **Layer 3:** Activity context mapping via `WorkflowSpanProcessor` — Imported from `openbox-langgraph-sdk-python`

**Key principle:** Only the middleware integration layer is new code. All governance infrastructure is imported from `openbox-langgraph-sdk-python` (not copied).

## Package Structure

```
openbox_langchain/
├── __init__.py                     # Public API (re-exports + exports middleware)
├── middleware.py                   # OpenBoxLangChainMiddleware + options
├── middleware_factory.py           # create_openbox_langchain_middleware() factory
├── middleware_hooks.py             # Event builders, PII redaction, OTel helpers
├── middleware_hook_handlers.py     # before_agent, after_agent, wrap_model_call handlers
└── middleware_tool_hook.py         # wrap_tool_call handler
```

**Code count:** 1,025 total lines (all new), focused on middleware integration only.

## Key Classes

- `OpenBoxLangChainMiddleware` — Main middleware (AgentMiddleware subclass)
- `OpenBoxLangChainMiddlewareOptions` — Configuration dataclass

## Key Functions

- `create_openbox_langchain_middleware()` — Factory (primary entry point)
- `handle_before_agent()` / `ahandle_before_agent()` — Session setup, pre-screen
- `handle_after_agent()` / `ahandle_after_agent()` — Cleanup
- `handle_wrap_model_call()` / `ahandle_wrap_model_call()` — LLM interception, PII redaction
- `handle_wrap_tool_call()` / `ahandle_wrap_tool_call()` — Tool governance, OTel spans

## Quick Start

```python
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from openbox_langchain import create_openbox_langchain_middleware

# Create middleware
middleware = create_openbox_langchain_middleware(
    api_url="https://core.openbox.ai",
    api_key="obx_live_...",
    agent_name="MyAgent",
)

# Create agent with middleware
model = ChatOpenAI(model="gpt-4")
tools = [...]

agent = create_react_agent(
    model=model,
    tools=tools,
    middleware=[middleware],
)

# Invoke — governance applied automatically
result = agent.invoke({"messages": [("user", "your query")]})
```

## Commands

```bash
# Install
pip install -e ".[dev]"

# Test
pytest tests/ -v

# Lint
ruff check openbox_langchain/

# Type check
mypy openbox_langchain/
```

## Testing

**Current:** 116 tests, 100% pass rate, 89% package coverage

**Coverage:**
- Unit tests for each hook handler
- Integration tests for full agent workflow
- Mock client tests for verdict enforcement
- Pre-screen caching, OTel context, async/sync bridge
- Error handling (network, timeout, invalid verdicts)

## Documentation

- **README.md** — Quick start + configuration
- **docs/project-overview-pdr.md** — Functional/non-functional requirements
- **docs/system-architecture.md** — 3-layer design, data flow, verdict system
- **docs/code-standards.md** — Patterns, conventions, guidelines
- **docs/codebase-summary.md** — Module organization, dependencies
- **docs/project-roadmap.md** — Timeline, milestones, future work

## References

- **DeepAgent SDK:** `/Users/tino/code/openbox-deepagent-sdk-python/` (reference middleware pattern)
- **LangGraph SDK:** `/Users/tino/code/openbox-langgraph-sdk-python/` (governance infrastructure)
- **SDK Guide:** `/Users/tino/code/sdk-implementation-guide/README.md`
