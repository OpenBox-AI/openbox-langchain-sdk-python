# OpenBox LangChain SDK — Python

LangChain-Core adapter for OpenBox governance. Provides callback handlers that emit tool and LLM lifecycle events to OpenBox Core's policy engine, enabling real-time governance, guardrails, HITL approval flows, and base-SDK hook governance (HTTP/DB/File I/O) for LangChain agents.

## Installation

```bash
pip install openbox-langchain-sdk-python
```

## Quick Start

```python
from langchain.agents import create_agent
from openbox_langchain import create_openbox_langchain_middleware

# 1. Create middleware
middleware = create_openbox_langchain_middleware(
    api_url="https://core.openbox.ai",
    api_key="obx_live_...",
    agent_name="MyAgent",
)

# 2. Create agent with middleware
agent = create_agent(
    model="openai:gpt-4o",
    tools=[...],
    middleware=[middleware],
)

# 3. Invoke — governance applied automatically
result = agent.invoke({"messages": [("user", "your query")]})
```

## How It Works

Two-layer governance architecture:

| Layer | Mechanism | Governs |
|-------|-----------|---------|
| 1 | LangChain-Core Callbacks | Tool and LLM lifecycle emission via `ActivityBridge` |
| 2 | Base SDK Hook Governance | HTTP requests, DB queries, file I/O at process boundary |

**Core components:**
- `ActivityBridge` — Ownership channel for tool and LLM events; prevents duplicate governance evaluation
- `OpenBoxLangChainCoreAsyncCallbackHandler` / `...SyncCallbackHandler` — LangChain-Core callbacks emitting tool and LLM lifecycle events to OpenBox Core
- AgentMiddleware (for `create_agent`) — Wraps model and tool calls with additional governance context

## Configuration

```python
middleware = create_openbox_langchain_middleware(
    api_url="https://core.openbox.ai",  # OpenBox Core URL
    api_key="obx_live_...",              # API key (obx_live_* or obx_test_*)
    agent_name="MyAgent",                # Agent name (from dashboard)
    governance_timeout=30.0,             # HTTP timeout in seconds
    validate=True,                       # Validate API key on startup
    session_id="session-123",            # Optional session tracking
    tool_type_map={                      # Optional tool classification
        "search_web": "http",
        "query_db": "database",
    },
)
```

## Supported Agent Types

- `create_agent(model, tools, middleware=[...])` — recommended
- Any LangChain agent builder that accepts `middleware`

## Verdict Enforcement

5-tier verdict system:
- **ALLOW** — Request permitted
- **CONSTRAIN** — Request constrained (e.g., rate limit)
- **REQUIRE_APPROVAL** — Human approval required (HITL polling)
- **BLOCK** — Request blocked with error
- **HALT** — Entire workflow halted (unrecoverable error)

## Requirements

- Python 3.11+
- openbox-sdk-python >= 0.2.0
- langchain-core >= 1.3.3
- LangChain >= 1.0.0 (required only for `[agent]` extra, which enables `create_agent` middleware)

## API Reference

**LangChain-Core callbacks:**
- `OpenBoxLangChainCoreAsyncCallbackHandler` — Async callback handler for tool/LLM lifecycle
- `OpenBoxLangChainCoreSyncCallbackHandler` — Sync callback handler for tool/LLM lifecycle
- `ActivityBridge` — Ownership channel for lifecycle event deduplication

**AgentMiddleware (optional):**
- `create_openbox_langchain_middleware()` — Creates configured middleware for `create_agent`

See `openbox_langchain.__init__.py` for full API export list.

## License

MIT
