# OpenBox LangChain SDK — Python

Governance and observability SDK for LangChain agents. Provides real-time policy enforcement, guardrails, HITL approval flows, and hook-level governance via `AsyncCallbackHandler`.

## Installation

```bash
pip install openbox-langchain-sdk-python
```

## Quick Start

```python
from openbox_langchain import create_openbox_langchain_handler

handler = create_openbox_langchain_handler(
    api_url="https://core.openbox.ai",
    api_key="obx_live_...",
    agent_name="MyAgent",
    tool_type_map={"search_web": "http", "query_db": "database"},
)

# Attach to any LangChain agent
result = agent.invoke(
    {"input": "Hello"},
    config={"callbacks": [handler]},
)
```

## Pre-Screen (Recommended)

For reliable blocking before the agent starts, use `pre_screen()`:

```python
handler = create_openbox_langchain_handler(api_url="...", api_key="...")

# Pre-screen raises GovernanceBlockedError/GovernanceHaltError if blocked
await handler.pre_screen({"input": "user message"})

# If pre-screen passes, run the agent
result = agent.invoke({"input": "user message"}, config={"callbacks": [handler]})
```

## Architecture

Three-layer governance (same as LangGraph SDK):

| Layer | Mechanism | What It Governs |
|-------|-----------|-----------------|
| 1 | AsyncCallbackHandler | Chain/Tool/LLM lifecycle events |
| 2 | Hook Governance | HTTP requests, DB queries, file I/O |
| 3 | Activity Context Mapping | Links hook traces to governance activities |

## Configuration

```python
handler = create_openbox_langchain_handler(
    api_url="https://core.openbox.ai",
    api_key="obx_live_...",
    agent_name="MyAgent",
    governance_timeout=30.0,           # API timeout in seconds
    validate=True,                     # Validate API key on startup
    tool_type_map={                    # Classify tools for execution tree
        "search_web": "http",
        "query_db": "database",
        "write_file": "builtin",
    },
    hitl={"enabled": True, "poll_interval_ms": 5000},
    session_id="session-123",          # Optional session tracking
    sqlalchemy_engine=engine,          # Optional DB governance
)
```

## Supported Agent Types

- `create_tool_calling_agent` + `AgentExecutor` (primary)
- `create_react_agent` + `AgentExecutor`
- `create_structured_chat_agent`
- Raw `RunnableSequence` / LCEL chains

## Requirements

- Python 3.11+
- `langchain-core >= 0.3.0`

## License

MIT
