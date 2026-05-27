# OpenBox LangChain SDK — Python

Governance and observability SDK for LangChain agents. Intercepts agent execution via `AgentMiddleware` to enforce OpenBox policies, guardrails, HITL approval flows, and hook-level governance (HTTP/DB/File I/O).

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
    agent_did="did:aip:...",
    agent_private_key="...",
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

Three-layer governance architecture:

| Layer | Mechanism | Governs |
|-------|-----------|---------|
| 1 | AgentMiddleware hooks | Agent lifecycle (before/after), model calls, tool execution |
| 2 | Hook Governance | HTTP requests, DB queries, file I/O at kernel boundary |
| 3 | Activity Context Mapping | Links hook traces to governance activities via OTel |

**Middleware hooks:**
- `before_agent` / `abefore_agent` — Session setup, pre-screen guardrails
- `wrap_model_call` / `awrap_model_call` — LLM interception, PII redaction
- `wrap_tool_call` / `awrap_tool_call` — Tool governance, OTel span registration
- `after_agent` / `aafter_agent` — Session cleanup

## Configuration

```python
middleware = create_openbox_langchain_middleware(
    api_url="https://core.openbox.ai",  # OpenBox Core URL
    api_key="obx_live_...",              # API key (obx_live_* or obx_test_*)
    agent_did="did:aip:...",             # Required by default; can use OPENBOX_AGENT_DID
    agent_private_key="...",             # Required by default; can use OPENBOX_AGENT_PRIVATE_KEY
    agent_name="MyAgent",                # Agent name (from dashboard)
    governance_timeout=30.0,             # HTTP timeout in seconds
    validate=True,                       # Validate API key on startup
    session_id="session-123",            # Optional session tracking
    sqlalchemy_engine=engine,            # Optional DB governance
    tool_type_map={                      # Optional tool classification
        "search_web": "http",
        "query_db": "database",
    },
)
```

## Agent Identity and DID Signing

OpenBox issues each registered agent a decentralized identifier (DID) and
private key. DID signing is enabled by default for newly registered agents. The
OpenBox UI returns both values when the agent is created. Pass them to the
middleware so governance events are signed and attributable to that agent.

You can provide them directly:

```python
middleware = create_openbox_langchain_middleware(
    api_url="https://core.openbox.ai",
    api_key="obx_live_...",
    agent_did="did:aip:...",
    agent_private_key="...",
)
```

Or set them through the environment:

```bash
export OPENBOX_AGENT_DID="did:aip:..."
export OPENBOX_AGENT_PRIVATE_KEY="..."
```

If DID signing is explicitly disabled for the agent in OpenBox, these values can
be omitted. Otherwise, provide both values together.

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
- LangChain >= 0.3.0
- LangGraph >= 0.2.0
- openbox-langgraph-sdk-python >= 0.2.0

## API Reference

**Primary factory:**
- `create_openbox_langchain_middleware()` — Creates configured middleware

**Re-exported from langgraph SDK:**
- `enforce_verdict()` — Enforce verdicts
- `poll_until_decision()` — HITL approval polling
- `GovernanceClient`, `GovernanceConfig` — Core types

See `openbox_langchain.__init__.py` for full API export list.

## License

MIT
