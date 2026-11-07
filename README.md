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
    agent_did="did:aip:...",             # Required by default; can use OPENBOX_AGENT_DID
    agent_private_key="...",             # Required by default; can use OPENBOX_AGENT_PRIVATE_KEY
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

## Exposure Barriers

Whole-run state needs nothing from this package: run the agent inside the base
SDK's `exposure_scope(...)` and every event carries the scope, purpose and
instance. What a single tool call touches has to be declared here, because the
tool's start event is checked before the tool body runs:

```python
import json
from openbox_core import Participant, exposure_scope, relay_via_executor
from openbox_langchain import ToolExposure, current_verdict_token, exposure_tool

def search_docs(q: str) -> str:
    body = json.dumps({"q": q}, sort_keys=True)
    # The Executor checks the token, narrows the query and runs the admission
    # check on what comes back before returning it.
    resp = relay_via_executor(EXECUTOR_URL, "dms:main", body, [DMS], current_verdict_token())
    return render(resp.json())

search = exposure_tool(
    StructuredTool.from_function(search_docs),
    ToolExposure(
        action_class="retrieve",
        participants=[DMS],
        payload=lambda args: json.dumps(args, sort_keys=True),
    ),
)

with exposure_scope("matter:coke-2026", purpose="pricing_review"):
    agent.invoke(...)
```

- **Before the call**, the action class, resources, participants and payload
  hash go on the tool's start event. Tools can also be declared by name with
  `tool_exposure={"search_docs": ToolExposure(...)}` on the middleware or
  callback options.
- **Inside the call**, `current_verdict_token()` returns the token Core issued
  for that exact payload, for `relay_via_executor`.
- **After the call**, a tool that reaches a source directly (not through the
  Executor, which runs this check itself) reports what it returned for Core's
  admission check. It can return it as a `content_and_artifact` list of `Resource`s,
  through `ToolExposure(returned=...)`, or with `report_returned(...)`. With
  the middleware, a refused admission replaces the result with
  `Refused by an exposure barrier`, so nothing refused reaches the model, and a
  HALT stops the run. The refusal never names labels or rules. With the
  callback handlers, the verdict is stored on the `ActivityBridge` record, as
  for every completion.

## Requirements

- Python 3.11+
- openbox-sdk-python >= 1.5.0
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
