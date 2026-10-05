# OpenBox LangChain SDK — Python

LangChain-Core adapter for OpenBox governance. Provides callback handlers that emit tool and LLM lifecycle events to OpenBox Core's policy engine, enabling real-time governance, guardrails, HITL approval flows, and base-SDK hook governance (HTTP/DB/File I/O) for LangChain agents.

## Installation

```bash
pip install "openbox-langchain-sdk-python[agent]"
```

## Quick Start

```python
from langchain.agents import create_agent
from openbox_langchain import create_openbox_langchain_middleware

# 1. Create middleware
middleware = create_openbox_langchain_middleware(
    api_url="https://core.openbox.ai",
    api_key="obx_live_...",
    workload_private_key="...",  # PKCS8 PEM RSA key for this agent's workload identity
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
    workload_private_key="...",         # IAM v3 key; can use OPENBOX_WORKLOAD_PRIVATE_KEY
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

## IAM v3 Workload Identity

Supply the PKCS8 PEM RSA private key associated with the agent's active Keycloak
service account. This works for OpenBox-, Okta-, and Entra-managed workload
identities. The base SDK fetches the active identity metadata from Core, exchanges
a signed `private_key_jwt` for a short-lived Keycloak token, and sends the API key
plus `X-OpenBox-Workload-Token` on v3 validation, governance, and approval requests.

You can pass the key directly or use environment variables:

```python
import os

middleware = create_openbox_langchain_middleware(
    api_url=os.environ["OPENBOX_URL"],
    api_key=os.environ["OPENBOX_API_KEY"],
    workload_private_key=os.environ["OPENBOX_WORKLOAD_PRIVATE_KEY"],
)
```

If the argument is omitted, the key is resolved in this order:

1. `OPENBOX_LANGCHAIN_WORKLOAD_PRIVATE_KEY`
2. `OPENBOX_WORKLOAD_PRIVATE_KEY`

The variable contains the PEM key text. The base SDK also supports existing Okta
agents' `OPENBOX_OKTA_AGENT_PRIVATE_KEY` as a workload-key fallback. Core supplies
the issuer, token endpoint, client ID, and active authority metadata.

Startup validation is enabled by default. `validate=False` defers authentication
until the first governed request. Governance and human-approval polling share one
client and token cache. Use `middleware.close()` after synchronous runs or
`await middleware.aclose()` after asynchronous runs to release resources.

Authentication failures propagate even with `on_api_error="fail_open"`. The base
SDK preserves legacy routing only when Core explicitly reports no v3 authority
(bootstrap HTTP 404, or HTTP 409 with `workload_identity_unavailable`); other
bootstrap or token failures do not fall back to a legacy request.

### Legacy identities

Existing DID-signed agents can continue supplying `agent_did` and
`agent_private_key`, or `OPENBOX_AGENT_DID` and `OPENBOX_AGENT_PRIVATE_KEY`.
API-key-only agents can omit identity keys when their Core configuration permits
it. An Ed25519 DID key and an RSA workload key are different credentials.

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
- openbox-sdk-python >= 1.3.1
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
