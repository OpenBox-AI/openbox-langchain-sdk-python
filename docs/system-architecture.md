# System Architecture — OpenBox LangChain SDK

## Overview

Three-layer governance SDK for LangChain agents, mirroring the LangGraph SDK architecture but using `AsyncCallbackHandler` for interception instead of event stream wrapping.

## Layer 1: Callback Governance

`OpenBoxGovernanceCallbackHandler` extends `AsyncCallbackHandler` and intercepts:

| Callback | Governance Event | Enforced? |
|----------|-----------------|-----------|
| on_chain_start (root) | WorkflowStarted | No (auto-allow) |
| on_chain_start (nested) | ChainStarted | Yes |
| on_chain_end (root) | WorkflowCompleted | No (observation) |
| on_tool_start | ToolStarted | Yes |
| on_tool_end | ToolCompleted | Yes (behavior rules) |
| on_chat_model_start | LLMStarted | PII redaction only |
| on_llm_end | LLMCompleted | No (observation) |
| on_agent_action | ToolStarted | Yes |

## Layer 2: Hook Governance

Intercepts I/O at kernel boundary (reused from LangGraph SDK):
- **HTTP:** httpx, requests, urllib3, urllib
- **Database:** SQLAlchemy, asyncpg, psycopg2, pymongo, redis, MySQL, SQLite
- **File I/O:** builtins.open(), os.fdopen()

## Layer 3: Activity Context Mapping

`WorkflowSpanProcessor` maps OTel trace_id to (workflow_id, activity_id) so hooks fired during tool execution are attributed to the correct governance activity.

## Data Flow

```
agent.invoke(input, config={"callbacks": [handler]})
  │
  ├─ on_chain_start(root) → SignalReceived + WorkflowStarted + LLMStarted pre-screen
  ├─ on_tool_start → ToolStarted → enforce verdict → register OTel span
  │   └─ HTTP/DB/File hooks fire during tool execution (Layer 2)
  ├─ on_tool_end → ToolCompleted → enforce behavior rules
  ├─ on_chat_model_start → LLMStarted → PII redaction
  ├─ on_llm_end → LLMCompleted (observation)
  └─ on_chain_end(root) → WorkflowCompleted → reset state
```

## Pre-Screen API

`handler.pre_screen(input)` provides reliable verdict enforcement before the agent loop starts. Callback-based enforcement is best-effort since `AgentExecutor` may catch callback exceptions.

## Verdict Enforcement

5-tier system: ALLOW < CONSTRAIN < REQUIRE_APPROVAL < BLOCK < HALT

Enforcement points: tool_start, tool_end, agent_action, chain_start (nested), llm_start (pre-screen only).
Observation-only: chain_end, llm_end, agent_finish.
