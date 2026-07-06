# Project Overview & PDR — OpenBox LangChain SDK

## Project Summary

OpenBox governance SDK for LangChain agents. Provides real-time policy enforcement, guardrails, Human-in-the-Loop (HITL) approval flows, and hook-level governance (HTTP/DB/File I/O) via `AgentMiddleware` integration pattern.

**Status:** v0.2.0 (production-ready middleware with DID signing support)
**Architecture:** Middleware-based (AgentMiddleware subclass, not callbacks)
**Language:** Python 3.11+

## Goals & Vision

Enable developers to:
1. Deploy LangChain agents with governance guardrails
2. Control agent actions at execution time (pre-screen, tool approval, HITL)
3. Monitor and audit all agent I/O (HTTP, DB, files)
4. Enforce organizational policies without modifying agent code
5. Support both sync and async agent execution

## Functional Requirements

### FR1: Agent Middleware Integration
- **Requirement:** Integrate with LangChain v1.0+ `AgentMiddleware` interface
- **Why:** LangChain deprecated callbacks in favor of middleware pattern
- **Acceptance:** `create_openbox_langchain_middleware()` factory returns configured middleware, works with `create_react_agent()` and other LangChain agent builders
- **Status:** COMPLETE

### FR2: Lifecycle Interception
- **Requirement:** Hook into agent execution lifecycle
- **Hooks:**
  - `before_agent` / `abefore_agent` — Pre-screen guardrails before agent starts
  - `wrap_model_call` / `awrap_model_call` — LLM calls (PII redaction, policy enforcement)
  - `wrap_tool_call` / `awrap_tool_call` — Tool execution (governance verdicts, OTel tracing)
  - `after_agent` / `aafter_agent` — Cleanup, session close
- **Status:** COMPLETE

### FR3: Governance Verdict System
- **Requirement:** 5-tier verdict enforcement
  - ALLOW (permit)
  - CONSTRAIN (rate limit, truncate)
  - REQUIRE_APPROVAL (HITL polling)
  - BLOCK (raise GovernanceBlockedError)
  - HALT (raise GovernanceHaltError)
- **Enforcement points:** Tool start, tool end, agent action, LLM start
- **Status:** COMPLETE

### FR4: HITL Approval Flows
- **Requirement:** Async polling loop for `require_approval` verdicts
- **Features:**
  - Configurable poll interval (default 5000ms)
  - Approval/rejection/timeout handling
  - Re-raises `ApprovalRejectedError` or `ApprovalTimeoutError` on decision
- **Status:** COMPLETE (via langgraph SDK)

### FR5: Layer 2 Hook Governance
- **Requirement:** Intercept I/O at kernel boundary (not app level)
- **Coverage:** HTTP (httpx, requests), DB (SQLAlchemy, asyncpg, psycopg2), File I/O
- **Implementation:** Imported from `openbox-langgraph-sdk-python` (not copied)
- **Status:** COMPLETE

### FR6: Layer 3 Activity Context Mapping
- **Requirement:** Link OTel trace_id to (workflow_id, activity_id)
- **Purpose:** Attribute hook-level I/O to correct governance activity
- **Implementation:** `WorkflowSpanProcessor` (imported, configured in middleware constructor)
- **Status:** COMPLETE

### FR7: Async/Sync Dual Support
- **Requirement:** Support both sync and async agent execution
- **Implementation:** Sync hooks use `_run_async()` bridge with OTel context propagation
- **Status:** COMPLETE

### FR8: Pre-Screen Guardrails
- **Requirement:** Evaluate user input before agent loop starts
- **Implementation:** First LLM call during `before_agent`, reuse response to avoid duplicate
- **Purpose:** Reliable blocking before agent consumes tokens
- **Status:** COMPLETE

### FR9: PII Redaction
- **Requirement:** Detect and redact personally identifiable info in model calls
- **Implementation:** Event building helpers in `middleware_hooks.py`
- **Status:** COMPLETE

### FR10: Configuration & Initialization
- **Requirement:** Single-point configuration via `create_openbox_langchain_middleware()`
- **Features:**
  - API key validation on startup
  - Timeout configuration
  - Agent name / session ID tracking
  - Optional SQLAlchemy engine for DB governance
  - Optional tool type classification
- **Status:** COMPLETE

## Non-Functional Requirements

### NFR1: Performance
- **Requirement:** Governance overhead < 10% of agent execution time
- **Target:** < 2s for governance API calls (timeout: 30s)
- **Async:** All I/O non-blocking via httpx async client
- **Status:** ON TRACK

### NFR2: Security
- **API keys:** Stored in memory only, not logged
- **Network:** HTTPS-only (OpenBox API enforced)
- **Secrets:** PII redaction in LLM events
- **Status:** COMPLETE

### NFR3: Reliability
- **Retry:** Automatic retries for transient network errors (via httpx)
- **Timeout:** Configurable governance API timeout (default 30s)
- **Graceful degradation:** Sync hooks with OTel context propagation
- **Status:** COMPLETE

### NFR4: Observability
- **Logging:** DEBUG logs at key decision points
- **OTel:** Automatic span propagation for distributed tracing
- **Metrics:** Verdict counts, approval loops, latency (via OpenBox dashboard)
- **Status:** COMPLETE

### NFR5: Compatibility
- **Python:** 3.11+ (type hints, async/await stable)
- **LangChain:** 0.3.0+ (AgentMiddleware required)
- **LangGraph:** 0.2.0+ (create_react_agent, create_tool_calling_agent)
- **Status:** COMPLETE

### NFR6: Maintainability
- **Code reuse:** Import governance infrastructure from langgraph SDK (no copying)
- **Module count:** 5 new modules (middleware, factory, hooks, handlers, tool_hook)
- **Test coverage:** 116 tests, 100% pass rate
- **Documentation:** This file, system-architecture, codebase-summary, code-standards
- **Status:** ON TRACK

## Architecture

Middleware-based (3 layers):

**Layer 1:** AgentMiddleware hooks intercept agent lifecycle
- `before_agent` → WorkflowStarted + SignalReceived + pre-screen guardrails
- `wrap_model_call` → LLMStarted (PII redaction) → Model → LLMCompleted
- `wrap_tool_call` → ToolStarted → Tool (OTel spans) → ToolCompleted
- `after_agent` → WorkflowCompleted + cleanup

**Layer 2:** Hook governance (imported from langgraph SDK)
- HTTP/DB/File I/O interception at kernel boundary
- Verdict enforcement with OTel span attribution

**Layer 3:** Activity context mapping
- `WorkflowSpanProcessor` maps OTel trace_id → (workflow_id, activity_id)
- Enables correct activity attribution for Layer 2 hooks

## Key Metrics & Success Criteria

| Metric | Target | Current |
|--------|--------|---------|
| Test coverage | 80%+ release gate | 89% (116 tests) |
| Governance latency | < 2s | On track |
| API key validation | On startup | Complete |
| Async/sync support | Both | Complete |
| Layer 2 hook coverage | HTTP/DB/File | Complete (imported) |
| Documentation completeness | All sections | In progress |
| Example agent | 1 working example | Complete (content-builder-agent) |

## Dependencies

**Runtime:**
- openbox-langgraph-sdk-python >= 0.2.0
- langchain >= 0.3.0, langchain-core >= 0.3.0
- langgraph >= 0.2.0

**Dev:**
- pytest, pytest-asyncio
- ruff, mypy

## Timeline & Milestones

**Phase 1: Core Middleware (COMPLETE)**
- Middleware class + factory
- Lifecycle hooks (before/after agent, wrap calls)
- Verdict enforcement & HITL polling
- Event building & PII redaction
- Tests & examples

**Phase 2: Documentation (COMPLETE)**
- README update (middleware API)
- Architecture documentation
- Code standards & patterns
- Codebase summary
- Project roadmap

**Phase 3: v0.2.0 Release (COMPLETE)**
- Performance optimization
- Additional examples
- Community feedback integration
- DID signing configuration
- Release v0.2.0 stable

## Risks & Mitigation

| Risk | Impact | Mitigation |
|------|--------|-----------|
| LangChain API changes | Medium | Monitor LangChain releases, version pin at 0.3.0+ |
| OpenBox Core unavailable | High | Timeout gracefully, log error, continue agent execution |
| Hook interference | Medium | Test with real agents, monitor CPU overhead |
| Async context loss | Medium | Proper OTel context propagation in sync bridge |

## References

- **System Architecture:** `./docs/system-architecture.md`
- **Code Standards:** `./docs/code-standards.md`
- **Codebase Summary:** `./docs/codebase-summary.md`
- **DeepAgent SDK:** `/Users/tino/code/openbox-deepagent-sdk-python/` (reference implementation)
- **LangGraph SDK:** `/Users/tino/code/openbox-langgraph-sdk-python/` (governance infrastructure)
- **SDK Guide:** `/Users/tino/code/sdk-implementation-guide/README.md`
