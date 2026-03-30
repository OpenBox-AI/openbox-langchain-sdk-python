# Codebase Summary — OpenBox LangChain SDK

## Package Structure

```
openbox_langchain/
├── __init__.py                    # Public API (re-exports all modules)
├── langchain_handler.py           # NEW — AsyncCallbackHandler + factory
├── client.py                      # HTTP client to OpenBox Core
├── config.py                      # GovernanceConfig + initialization
├── verdict_handler.py             # Verdict enforcement logic
├── types.py                       # Shared data types (events, responses)
├── errors.py                      # Exception hierarchy
├── hitl.py                        # Human-in-the-loop polling
├── hook_governance.py             # Hook trigger orchestration
├── http_governance_hooks.py       # HTTP interception (httpx, requests, etc.)
├── db_governance_hooks.py         # Database interception (SQLAlchemy, asyncpg, etc.)
├── file_governance_hooks.py       # File I/O interception
├── span_processor.py              # OTel trace→activity mapping
├── otel_setup.py                  # OpenTelemetry instrumentation setup
└── tracing.py                     # @traced decorator + create_span()
```

## New vs Reused Code

| Module | Source | Lines |
|--------|--------|-------|
| langchain_handler.py | NEW | ~500 |
| __init__.py | NEW | ~130 |
| All other 13 modules | Copied from LangGraph SDK | ~3000 |

Total new code: ~630 lines. Total reused: ~3000 lines.

## Key Classes

- `OpenBoxGovernanceCallbackHandler` — Main handler (AsyncCallbackHandler)
- `OpenBoxLangChainHandlerOptions` — Configuration dataclass
- `GovernanceClient` — HTTP client for Core API
- `GovernanceConfig` — Merged governance configuration
- `WorkflowSpanProcessor` — OTel span→activity mapper

## Key Functions

- `create_openbox_langchain_handler()` — Factory function (primary entry point)
- `enforce_verdict()` — Raises on block/halt verdicts
- `poll_until_decision()` — HITL approval polling loop
- `setup_opentelemetry_for_governance()` — Initializes OTel hooks

## Test Structure

```
tests/
├── conftest.py                    # Shared fixtures
├── test_callback_handler.py       # Core handler tests
├── test_pre_screen.py             # Pre-screen method tests
├── test_utility_functions.py      # Helper function tests
└── test_handler_integration.py    # Integration workflow tests
```

99 tests, 100% pass rate.
