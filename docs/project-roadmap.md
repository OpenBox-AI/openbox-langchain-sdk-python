# Project Roadmap — OpenBox LangChain SDK

## Current Status

**Version:** v0.2.0 (production-ready baseline with DID signing support)
**Release Date:** 2026-05-28
**Status:** Stable for middleware-based governance of LangChain agents

### Completed

- ✅ Middleware pattern migration (callbacks → AgentMiddleware)
- ✅ Core hooks: before/after agent, wrap model/tool calls
- ✅ Verdict enforcement (5-tier system)
- ✅ HITL approval polling
- ✅ Pre-screen guardrails
- ✅ PII redaction
- ✅ OTel span registration + context mapping
- ✅ Async/sync bridge with context preservation
- ✅ DID signing configuration via `agent_did` and `agent_private_key`
- ✅ Test suite (116 tests, 100% pass; 89% package coverage)
- ✅ Working example agent (content-builder-agent)

### In Progress

- 🔄 **Documentation Suite**
  - README.md update (middleware API)
  - System architecture documentation
  - Code standards & patterns
  - Codebase summary
  - Project overview & PDR
  - Project roadmap (this file)

## Phases

### Phase 1: Core Middleware Implementation (COMPLETE)

**Duration:** 2 weeks
**Status:** ✅ Complete

**Deliverables:**
- `OpenBoxLangChainMiddleware` class (239 lines)
- `create_openbox_langchain_middleware()` factory
- Lifecycle hooks: before/after agent, wrap_model_call, wrap_tool_call
- Async/sync dual support with OTel context preservation
- Configuration via `OpenBoxLangChainMiddlewareOptions`

**Key decisions:**
- Use AgentMiddleware pattern (LangChain v1.0+) instead of callbacks
- Import governance infrastructure from langgraph SDK (no copying)
- Sync hooks bridge to async with context propagation
- Pre-screen guardrails on first LLM call with caching

**Tests:** 116 tests, 100% pass rate

### Phase 2: Documentation (COMPLETE)

**Duration:** 1 week
**Status:** ✅ Complete

**Deliverables:**
- README.md — Quick start + configuration guide
- Project Overview PDR — Requirements, success metrics
- System Architecture — 3-layer design, data flow, verdict system
- Code Standards — Patterns, conventions, guidelines
- Codebase Summary — Module organization, dependencies
- Project Roadmap — This file

**Timeline:**
- README: ✅ Complete
- Project Overview PDR: ✅ Complete
- Codebase Summary: ✅ Complete
- Code Standards: ✅ Complete
- System Architecture: ✅ Complete
- Project Roadmap: ✅ Complete

### Phase 3: v0.2.0 Release (COMPLETE)

**Duration:** 1 week
**Status:** ✅ Complete

**Deliverables:**
- Update AGENTS.md to reflect middleware architecture
- Add integration examples (multiple agent types)
- Performance benchmarking
- Security audit checklist
- DID signing support
- Release v0.2.0 stable

**Tasks:**
- [x] Update AGENTS.md
- [x] Create example: react_agent with tools
- [x] Security: verify no API key leaks, PII redaction coverage
- [x] Add DID signing configuration
- [x] Tag v0.2.0 release
- [ ] Create example: tool_calling_agent with multi-turn
- [ ] Benchmark: governance latency (target < 500ms)
- [ ] Create CONTRIBUTING.md

## Milestone Timeline

```
Week 1: Core Middleware ✅
├─ Middleware class + hooks
├─ Factory + configuration
├─ Test suite (116 tests)
└─ Example agent

Week 2: Documentation (Current)
├─ README + API docs
├─ Architecture docs
├─ Code standards
└─ Examples + tutorials

Week 3: Polish & Release
├─ Additional examples
├─ Performance tuning
├─ Release preparation
└─ v0.2.0 stable release
```

## Release Roadmap

### v0.1.0
- ✅ Middleware pattern (AgentMiddleware subclass)
- ✅ Lifecycle hooks (before/after agent, wrap model/tool calls)
- ✅ Verdict enforcement (5-tier system: ALLOW/CONSTRAIN/REQUIRE_APPROVAL/BLOCK/HALT)
- ✅ HITL approval flows (async polling)
- ✅ Pre-screen guardrails
- ✅ PII redaction
- ✅ OTel integration (span registration, context mapping)
- ✅ Async/sync support
- ✅ Configuration via factory
- ✅ Test suite

### v0.2.0 (Current)
- ✅ DID signing configuration via `agent_did` and `agent_private_key`
- ✅ Runtime dependency on openbox-langgraph-sdk-python >= 0.2.0
- ✅ Refreshed dependency locks for security scans
- ✅ Release coverage gate raised to 80% with 89% current package coverage

### v0.3.0 (Q3 2026)
- 🔲 Advanced verdict rules (conditional enforcement)
- 🔲 Behavior rule chains (post-execution verification)
- 🔲 Custom event builders (domain-specific events)
- 🔲 Distributed tracing (cross-service traces)
- 🔲 Performance monitoring dashboard integration
- 🔲 Rate limiting guardrails (token budgets)

### v0.4.0 (Q4 2026)
- 🔲 Multi-agent orchestration (agent composition)
- 🔲 Knowledge base integration (RAG governance)
- 🔲 Tool sandboxing (resource limits)
- 🔲 Audit logging (compliance records)
- 🔲 ML-based anomaly detection

### v1.0.0
- 🔲 Enterprise features (SSO, advanced HITL, audit)
- 🔲 SLA guarantees (99.99% uptime)
- 🔲 Multi-region deployment
- 🔲 Kubernetes integration

## Success Metrics

| Metric | Target | Status |
|--------|--------|--------|
| **Functionality** | | |
| Test coverage | 80%+ release gate | ✅ 89% (116 tests) |
| Supported agent types | ≥3 | ✅ react_agent, tool_calling_agent, custom |
| Hook coverage | 100% | ✅ before/after agent, wrap model/tool |
| **Performance** | | |
| Governance API latency | < 500ms | 🔄 ~200-500ms |
| Overhead % | < 10% of agent time | 🔄 Benchmarking |
| Pre-screen cache hit rate | > 80% | 🔄 Measuring |
| **Documentation** | | |
| README quality | Clear quick start | ✅ Complete |
| Architecture docs | All layers covered | ✅ Complete |
| Code examples | ≥3 working examples | 🔄 1 complete (content-builder-agent) |
| **Security** | | |
| PII redaction coverage | > 95% patterns | ✅ Imported from langgraph SDK |
| API key handling | No leaks in logs | ✅ Memory-only storage |
| HTTPS enforcement | Always | ✅ Enforced in client |
| **Compatibility** | | |
| Python versions | 3.11+ | ✅ 3.11, 3.12, 3.13 tested |
| LangChain versions | 0.3.0+ | ✅ v0.3.0, v0.4.0 compatible |
| LangGraph versions | 0.2.0+ | ✅ Compatible with current 1.x releases |

## Known Limitations

| Limitation | Workaround | Future Fix |
|------------|-----------|-----------|
| Single pre-screen per session | Cache response, reuse for first LLM call | Allow parameterized pre-screen |
| No custom verdict hooks | Implement in governance API rules | Client-side verdict customization |
| OTel context may be lost in sync bridge | Use async agents when possible | Improve context propagation |
| HITL timeout is global | Adjust via config at startup | Per-activity timeout |

## Dependencies & Compatibility

### Required
- Python 3.11+
- openbox-langgraph-sdk-python >= 0.2.0
- langchain >= 0.3.0
- langchain-core >= 0.3.0
- langgraph >= 0.2.0

### Optional
- sqlalchemy (for DB governance)
- asyncpg, psycopg2, pymongo, redis (DB drivers)

### Development
- pytest, pytest-asyncio
- ruff (linting)
- mypy (type checking)

## Testing & QA

### Test Coverage

**Current:** 116 tests, 100% pass rate, 89% package coverage

**Areas:**
- Unit tests for each hook handler
- Integration tests for full agent workflow
- Mock client tests for verdict enforcement
- Pre-screen caching tests
- OTel context propagation tests
- Async/sync bridge tests
- Error handling tests (network, timeout, invalid verdicts)

### CI/CD

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

### Manual Testing Checklist

- [ ] Create agent with middleware
- [ ] Invoke with ALLOW verdict
- [ ] Invoke with BLOCK verdict (expect GovernanceBlockedError)
- [ ] Invoke with REQUIRE_APPROVAL + approve in HITL
- [ ] Invoke with REQUIRE_APPROVAL + reject in HITL
- [ ] Verify pre-screen cached response reused
- [ ] Verify PII redacted in event logs
- [ ] Verify OTel spans created for tools
- [ ] Test with sync agent
- [ ] Test with async agent

## Risk Assessment

| Risk | Severity | Probability | Mitigation |
|------|----------|-------------|-----------|
| LangChain API breaking change | High | Medium | Pin version range, monitor releases, 1 month migration window |
| OpenBox Core unavailable | High | Low | Timeout 30s, log error, continue agent (fail-open) |
| OTel context loss in sync bridge | Medium | Medium | Test thoroughly, document limitations, improve bridge |
| PII redaction false negatives | Medium | Low | Test patterns, submit new patterns to langgraph SDK |
| HITL approval timeout during execution | Medium | Low | Timeout configurable, document best practices |

## Next Steps (Q2 2026)

1. **Documentation Polish** (1 week)
   - Finalize Phase 2 docs
   - Add integration examples
   - Create contributing guide

2. **Performance Optimization** (2 weeks)
   - Benchmark governance latency
   - Optimize hot paths
   - Cache optimization

3. **Extended Examples** (2 weeks)
   - RAG agent with DB governance
   - Multi-agent orchestration
   - Custom tool classification

4. **v0.3.0 Planning** (1 week)
   - Collect feedback from early users
   - Design advanced features
   - Create implementation plan

## References

- **Project Overview & PDR:** `./docs/project-overview-pdr.md`
- **System Architecture:** `./docs/system-architecture.md`
- **Code Standards:** `./docs/code-standards.md`
- **Codebase Summary:** `./docs/codebase-summary.md`
- **DeepAgent SDK:** `/Users/tino/code/openbox-deepagent-sdk-python/`
- **LangGraph SDK:** `/Users/tino/code/openbox-langgraph-sdk-python/`
- **SDK Guide:** `/Users/tino/code/sdk-implementation-guide/README.md`
