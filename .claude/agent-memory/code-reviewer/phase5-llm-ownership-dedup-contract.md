---
name: phase5-llm-ownership-dedup-contract
description: How the shared LangChain-Core callback dedups LLM lifecycle across async+sync cross-dispatch, and the -pre/-c id invariants
metadata:
  type: project
---

Phase 5 moved LLM lifecycle ownership (telemetry/redaction/trace/same-id close) to the shared LangChain-Core callback (`core_callback_async_llm_mixin.py` + sync twin). Consumer keeps first-call pre-screen enforcement in `_pre_screen_input`; M13/M14 (2nd+-call enforcement) stay Non-Goals.

**Evaluate-once contract (LLM):** `on_chat_model_start` runs on BOTH async and sync handlers per event (LangChain cross-dispatch, run serially within one callback-manager dispatch — not concurrent). Dedup: resolve `get_by_event_run_id(event_run_id)` FIRST; if `existing.start_result is not None`, reuse activity_id+verdict, skip the gate. This mirrors the tool path's `record.start_result is not None` check. Resolving the alias BEFORE consuming `options.pre_screen_response` is load-bearing — otherwise the 2nd dispatch re-decides the pre-screen branch off a cleared field and double-calls the gate → duplicate ActivityStarted.

**Why:** the fix (resolve-alias-first) prevents a second unwanted `gate.aevaluate` on the 2nd cross-dispatch. Bridge `threading.Lock` covers the sync-tool-executor-thread case; the two LLM dispatches themselves are serial so no TOCTOU.

**How to apply:** any future edit to the LLM mixin ordering must keep alias-resolution before `pre_screen_response` consumption. `send_llm_completed`/`_finish_llm` guard double-send via `is_callback_owned(..., "llm_complete")` before send — needed because `on_llm_error` fires via `agenerate`'s `return_exceptions=True` after a successful `on_llm_end`.

**Id invariants:**
- Callback-owned close = SAME id as LLMStarted, NO `-c` suffix. First call's id is the pre-screen `{run_id}-pre` (resolved via H11 alias). Completion never emits an orphan `-c` on the owned path.
- `-c` suffix appears ONLY on the fallback (no-bridge/injected-client) consumer close (`langgraph_handler.py` LLMCompleted fallback branch), preserving pre-Phase-5 wire behavior. `_PreScreenClaim` (per-turn local, not shared state) routes call-1's fallback completion to the `-pre` row.

**C4:** LLM completion uses `gate.aevaluate` (verdict-only), stashed on the bridge; consumer poll-and-continues via `enforce_verdict`+`poll_until_decision`. NEVER `aevaluate_lifecycle` (would replay the whole graph). Verified: grep for `aevaluate_lifecycle` returns zero hits on any LLM path in both repos.

**M18 mapper:** `GovernanceVerdictResponse.to_evaluation_result()` (langgraph `types.py`) is a faithful inverse of `from_result` — all 16 EvaluationResult + 5 GuardrailsResult fields mapped. Import direction langgraph→core is boundary-legal; the reverse is forbidden by the langchain `test_package_boundaries.py`.
