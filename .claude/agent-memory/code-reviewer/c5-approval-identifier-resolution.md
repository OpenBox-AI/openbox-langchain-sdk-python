---
name: c5-approval-identifier-resolution
description: How the LangGraph adapter resolves the activity_id an HITL approval poll targets, and the latent hang risk when Core echoes an identifier field
metadata:
  type: project
---

The C5 approval-identity path spans two repos (openbox-langgraph-sdk-python + openbox-sdk-python base).

**Fact:** `LangGraphFrameworkAdapter._raise_pending_approval` (core_adapter.py) builds the raised `GovernanceBlockedError.identifier` via `_resolve_identifier(result, ctx)`, which returns `result.raw.get("identifier")` FIRST, falling back to `ctx.activity_id` only when absent. `EvaluationResult.from_dict` sets `raw = dict(data)` = the ENTIRE Core response.

`ainvoke`'s outer HITL poll uses `_approval_poll_activity_id(hook_err, run_id)` = `hook_err.identifier or f"{run_id}-hook"`. Core matches approvals on `(workflow_id, run_id, activity_id)` EXACTLY (client.py `poll_approval` posts only those three; `poll_until_decision` is unbounded `while True`, no deadline — hitl.py).

**Why this matters:** For a TOOL REQUIRE_APPROVAL, correctness depends on `.identifier` being the tool's real activity_id. That holds ONLY when Core does NOT echo an `identifier` field in the lifecycle evaluate response (then it falls back to `ctx.activity_id`, which is correct because `run_inline=True` runs the callback inside the ToolNode-seam `activity_scope`). If a future Core build echoes any `identifier` on a lifecycle response, the poll key diverges → `ainvoke` hangs forever.

**How to apply:** FakeCore tests cannot catch this (FakeCore never echoes `identifier`). Any review of the approval path should flag the `result.raw["identifier"]` precedence as the single point that turns a wrong Core response into a production hang. The robust fix is to pass the known tool activity_id explicitly rather than trusting error-carried identifier for the tool-lifecycle case. See [[MEMORY]].
