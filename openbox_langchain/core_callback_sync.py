"""Sync pure LangChain-Core callback handler — the fail-closed fix for the
sync-only-tool corner (C2).

Phase 0-measured: a sync-only tool executed on the async graph path
(``_arun_one`` → ``_execute_tool_sync`` → ``invoke`` → ``BaseTool.run``) runs
its callbacks through the SYNC callback manager, which drives async-handler
coroutines via ``_run_coros`` — logging exceptions instead of re-raising. An
ASYNC-handler-only install is therefore silently bypassed for that corner;
this SYNC handler's raise propagates through ``BaseTool.run`` correctly
(measured PROPAGATE). Both handlers must be installed together for a fully
fail-closed pipeline; evaluate-once/enforce-from-stash on the shared
``ActivityBridge`` prevents the cross-dispatch from doubling gate calls.

Tool and LLM lifecycle methods live in
``core_callback_sync_tool_mixin``/``core_callback_sync_llm_mixin`` (split to
stay under 200 lines per file); this module is the composition + ctor.
"""

from __future__ import annotations

from langchain_core.callbacks import BaseCallbackHandler

from openbox_langchain.core_callback_options import OpenBoxLangChainCoreCallbackOptions
from openbox_langchain.core_callback_sync_llm_mixin import SyncLLMLifecycleMixin
from openbox_langchain.core_callback_sync_tool_mixin import SyncToolLifecycleMixin

__all__ = ["OpenBoxLangChainCoreSyncCallbackHandler"]


class OpenBoxLangChainCoreSyncCallbackHandler(
    SyncToolLifecycleMixin, SyncLLMLifecycleMixin, BaseCallbackHandler
):
    """Sync governance callback: tool + LLM lifecycle, evaluate-once/enforce-from-stash."""

    raise_error = True
    run_inline = True

    def __init__(self, options: OpenBoxLangChainCoreCallbackOptions) -> None:
        super().__init__()
        self._options = options
