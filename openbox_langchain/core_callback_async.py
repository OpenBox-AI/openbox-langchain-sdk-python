"""Async pure LangChain-Core callback handler — the PRODUCER of tool/LLM
ActivityStarted/ActivityCompleted events on the normal async graph path.

``raise_error=True, run_inline=True`` (Phase 0-measured config): ``on_tool_start``
raises PROPAGATE on the async tool path; a start-raise here is NOT caught by
``on_tool_error`` (``BaseTool.arun`` awaits ``on_tool_start`` outside its body
try-block) — see ``core_callback_tool_complete.close_orphan_start`` (C6).

Cross-dispatch (Phase 0-measured): the ASYNC callback manager ALSO runs a
SYNC ``BaseCallbackHandler`` once per event. When both handlers are installed,
both fire — the evaluate-once/enforce-from-stash contract on
``ActivityBridge.start_result`` is what prevents a double gate call (C2).

Tool and LLM lifecycle methods live in
``core_callback_async_tool_mixin``/``core_callback_async_llm_mixin`` (split to
stay under 200 lines per file); this module is the composition + ctor.
"""

from __future__ import annotations

from langchain_core.callbacks import AsyncCallbackHandler

from openbox_langchain.core_callback_async_llm_mixin import AsyncLLMLifecycleMixin
from openbox_langchain.core_callback_async_tool_mixin import AsyncToolLifecycleMixin
from openbox_langchain.core_callback_options import OpenBoxLangChainCoreCallbackOptions

__all__ = ["OpenBoxLangChainCoreAsyncCallbackHandler"]


class OpenBoxLangChainCoreAsyncCallbackHandler(
    AsyncToolLifecycleMixin, AsyncLLMLifecycleMixin, AsyncCallbackHandler
):
    """Async governance callback: tool + LLM lifecycle, evaluate-once/enforce-from-stash."""

    raise_error = True
    run_inline = True

    def __init__(self, options: OpenBoxLangChainCoreCallbackOptions) -> None:
        super().__init__()
        self._options = options
