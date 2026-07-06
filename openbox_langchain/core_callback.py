"""Public entry point for the pure LangChain-Core callback adapter.

Re-exports ``OpenBoxLangChainCoreCallbackOptions`` and both handler classes.
The implementation is split across ``core_callback_options.py``,
``core_callback_async*.py``, and ``core_callback_sync*.py`` (per-file 200-line
modularization) — this module is the single import surface consumers use.
"""

from __future__ import annotations

from openbox_langchain.core_callback_async import OpenBoxLangChainCoreAsyncCallbackHandler
from openbox_langchain.core_callback_options import OpenBoxLangChainCoreCallbackOptions
from openbox_langchain.core_callback_sync import OpenBoxLangChainCoreSyncCallbackHandler

__all__ = [
    "OpenBoxLangChainCoreAsyncCallbackHandler",
    "OpenBoxLangChainCoreCallbackOptions",
    "OpenBoxLangChainCoreSyncCallbackHandler",
]
