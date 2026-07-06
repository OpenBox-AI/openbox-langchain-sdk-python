"""OpenBox LangChain SDK — governance for LangChain agents.

Pure LangChain / LangChain-Core adapter built on the framework-neutral base SDK
(``openbox_core``) plus the LangChain-Core public callback system. It carries no
dependency on the LangGraph runtime or the OpenBox LangGraph SDK.

The middleware / ``create_agent`` surface requires the optional ``agent`` extra
(``pip install openbox-langchain-sdk-python[agent]``) and is imported lazily so
that ``import openbox_langchain`` succeeds with only the base dependencies.
"""

from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING, Any

# Eager imports: the ActivityBridge + core callback handlers depend ONLY on
# openbox_core + langchain-core (never the `agent` extra, never langgraph /
# openbox_langgraph) — see tests/test_package_boundaries.py
# test_base_import_pulls_nothing_forbidden, which enforces this at import time.
from openbox_langchain.activity_bridge import ActivityBridge
from openbox_langchain.core_callback import (
    OpenBoxLangChainCoreAsyncCallbackHandler,
    OpenBoxLangChainCoreCallbackOptions,
    OpenBoxLangChainCoreSyncCallbackHandler,
)

try:
    __version__ = version("openbox-langchain-sdk-python")
except PackageNotFoundError:
    __version__ = "unknown"

# Lazily-resolved public symbols → the module that defines each. Imported on
# first attribute access (PEP 562) so the base import never eagerly pulls in the
# middleware modules and their heavier `langchain` dependency.
_LAZY_EXPORTS = {
    "OpenBoxLangChainMiddleware": "openbox_langchain.middleware",
    "OpenBoxLangChainMiddlewareOptions": "openbox_langchain.middleware",
    "create_openbox_langchain_middleware": "openbox_langchain.middleware_factory",
}

if TYPE_CHECKING:  # static type-checkers still see the real symbols
    from openbox_langchain.middleware import (
        OpenBoxLangChainMiddleware,
        OpenBoxLangChainMiddlewareOptions,
    )
    from openbox_langchain.middleware_factory import create_openbox_langchain_middleware


def __getattr__(name: str) -> Any:
    module_path = _LAZY_EXPORTS.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:  # pragma: no cover - surfaced to caller
        # The LangChain agent middleware needs the optional `agent` extra once
        # rebuilt. The chained cause (see below) shows the real missing module —
        # do not assume the `agent` extra alone resolves it.
        raise ImportError(
            f"{name!r} (the LangChain agent middleware) could not be imported: "
            f"{exc}. It requires the optional 'agent' extra "
            "(pip install openbox-langchain-sdk-python[agent]); if that is already "
            "installed, see the chained error above for the underlying cause."
        ) from exc
    return getattr(module, name)


def __dir__() -> list[str]:
    return sorted([*globals().keys(), *_LAZY_EXPORTS])


__all__ = [
    "ActivityBridge",
    "OpenBoxLangChainCoreAsyncCallbackHandler",
    "OpenBoxLangChainCoreCallbackOptions",
    "OpenBoxLangChainCoreSyncCallbackHandler",
    "OpenBoxLangChainMiddleware",
    "OpenBoxLangChainMiddlewareOptions",
    "__version__",
    "create_openbox_langchain_middleware",
]
