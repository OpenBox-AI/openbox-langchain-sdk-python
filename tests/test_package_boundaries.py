"""Package-boundary guards for the pure LangChain-Core adapter.

The SDK must build on ``openbox_core`` + ``langchain-core`` only — no
``langgraph`` / ``openbox_langgraph`` imports in production code.

Full-package scan (Phase 3 success gate, P2-new-C): every production module —
including the middleware, now rebuilt against ``openbox_core`` — is import-clean,
so ``CUT_MODULES`` covers the entire ``openbox_langchain/`` package via
``rglob('*.py')`` rather than the Phase-1-narrowed ``__init__.py``-only scope.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_PKG_ROOT = _REPO_ROOT / "openbox_langchain"

# Import statements that reference these top-level packages are forbidden.
_FORBIDDEN = ("langgraph", "openbox_langgraph")

# Every production module in the package — widened from the Phase-1-narrowed
# `__init__.py`-only scope now that the middleware is rebuilt import-clean.
CUT_MODULES = sorted(_PKG_ROOT.rglob("*.py"))

# Dynamic-import call targets whose FIRST string-literal argument names a module.
_DYNAMIC_IMPORT_FUNCS = ("import_module", "__import__")


def _root(module: str) -> str:
    return module.split(".", 1)[0]


def _forbidden_imports(source: str) -> list[str]:
    """Return module names imported from a forbidden root.

    AST-based so it is robust to multi-target imports (`import a, b`), aliases
    (`import x.y as z`), and parenthesised `from` imports — all of which a regex
    scan misses. Also catches static-string dynamic imports
    (`importlib.import_module("...")`, `__import__("...")`) since middleware may
    defer imports that way.
    """
    hits: list[str] = []
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:  # handles `import a, b`
                if _root(alias.name) in _FORBIDDEN:
                    hits.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            # module is None for `from . import x` (relative) — never forbidden here
            if node.module and _root(node.module) in _FORBIDDEN:
                hits.append(node.module)
        elif isinstance(node, ast.Call):
            func = node.func
            fname = getattr(func, "attr", None) or getattr(func, "id", None)
            if fname in _DYNAMIC_IMPORT_FUNCS and node.args:
                arg0 = node.args[0]
                if isinstance(arg0, ast.Constant) and isinstance(arg0.value, str):
                    if _root(arg0.value) in _FORBIDDEN:
                        hits.append(arg0.value)
    return hits


def test_base_import_pulls_nothing_forbidden() -> None:
    """`import openbox_langchain` must not drag in langgraph/openbox_langgraph.

    Runs in a fresh interpreter so a module already imported by the test session
    cannot mask the result. Asserts on ``sys.modules`` after the base import, so
    it holds even if langgraph happens to be installed (transitively via the
    optional ``agent`` extra).
    """
    code = (
        "import openbox_langchain, sys\n"
        "bad = [m for m in sys.modules "
        "if m in ('langgraph', 'openbox_langgraph') "
        "or m.startswith('langgraph.') or m.startswith('openbox_langgraph.')]\n"
        "assert not bad, f'base import pulled forbidden modules: {bad}'\n"
    )
    subprocess.check_call([sys.executable, "-c", code])


def test_cut_modules_have_no_forbidden_imports() -> None:
    """Statically assert the cut modules import no forbidden package."""
    for path in CUT_MODULES:
        hits = _forbidden_imports(path.read_text())
        rel = path.relative_to(_REPO_ROOT)
        assert not hits, f"{rel} imports forbidden package(s): {hits}"


def test_forbidden_import_scanner_catches_evasive_forms() -> None:
    """The AST scanner must catch the vectors a regex scan misses."""
    bad_sources = [
        "import openbox_langgraph",
        "import os, openbox_langgraph",  # multi-target import
        "import openbox_langgraph.client as c",  # alias
        "from openbox_langgraph.errors import GovernanceBlockedError",
        "from langgraph.prebuilt import ToolNode",
        "import importlib\nm = importlib.import_module('openbox_langgraph.client')",
        "x = __import__('langgraph')",
    ]
    for src in bad_sources:
        assert _forbidden_imports(src), f"scanner missed forbidden import: {src!r}"

    good_sources = [
        "import openbox_core",
        "from openbox_core.contracts.events import activity_started",
        "from langchain_core.callbacks import AsyncCallbackHandler",
        "from . import middleware",  # relative import, module is None
        "import importlib\nm = importlib.import_module('openbox_core.gate')",
    ]
    for src in good_sources:
        assert not _forbidden_imports(src), f"scanner false-positive on: {src!r}"
