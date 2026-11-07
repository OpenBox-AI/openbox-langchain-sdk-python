"""Exposure barriers for LangChain tool calls.

The base SDK carries whole-run state on its own: run the agent inside
``openbox_core.exposure_scope(...)`` and every event carries the scope,
purpose and instance. What a single tool call touches is different: the
tool's start event is evaluated before the tool body runs, so only the
adapter can declare it in time. Declare it once per tool::

    search = exposure_tool(search_docs, ToolExposure(
        action_class="retrieve",
        participants=[Participant("destination", "dms:main")],
        payload=lambda args: json.dumps(args, sort_keys=True),
        returned=lambda result: [Resource(d["uri"], d["labels"]) for d in result.artifact],
    ))

or by name, with ``tool_exposure={"search_docs": ToolExposure(...)}`` on the
middleware or callback options. The adapter then:

1. sends the action class, resources, participants and payload hash on the
   tool's start event, so the barrier sees the call before it runs;
2. hands the tool its verdict token (:func:`current_verdict_token`,
   :func:`current_tool_call`), so the tool can ``relay_via_executor``;
3. sends what the tool returned on its completed event, for the admission
   check. Results come from ``returned``, from a ``content_and_artifact``
   tool whose artifact is a list of :class:`~openbox_core.Resource`, or from
   :func:`report_returned` inside the tool body.
"""

from __future__ import annotations

import contextlib
import contextvars
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any

from openbox_core.exposure import (
    ACTION_CLASSES,
    Participant,
    Resource,
    governed_action,
    hash_payload,
)

__all__ = [
    "ToolCallGrant",
    "ToolExposure",
    "current_tool_call",
    "current_verdict_token",
    "exposure_tool",
    "report_returned",
]

_ATTR = "_openbox_exposure"


@dataclass(frozen=True)
class ToolExposure:
    """What one tool's calls touch. Each field may be fixed, or a function of
    the tool call's arguments (``resources``, ``participants``, ``payload``)
    or of its result (``returned``)."""

    action_class: str
    resources: Iterable[Resource] | Callable[[dict[str, Any]], Iterable[Resource]] = ()
    participants: Iterable[Participant] | Callable[[dict[str, Any]], Iterable[Participant]] = ()
    #: The exact body the tool will send, as a function of its arguments. Its
    #: hash binds the call, and an allowed call then gets a verdict token.
    payload: Callable[[dict[str, Any]], bytes | str] | None = None
    #: The objects the tool returned, as a function of its result.
    returned: Callable[[Any], Iterable[Resource]] | None = None

    def __post_init__(self) -> None:
        if self.action_class not in ACTION_CLASSES:
            raise ValueError(f"unknown action_class {self.action_class!r}")


@dataclass
class ToolCallGrant:
    """One governed tool call, as the tool body sees it."""

    action_class: str
    resources: list[Resource]
    participants: list[Participant]
    payload_hash: str | None
    verdict_token: str | None = None
    returned: list[Resource] = field(default_factory=list)


_call: contextvars.ContextVar[ToolCallGrant | None] = contextvars.ContextVar(
    "openbox_langchain_tool_call", default=None
)


def exposure_tool(tool: Any, exposure: ToolExposure) -> Any:
    """Attach ``exposure`` to a LangChain tool and return the tool."""
    tool.__dict__[_ATTR] = exposure
    return tool


def resolve_exposure(
    registry: dict[str, ToolExposure] | None, tool_name: str, tool: Any = None
) -> ToolExposure | None:
    """The tool's declared exposure: the options registry wins over the tool."""
    if registry and tool_name in registry:
        return registry[tool_name]
    found = getattr(tool, "__dict__", {}).get(_ATTR) if tool is not None else None
    return found if isinstance(found, ToolExposure) else None


def _value(v: Any, args: dict[str, Any]) -> list[Any]:
    return list(v(args) if callable(v) else v)


def declare(exposure: ToolExposure, args: Any) -> ToolCallGrant:
    """Resolve ``exposure`` for one call's arguments."""
    a = args if isinstance(args, dict) else {}
    return ToolCallGrant(
        action_class=exposure.action_class,
        resources=_value(exposure.resources, a),
        participants=_value(exposure.participants, a),
        payload_hash=hash_payload(exposure.payload(a)) if exposure.payload else None,
    )


@contextlib.contextmanager
def declaring(call: ToolCallGrant | None) -> Iterator[None]:
    """Put ``call`` on the start event sent inside this block."""
    if call is None:
        yield
        return
    with governed_action(call.action_class, call.resources, call.participants, call.payload_hash):
        yield


@contextlib.contextmanager
def reporting(call: ToolCallGrant | None, returned: list[Resource]) -> Iterator[None]:
    """Put what the call returned on the completed event sent inside this block."""
    if call is None or not returned:
        yield
        return
    with governed_action(call.action_class, returned, call.participants):
        yield


@contextlib.contextmanager
def bound(call: ToolCallGrant | None) -> Iterator[None]:
    """Make ``call`` visible to the tool body run inside this block."""
    token = _call.set(call)
    try:
        yield
    finally:
        _call.reset(token)


def bind(call: ToolCallGrant | None) -> None:
    """Bind ``call`` without a block, for callback handlers whose start and
    end run as separate calls. :func:`unbind` clears it."""
    _call.set(call)


def unbind() -> None:
    _call.set(None)


def returned_resources(
    exposure: ToolExposure | None, call: ToolCallGrant | None, result: Any
) -> list[Resource]:
    """Every object the call returned: reported in the body, extracted by
    ``returned``, or carried as a ``content_and_artifact`` artifact."""
    out: list[Resource] = list(call.returned) if call else []
    if exposure is not None and exposure.returned is not None and result is not None:
        out.extend(exposure.returned(result))
    elif exposure is not None:
        artifact = getattr(result, "artifact", None)
        if isinstance(artifact, (list, tuple)) and artifact and all(
            isinstance(r, Resource) for r in artifact
        ):
            out.extend(artifact)
    return out


def current_tool_call() -> ToolCallGrant | None:
    """The governed tool call this code runs in, if any."""
    return _call.get()


def current_verdict_token() -> str | None:
    """The verdict token Core issued for the current tool call: present when
    the tool declared a ``payload`` and an exposure barrier took part. Pass
    it to ``openbox_core.relay_via_executor``."""
    call = _call.get()
    return call.verdict_token if call else None


def report_returned(*resources: Resource) -> None:
    """Report objects the current tool call returned, for the admission check."""
    call = _call.get()
    if call is None:
        raise RuntimeError("report_returned() called outside a governed tool call")
    call.returned.extend(resources)


def barrier_refusal_text(result: Any) -> str:
    """The text a refused tool result is replaced with. It names the class of
    refusal and the remedy, never labels or rules."""
    barrier = getattr(result, "barrier", None) or {}
    remedy = barrier.get("remedy") if isinstance(barrier, dict) else None
    return "Refused by an exposure barrier" + (f" (remedy: {remedy})." if remedy else ".")
