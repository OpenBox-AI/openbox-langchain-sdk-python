"""
OpenBox LangChain SDK — Python governance SDK for LangChain agents.

Provides OpenBox governance and observability for any LangChain agent
via AsyncCallbackHandler.

Example:
    >>> from openbox_langchain import create_openbox_langchain_handler
    >>> handler = create_openbox_langchain_handler(
    ...     api_url="https://...",
    ...     api_key="obx_live_...",
    ...     agent_name="MyAgent",
    ... )
    >>> result = agent.invoke(
    ...     {"input": "Hello"},
    ...     config={"callbacks": [handler]},
    ... )
"""

from openbox_langchain.client import GovernanceClient, build_auth_headers
from openbox_langchain.config import (
    GovernanceConfig,
    get_global_config,
    initialize,
    merge_config,
)
from openbox_langchain.errors import (
    ApprovalExpiredError,
    ApprovalRejectedError,
    ApprovalTimeoutError,
    GovernanceBlockedError,
    GovernanceHaltError,
    GuardrailsValidationError,
    OpenBoxAuthError,
    OpenBoxError,
    OpenBoxInsecureURLError,
    OpenBoxNetworkError,
)
from openbox_langchain.hitl import poll_until_decision
from openbox_langchain.langchain_handler import (
    OpenBoxGovernanceCallbackHandler,
    OpenBoxLangChainHandlerOptions,
    create_openbox_langchain_handler,
)
from openbox_langchain.otel_setup import setup_opentelemetry_for_governance
from openbox_langchain.span_processor import WorkflowSpanProcessor
from openbox_langchain.tracing import create_span, traced
from openbox_langchain.types import (
    DEFAULT_HITL_CONFIG,
    ApprovalResponse,
    GovernanceVerdictResponse,
    GuardrailsReason,
    GuardrailsResult,
    HITLConfig,
    LangChainGovernanceEvent,
    Verdict,
    WorkflowEventType,
    WorkflowSpanBuffer,
    highest_priority_verdict,
    parse_approval_response,
    parse_governance_response,
    rfc3339_now,
    safe_serialize,
    to_server_event_type,
    verdict_from_string,
    verdict_priority,
    verdict_requires_approval,
    verdict_should_stop,
)
from openbox_langchain.verdict_handler import (
    VerdictContext,
    enforce_verdict,
    is_hitl_applicable,
    lang_graph_event_to_context,
)

__all__ = [
    # Types
    "DEFAULT_HITL_CONFIG",
    # Errors
    "ApprovalExpiredError",
    "ApprovalRejectedError",
    "ApprovalResponse",
    "ApprovalTimeoutError",
    "GovernanceBlockedError",
    # Client
    "GovernanceClient",
    # Config
    "GovernanceConfig",
    "GovernanceHaltError",
    "GovernanceVerdictResponse",
    "GuardrailsReason",
    "GuardrailsResult",
    "GuardrailsValidationError",
    "HITLConfig",
    "LangChainGovernanceEvent",
    "OpenBoxAuthError",
    "OpenBoxError",
    "OpenBoxGovernanceCallbackHandler",
    "OpenBoxInsecureURLError",
    "OpenBoxLangChainHandlerOptions",
    "OpenBoxNetworkError",
    "Verdict",
    # Verdict
    "VerdictContext",
    "WorkflowEventType",
    "WorkflowSpanBuffer",
    # OTel
    "WorkflowSpanProcessor",
    "build_auth_headers",
    # Primary API
    "create_openbox_langchain_handler",
    "create_span",
    "enforce_verdict",
    "get_global_config",
    "highest_priority_verdict",
    "initialize",
    "is_hitl_applicable",
    "lang_graph_event_to_context",
    "merge_config",
    "parse_approval_response",
    "parse_governance_response",
    "poll_until_decision",
    "rfc3339_now",
    "safe_serialize",
    "setup_opentelemetry_for_governance",
    "to_server_event_type",
    "traced",
    "verdict_from_string",
    "verdict_priority",
    "verdict_requires_approval",
    "verdict_should_stop",
]
