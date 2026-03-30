"""OpenBox LangChain SDK — governance for LangChain agents via middleware."""

from importlib.metadata import PackageNotFoundError, version

# Re-export the openbox-langgraph-sdk public surface
from openbox_langgraph import (
    DEFAULT_HITL_CONFIG,
    ApprovalExpiredError,
    ApprovalRejectedError,
    ApprovalResponse,
    ApprovalTimeoutError,
    GovernanceBlockedError,
    GovernanceClient,
    GovernanceConfig,
    GovernanceHaltError,
    GovernanceVerdictResponse,
    GuardrailsReason,
    GuardrailsResult,
    GuardrailsValidationError,
    HITLConfig,
    LangChainGovernanceEvent,
    OpenBoxAuthError,
    OpenBoxError,
    OpenBoxInsecureURLError,
    OpenBoxNetworkError,
    Verdict,
    VerdictContext,
    WorkflowEventType,
    WorkflowSpanBuffer,
    WorkflowSpanProcessor,
    build_auth_headers,
    create_span,
    enforce_verdict,
    get_global_config,
    highest_priority_verdict,
    initialize,
    is_hitl_applicable,
    lang_graph_event_to_context,
    merge_config,
    parse_approval_response,
    parse_governance_response,
    poll_until_decision,
    rfc3339_now,
    safe_serialize,
    setup_opentelemetry_for_governance,
    to_server_event_type,
    traced,
    verdict_from_string,
    verdict_priority,
    verdict_requires_approval,
    verdict_should_stop,
)

from openbox_langchain.middleware import (
    OpenBoxLangChainMiddleware,
    OpenBoxLangChainMiddlewareOptions,
)
from openbox_langchain.middleware_factory import create_openbox_langchain_middleware

try:
    __version__ = version("openbox-langchain-sdk-python")
except PackageNotFoundError:
    __version__ = "unknown"

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
    "OpenBoxInsecureURLError",
    # Middleware
    "OpenBoxLangChainMiddleware",
    "OpenBoxLangChainMiddlewareOptions",
    "OpenBoxNetworkError",
    "Verdict",
    # Verdict
    "VerdictContext",
    "WorkflowEventType",
    "WorkflowSpanBuffer",
    # OTel
    "WorkflowSpanProcessor",
    # Version
    "__version__",
    "build_auth_headers",
    # Primary API
    "create_openbox_langchain_middleware",
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
