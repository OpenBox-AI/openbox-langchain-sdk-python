"""Factory for creating configured OpenBoxLangChainMiddleware instances.

Usage:
    from openbox_langchain import create_openbox_langchain_middleware
    middleware = create_openbox_langchain_middleware(
        api_url=os.environ["OPENBOX_URL"],
        api_key=os.environ["OPENBOX_API_KEY"],
        agent_name="MyAgent",
    )
    agent = create_agent(model=..., tools=[...], middleware=[middleware])
    result = agent.invoke({"messages": [("user", "Hello")]})
"""

from __future__ import annotations

import dataclasses
from typing import Any

from openbox_langchain.middleware import (
    OpenBoxLangChainMiddleware,
    OpenBoxLangChainMiddlewareOptions,
)


def create_openbox_langchain_middleware(
    *,
    api_url: str,
    api_key: str,
    agent_name: str | None = None,
    agent_did: str | None = None,
    agent_private_key: str | None = None,
    governance_timeout: float = 30.0,
    validate: bool = True,
    sqlalchemy_engine: Any = None,
    **kwargs: Any,
) -> OpenBoxLangChainMiddleware:
    """Create a configured OpenBoxLangChainMiddleware for create_agent(middleware=[...]).

    Validates the API key and sets up global config before returning the middleware.

    Args:
        api_url: Base URL of your OpenBox Core instance.
        api_key: API key in ``obx_live_*`` or ``obx_test_*`` format.
        agent_name: Agent name as configured in the dashboard.
        agent_did: Optional OpenBox agent DID. Falls back to ``OPENBOX_AGENT_DID``.
        agent_private_key: Optional raw Ed25519 private key seed. Falls back to
            ``OPENBOX_AGENT_PRIVATE_KEY``.
        governance_timeout: HTTP timeout in seconds (default 30.0).
        validate: If True, validates the API key against the server on startup.
        sqlalchemy_engine: Optional SQLAlchemy Engine for DB governance.
        **kwargs: Additional kwargs forwarded to OpenBoxLangChainMiddlewareOptions.

    Returns:
        A configured ``OpenBoxLangChainMiddleware`` ready for create_agent().
    """
    from openbox_langgraph.config import initialize

    initialize(
        api_url=api_url,
        api_key=api_key,
        governance_timeout=governance_timeout,
        validate=validate,
        agent_did=agent_did,
        agent_private_key=agent_private_key,
    )

    valid_fields = {f.name for f in dataclasses.fields(OpenBoxLangChainMiddlewareOptions)}
    options = OpenBoxLangChainMiddlewareOptions(
        agent_name=agent_name,
        governance_timeout=governance_timeout,
        sqlalchemy_engine=sqlalchemy_engine,
        **{k: v for k, v in kwargs.items() if k in valid_fields},
    )
    return OpenBoxLangChainMiddleware(options)
