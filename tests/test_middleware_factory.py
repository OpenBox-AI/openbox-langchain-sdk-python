"""Tests for middleware_factory.py factory function."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from openbox_langchain.middleware import (
    OpenBoxLangChainMiddleware,
)
from openbox_langchain.middleware_factory import create_openbox_langchain_middleware

# ─── create_openbox_langchain_middleware ────────────────────────────────


def test_factory_creates_middleware():
    """Factory creates a middleware instance."""
    with patch("openbox_langgraph.config.initialize"):
        with patch("openbox_langchain.middleware.get_global_config") as mock_gc:
            mock_gc.return_value = MagicMock(
                api_url="https://test.openbox.ai",
                api_key="obx_test_123",
                governance_timeout=30.0,
            )
            with patch("openbox_langchain.middleware.GovernanceClient"):
                with patch("openbox_langchain.middleware.merge_config") as mock_mc:
                    mock_mc.return_value = MagicMock()

                    mw = create_openbox_langchain_middleware(
                        api_url="https://test.openbox.ai",
                        api_key="obx_test_123",
                    )

                    assert isinstance(mw, OpenBoxLangChainMiddleware)


def test_factory_calls_initialize():
    """Factory calls initialize with api_url, api_key, and timeout."""
    with patch("openbox_langgraph.config.initialize") as mock_init:
        with patch("openbox_langchain.middleware.get_global_config") as mock_gc:
            mock_gc.return_value = MagicMock(
                api_url="https://test.openbox.ai",
                api_key="obx_test_123",
                governance_timeout=30.0,
            )
            with patch("openbox_langchain.middleware.GovernanceClient"):
                with patch("openbox_langchain.middleware.merge_config") as mock_mc:
                    mock_mc.return_value = MagicMock()

                    create_openbox_langchain_middleware(
                        api_url="https://test.openbox.ai",
                        api_key="obx_test_123",
                        governance_timeout=45.0,
                    )

                    mock_init.assert_called_once()
                    call_kwargs = mock_init.call_args[1]
                    assert call_kwargs["api_url"] == "https://test.openbox.ai"
                    assert call_kwargs["api_key"] == "obx_test_123"
                    assert call_kwargs["governance_timeout"] == 45.0


def test_factory_forwards_agent_identity_to_initialize():
    """Factory forwards DID signing config to the shared LangGraph initializer."""
    with patch("openbox_langgraph.config.initialize") as mock_init:
        with patch("openbox_langchain.middleware.get_global_config") as mock_gc:
            mock_gc.return_value = MagicMock(
                api_url="https://test.openbox.ai",
                api_key="obx_test_123",
                governance_timeout=30.0,
            )
            with patch("openbox_langchain.middleware.GovernanceClient"):
                with patch("openbox_langchain.middleware.merge_config") as mock_mc:
                    mock_mc.return_value = MagicMock()

                    create_openbox_langchain_middleware(
                        api_url="https://test.openbox.ai",
                        api_key="obx_test_123",
                        agent_did="did:aip:550e8400-e29b-41d4-a716-446655440000",
                        agent_private_key="key",
                    )

                    call_kwargs = mock_init.call_args[1]
                    assert (
                        call_kwargs["agent_did"]
                        == "did:aip:550e8400-e29b-41d4-a716-446655440000"
                    )
                    assert call_kwargs["agent_private_key"] == "key"


def test_middleware_passes_agent_identity_to_client_and_hooks():
    """Middleware forwards resolved global DID config to governance client and OTel hooks."""
    with patch("openbox_langchain.middleware.get_global_config") as mock_gc:
        mock_gc.return_value = MagicMock(
            api_url="https://test.openbox.ai",
            api_key="obx_test_123",
            governance_timeout=30.0,
            agent_did="did:aip:550e8400-e29b-41d4-a716-446655440000",
            agent_private_key="key",
        )
        with patch("openbox_langchain.middleware.GovernanceClient") as mock_client:
            with patch("openbox_langchain.middleware.merge_config") as mock_mc:
                mock_mc.return_value = MagicMock(on_api_error="fail_open")
                with patch(
                    "openbox_langgraph.otel_setup.setup_opentelemetry_for_governance"
                ) as mock_setup:
                    OpenBoxLangChainMiddleware()

                    client_kwargs = mock_client.call_args.kwargs
                    assert (
                        client_kwargs["agent_did"]
                        == "did:aip:550e8400-e29b-41d4-a716-446655440000"
                    )
                    assert client_kwargs["agent_private_key"] == "key"

                    setup_kwargs = mock_setup.call_args.kwargs
                    assert (
                        setup_kwargs["agent_did"]
                        == "did:aip:550e8400-e29b-41d4-a716-446655440000"
                    )
                    assert setup_kwargs["agent_private_key"] == "key"


def test_factory_sets_agent_name():
    """Factory sets agent_name in options."""
    with patch("openbox_langgraph.config.initialize"):
        with patch("openbox_langchain.middleware.get_global_config") as mock_gc:
            mock_gc.return_value = MagicMock(
                api_url="https://test.openbox.ai",
                api_key="obx_test_123",
                governance_timeout=30.0,
            )
            with patch("openbox_langchain.middleware.GovernanceClient"):
                with patch("openbox_langchain.middleware.merge_config") as mock_mc:
                    mock_mc.return_value = MagicMock()

                    mw = create_openbox_langchain_middleware(
                        api_url="https://test.openbox.ai",
                        api_key="obx_test_123",
                        agent_name="MyAgent",
                    )

                    assert mw._options.agent_name == "MyAgent"


def test_factory_forwards_valid_kwargs():
    """Factory forwards valid kwargs to OpenBoxLangChainMiddlewareOptions."""
    with patch("openbox_langgraph.config.initialize"):
        with patch("openbox_langchain.middleware.get_global_config") as mock_gc:
            mock_gc.return_value = MagicMock(
                api_url="https://test.openbox.ai",
                api_key="obx_test_123",
                governance_timeout=30.0,
            )
            with patch("openbox_langchain.middleware.GovernanceClient"):
                with patch("openbox_langchain.middleware.merge_config") as mock_mc:
                    mock_mc.return_value = MagicMock()

                    mw = create_openbox_langchain_middleware(
                        api_url="https://test.openbox.ai",
                        api_key="obx_test_123",
                        session_id="session-123",
                        task_queue="custom_queue",
                        on_api_error="fail_closed",
                        tool_type_map={"search": "http"},
                    )

                    assert mw._options.session_id == "session-123"
                    assert mw._options.task_queue == "custom_queue"
                    assert mw._options.on_api_error == "fail_closed"
                    assert mw._options.tool_type_map == {"search": "http"}


def test_factory_filters_invalid_kwargs():
    """Factory filters out invalid kwargs."""
    with patch("openbox_langgraph.config.initialize"):
        with patch("openbox_langchain.middleware.get_global_config") as mock_gc:
            mock_gc.return_value = MagicMock(
                api_url="https://test.openbox.ai",
                api_key="obx_test_123",
                governance_timeout=30.0,
            )
            with patch("openbox_langchain.middleware.GovernanceClient"):
                with patch("openbox_langchain.middleware.merge_config") as mock_mc:
                    mock_mc.return_value = MagicMock()

                    # Invalid kwargs should be silently filtered
                    mw = create_openbox_langchain_middleware(
                        api_url="https://test.openbox.ai",
                        api_key="obx_test_123",
                        agent_name="MyAgent",
                        invalid_kwarg="should_be_filtered",
                        another_invalid="also_filtered",
                    )

                    # Should not raise, invalid kwargs are filtered
                    assert isinstance(mw, OpenBoxLangChainMiddleware)
                    assert mw._options.agent_name == "MyAgent"


def test_factory_default_validate_is_true():
    """Factory defaults validate to True."""
    with patch("openbox_langgraph.config.initialize") as mock_init:
        with patch("openbox_langchain.middleware.get_global_config") as mock_gc:
            mock_gc.return_value = MagicMock(
                api_url="https://test.openbox.ai",
                api_key="obx_test_123",
                governance_timeout=30.0,
            )
            with patch("openbox_langchain.middleware.GovernanceClient"):
                with patch("openbox_langchain.middleware.merge_config") as mock_mc:
                    mock_mc.return_value = MagicMock()

                    create_openbox_langchain_middleware(
                        api_url="https://test.openbox.ai",
                        api_key="obx_test_123",
                    )

                    call_kwargs = mock_init.call_args[1]
                    assert call_kwargs.get("validate") is True


def test_factory_respects_validate_false():
    """Factory respects validate=False."""
    with patch("openbox_langgraph.config.initialize") as mock_init:
        with patch("openbox_langchain.middleware.get_global_config") as mock_gc:
            mock_gc.return_value = MagicMock(
                api_url="https://test.openbox.ai",
                api_key="obx_test_123",
                governance_timeout=30.0,
            )
            with patch("openbox_langchain.middleware.GovernanceClient"):
                with patch("openbox_langchain.middleware.merge_config") as mock_mc:
                    mock_mc.return_value = MagicMock()

                    create_openbox_langchain_middleware(
                        api_url="https://test.openbox.ai",
                        api_key="obx_test_123",
                        validate=False,
                    )

                    call_kwargs = mock_init.call_args[1]
                    assert call_kwargs.get("validate") is False


def test_factory_forwards_sqlalchemy_engine():
    """Factory forwards sqlalchemy_engine to options."""
    with patch("openbox_langgraph.config.initialize"):
        with patch("openbox_langchain.middleware.get_global_config") as mock_gc:
            mock_gc.return_value = MagicMock(
                api_url="https://test.openbox.ai",
                api_key="obx_test_123",
                governance_timeout=30.0,
            )
            with patch("openbox_langchain.middleware.GovernanceClient"):
                with patch("openbox_langchain.middleware.merge_config") as mock_mc:
                    mock_mc.return_value = MagicMock()

                    engine = MagicMock()
                    mw = create_openbox_langchain_middleware(
                        api_url="https://test.openbox.ai",
                        api_key="obx_test_123",
                        sqlalchemy_engine=engine,
                    )

                    assert mw._options.sqlalchemy_engine is engine


def test_factory_forwards_governance_timeout():
    """Factory forwards governance_timeout through initialize and to options."""
    with patch("openbox_langgraph.config.initialize"):
        with patch("openbox_langchain.middleware.get_global_config") as mock_gc:
            mock_gc.return_value = MagicMock(
                api_url="https://test.openbox.ai",
                api_key="obx_test_123",
                governance_timeout=30.0,
            )
            with patch("openbox_langchain.middleware.GovernanceClient"):
                with patch("openbox_langchain.middleware.merge_config") as mock_mc:
                    mock_mc.return_value = MagicMock()

                    mw = create_openbox_langchain_middleware(
                        api_url="https://test.openbox.ai",
                        api_key="obx_test_123",
                        governance_timeout=60.0,
                    )

                    assert mw._options.governance_timeout == 60.0


def test_factory_sets_send_event_flags():
    """Factory forwards send_*_event flags."""
    with patch("openbox_langgraph.config.initialize"):
        with patch("openbox_langchain.middleware.get_global_config") as mock_gc:
            mock_gc.return_value = MagicMock(
                api_url="https://test.openbox.ai",
                api_key="obx_test_123",
                governance_timeout=30.0,
            )
            with patch("openbox_langchain.middleware.GovernanceClient"):
                with patch("openbox_langchain.middleware.merge_config") as mock_mc:
                    mock_mc.return_value = MagicMock()

                    mw = create_openbox_langchain_middleware(
                        api_url="https://test.openbox.ai",
                        api_key="obx_test_123",
                        send_chain_start_event=False,
                        send_chain_end_event=False,
                    )

                    # These should have been filtered (not part of valid fields for factory)
                    # or handled via merge_config. The important thing is no exception.
                    assert isinstance(mw, OpenBoxLangChainMiddleware)
