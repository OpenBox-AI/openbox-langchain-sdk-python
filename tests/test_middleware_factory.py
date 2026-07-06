"""Tests for middleware_factory.py factory function (openbox_core based)."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from openbox_langchain.middleware import OpenBoxLangChainMiddleware
from openbox_langchain.sdk_metadata import SDK_ENGINE, SDK_LANGUAGE, SDK_PACKAGE_VERSION

API_URL = "https://test.openbox.ai"
API_KEY = "obx_test_123"
AGENT_DID = "did:aip:12345678-1234-5678-1234-567812345678"
AGENT_PRIVATE_KEY = "c2VjcmV0LXNlZWQtc2VjcmV0LXNlZWQtc2VjcmV0ISE="


@pytest.fixture(autouse=True)
def _no_validate_by_default():
    """Patch EvaluationClient.validate_api_key so tests never hit the network
    unless they opt in with validate=True + an explicit assertion."""
    with patch(
        "openbox_langchain.middleware_factory.EvaluationClient.validate_api_key",
        return_value=True,
    ) as mock_validate:
        yield mock_validate


def _import_factory():
    from openbox_langchain.middleware_factory import create_openbox_langchain_middleware

    return create_openbox_langchain_middleware


def test_factory_creates_middleware():
    """Factory creates a middleware instance."""
    create = _import_factory()
    mw = create(api_url=API_URL, api_key=API_KEY, validate=False)
    assert isinstance(mw, OpenBoxLangChainMiddleware)
    assert mw._runtime.config.sdk_version == SDK_PACKAGE_VERSION
    assert mw._runtime.config.sdk_engine == SDK_ENGINE
    assert mw._runtime.config.sdk_language == SDK_LANGUAGE


def test_factory_validates_api_key_by_default(_no_validate_by_default):
    """Factory calls EvaluationClient.validate_api_key when validate=True (default)."""
    create = _import_factory()
    create(api_url=API_URL, api_key=API_KEY)
    _no_validate_by_default.assert_called_once()


def test_factory_validation_client_uses_langchain_sdk_identifier():
    """Startup validation sends the LangChain SDK identifier, not base fallback."""
    create = _import_factory()
    with patch("openbox_langchain.middleware_factory.EvaluationClient") as client_cls:
        client_cls.return_value.validate_api_key.return_value = True
        create(api_url=API_URL, api_key=API_KEY)

    kwargs = client_cls.call_args.kwargs
    assert kwargs["sdk_version"] == SDK_PACKAGE_VERSION
    assert kwargs["sdk_engine"] == SDK_ENGINE
    assert kwargs["sdk_language"] == SDK_LANGUAGE


def test_factory_respects_validate_false(_no_validate_by_default):
    """Factory skips validation when validate=False."""
    create = _import_factory()
    create(api_url=API_URL, api_key=API_KEY, validate=False)
    _no_validate_by_default.assert_not_called()


def test_factory_forwards_agent_identity_to_runtime_options():
    """Factory forwards DID signing config through the base SDK config path."""
    create = _import_factory()
    mw = create(
        api_url=API_URL,
        api_key=API_KEY,
        agent_did=AGENT_DID,
        agent_private_key=AGENT_PRIVATE_KEY,
        validate=False,
    )
    try:
        assert mw._options.agent_did == AGENT_DID
        assert mw._options.agent_private_key == AGENT_PRIVATE_KEY
        assert mw._runtime.config.agent_did == AGENT_DID
        assert mw._runtime.config.agent_private_key == AGENT_PRIVATE_KEY
    finally:
        mw.close()


def test_factory_validation_client_receives_agent_identity():
    """Startup validation client receives a loaded base-SDK identity."""
    create = _import_factory()
    with patch("openbox_langchain.middleware_factory.EvaluationClient") as client_cls:
        client_cls.return_value.validate_api_key.return_value = True
        create(
            api_url=API_URL,
            api_key=API_KEY,
            agent_did=AGENT_DID,
            agent_private_key=AGENT_PRIVATE_KEY,
        )

    identity = client_cls.call_args.kwargs["identity"]
    assert identity is not None
    assert identity.agent_did == AGENT_DID


def test_factory_sets_agent_name():
    """Factory sets agent_name in options."""
    create = _import_factory()
    mw = create(api_url=API_URL, api_key=API_KEY, agent_name="MyAgent", validate=False)
    assert mw._options.agent_name == "MyAgent"


def test_factory_forwards_valid_kwargs():
    """Factory forwards valid kwargs to OpenBoxLangChainMiddlewareOptions."""
    create = _import_factory()
    mw = create(
        api_url=API_URL,
        api_key=API_KEY,
        validate=False,
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
    create = _import_factory()
    mw = create(
        api_url=API_URL,
        api_key=API_KEY,
        validate=False,
        agent_name="MyAgent",
        invalid_kwarg="should_be_filtered",
        another_invalid="also_filtered",
    )
    assert isinstance(mw, OpenBoxLangChainMiddleware)
    assert mw._options.agent_name == "MyAgent"


def test_factory_forwards_governance_timeout():
    """Factory forwards governance_timeout to options."""
    create = _import_factory()
    mw = create(api_url=API_URL, api_key=API_KEY, validate=False, governance_timeout=60.0)
    assert mw._options.governance_timeout == 60.0


def test_factory_sets_send_event_flags():
    """Factory forwards send_*_event flags without raising."""
    create = _import_factory()
    mw = create(
        api_url=API_URL,
        api_key=API_KEY,
        validate=False,
        send_chain_start_event=False,
        send_chain_end_event=False,
    )
    assert mw._options.send_chain_start_event is False
    assert mw._options.send_chain_end_event is False


def test_factory_raises_on_invalid_api_key_format():
    """Factory surfaces OpenBoxConfig's API-key format validation (fail fast)."""
    from openbox_core.errors import OpenBoxAuthError

    create = _import_factory()
    with pytest.raises(OpenBoxAuthError):
        create(api_url=API_URL, api_key="not-a-valid-key", validate=False)


def test_factory_env_prefix_resolution(monkeypatch):
    """OPENBOX_LANGCHAIN_* env vars are picked up when args are omitted for
    non-required fields (agent_name is env-resolvable via _ENV_FIELDS)."""
    monkeypatch.setenv("OPENBOX_LANGCHAIN_AGENT_NAME", "EnvAgent")
    create = _import_factory()
    mw = create(api_url=API_URL, api_key=API_KEY, validate=False)
    assert mw._options.agent_name == "EnvAgent"


def test_factory_validate_failure_propagates(_no_validate_by_default):
    """A validate_api_key failure propagates (fail fast, no swallowed error)."""
    from openbox_core.errors import OpenBoxAuthError

    _no_validate_by_default.side_effect = OpenBoxAuthError("invalid key")
    create = _import_factory()
    with pytest.raises(OpenBoxAuthError):
        create(api_url=API_URL, api_key=API_KEY)


def test_factory_closes_validation_client(_no_validate_by_default):
    """The validation EvaluationClient is closed after use (no leaked httpx client)."""
    with patch("openbox_langchain.middleware_factory.EvaluationClient.close") as mock_close:
        create = _import_factory()
        create(api_url=API_URL, api_key=API_KEY)
        mock_close.assert_called_once()


def test_factory_closes_validation_client_even_on_failure(_no_validate_by_default):
    """The validation client is closed even when validate_api_key raises."""
    _no_validate_by_default.side_effect = RuntimeError("boom")
    with patch("openbox_langchain.middleware_factory.EvaluationClient.close") as mock_close:
        create = _import_factory()
        with pytest.raises(RuntimeError):
            create(api_url=API_URL, api_key=API_KEY)
        mock_close.assert_called_once()


def test_factory_uses_magicmock_free_construction():
    """Sanity: constructing via the factory does not require any MagicMock
    patching beyond network validation (proves no leftover langgraph global-config
    coupling)."""
    create = _import_factory()
    mw = create(api_url=API_URL, api_key=API_KEY, validate=False)
    assert mw._runtime.config.api_url == API_URL
    assert mw._runtime.config.api_key == API_KEY
