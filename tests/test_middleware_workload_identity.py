"""Exercise IAM v3 through real base clients with an in-memory HTTP transport."""

from __future__ import annotations

import base64
import json
import os
from collections import Counter

import httpx
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519, rsa
from openbox_core.client import EvaluationClient
from openbox_core.errors import OpenBoxAuthError, OpenBoxConfigError, OpenBoxNetworkError

from openbox_langchain.middleware import (
    OpenBoxLangChainMiddleware,
    OpenBoxLangChainMiddlewareOptions,
)
from openbox_langchain.middleware_factory import create_openbox_langchain_middleware
from openbox_langchain.sdk_metadata import SDK_PACKAGE_VERSION
from tests.test_middleware_e2e import _build_agent, _final_text

API_URL = "https://core.example.test"
API_KEY = "obx_test_workload"
ISSUER = "https://identity.example.test/realms/openbox"
TOKEN_ENDPOINT = f"{ISSUER}/protocol/openid-connect/token"
WORKLOAD_HEADER = "X-OpenBox-Workload-Token"
AGENT_DID = "did:aip:12345678-1234-5678-1234-567812345678"


class WorkloadServer:
    def __init__(self):
        self.calls: list[httpx.Request] = []
        self.clients: list[EvaluationClient] = []
        self.bootstrap_status = 200
        self.token_status = 200
        self.runtime_status = 200
        self.identity_source = "openbox"

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.calls.append(request)
        if request.url.path == "/api/v3/auth/bootstrap":
            return httpx.Response(self.bootstrap_status, json={
                "bootstrap_version": 3,
                "contract_version": 3,
                "token_endpoint": TOKEN_ENDPOINT,
                "issuer": ISSUER,
                "audience": "openbox-core",
                "client_id": "openbox-agent-test",
                "service_account_id": "22222222-2222-4222-8222-222222222222",
                "activation_version": "33333333-3333-4333-8333-333333333333",
                "identity_source": self.identity_source,
                "kid": "workload-key-1",
            })
        if str(request.url) == TOKEN_ENDPOINT:
            return httpx.Response(self.token_status, json={
                "access_token": "test-workload-token",
                "token_type": "Bearer",
                "expires_in": 300,
            })
        return httpx.Response(self.runtime_status, json={"verdict": "ALLOW"})


@pytest.fixture(scope="module")
def workload_key():
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    return key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    ).decode()


@pytest.fixture
def server(monkeypatch):
    for name in os.environ:
        if name.startswith("OPENBOX_"):
            monkeypatch.delenv(name)
    # These tests cover auth transport wiring; the existing suite covers hooks.
    monkeypatch.setattr(
        "openbox_core.runtime.OpenBoxRuntime.install_instrumentation", lambda _: None,
    )
    server = WorkloadServer()

    def client_with_transport(*args, **kwargs):
        transport = httpx.MockTransport(server)
        client = EvaluationClient(*args, **kwargs, transport=transport, async_transport=transport)
        server.clients.append(client)
        return client

    monkeypatch.setattr(
        "openbox_langchain.middleware_factory.EvaluationClient", client_with_transport,
    )
    monkeypatch.setattr(
        "openbox_langchain.middleware_runtime_builder.EvaluationClient", client_with_transport,
    )
    return server


@pytest.fixture(scope="module")
def agent_private_key():
    key = ed25519.Ed25519PrivateKey.generate()
    raw_key = key.private_bytes(
        serialization.Encoding.Raw,
        serialization.PrivateFormat.Raw,
        serialization.NoEncryption(),
    )
    return base64.b64encode(raw_key).decode("ascii")


@pytest.mark.parametrize("entrypoint", ["factory", "options"])
@pytest.mark.parametrize("source", ["explicit", "sdk_env", "global_env", "okta_env"])
def test_workload_key_resolution_and_startup_validation(
    server, workload_key, monkeypatch, entrypoint, source,
):
    kwargs = {}
    if source == "explicit":
        kwargs["workload_private_key"] = workload_key
        monkeypatch.setenv("OPENBOX_LANGCHAIN_WORKLOAD_PRIVATE_KEY", "unused-sdk-key")
        monkeypatch.setenv("OPENBOX_WORKLOAD_PRIVATE_KEY", "unused-global-key")
    elif source == "sdk_env":
        monkeypatch.setenv("OPENBOX_LANGCHAIN_WORKLOAD_PRIVATE_KEY", workload_key)
        monkeypatch.setenv("OPENBOX_WORKLOAD_PRIVATE_KEY", "unused-global-key")
    elif source == "global_env":
        monkeypatch.setenv("OPENBOX_WORKLOAD_PRIVATE_KEY", workload_key)
    else:
        monkeypatch.setenv("OPENBOX_OKTA_AGENT_PRIVATE_KEY", workload_key)

    if entrypoint == "factory":
        middleware = create_openbox_langchain_middleware(
            api_url=API_URL, api_key=API_KEY, **kwargs,
        )
        assert server.clients[0]._sync_client is None  # startup client was closed
    else:
        middleware = OpenBoxLangChainMiddleware(OpenBoxLangChainMiddlewareOptions(
            api_url=API_URL, api_key=API_KEY, **kwargs,
        ))
        middleware._runtime.client.validate_api_key()

    try:
        assert middleware._runtime.config.keycloak_workload_private_key() == workload_key
        assert [request.url.path for request in server.calls] == [
            "/api/v3/auth/bootstrap", "/realms/openbox/protocol/openid-connect/token",
            "/api/v3/auth/validate",
        ]
        assert server.calls[-1].headers[WORKLOAD_HEADER] == "test-workload-token"
        assert middleware._runtime.adapter._poller._client is middleware._runtime.client
    finally:
        middleware.close()


def test_workload_key_is_not_in_options_repr(workload_key):
    options = OpenBoxLangChainMiddlewareOptions(workload_private_key=workload_key)
    assert workload_key not in repr(options)
    assert "PRIVATE KEY" not in repr(options)


@pytest.mark.parametrize("key", ["invalid-workload-key", "-----BEGIN PRIVATE KEY-----\nsecret"])
def test_malformed_workload_key_fails_before_network_without_disclosure(server, key):
    with pytest.raises(OpenBoxConfigError, match="Invalid workload_private_key") as error:
        create_openbox_langchain_middleware(
            api_url=API_URL, api_key=API_KEY, workload_private_key=key,
        )
    assert key not in str(error.value)
    assert server.calls == []


@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.parametrize("identity_source", ["openbox", "okta", "entra"])
async def test_agent_and_approval_polling_share_workload_auth(
    server, workload_key, mode, identity_source,
):
    server.identity_source = identity_source
    middleware = create_openbox_langchain_middleware(
        api_url=API_URL, api_key=API_KEY, workload_private_key=workload_key, validate=False,
    )
    assert server.calls == []
    client = middleware._runtime.client
    try:
        agent = _build_agent(middleware)
        state = {"messages": [("user", "hello")]}
        result = agent.invoke(state) if mode == "sync" else await agent.ainvoke(state)
        assert _final_text(result) == "final answer"
        poller = middleware._runtime.adapter._poller
        if mode == "sync":
            approval = poller.wait_for_decision("workflow-1", "run-1", "activity-1")
        else:
            approval = await poller.await_decision("workflow-1", "run-1", "activity-1")
        assert approval.allow_shaped

        counts = Counter(str(request.url) for request in server.calls)
        assert counts[f"{API_URL}/api/v3/auth/bootstrap"] == 1
        assert counts[TOKEN_ENDPOINT] == 1
        requests = server.calls[2:]
        assert {request.url.path for request in requests} == {
            "/api/v3/governance/evaluate", "/api/v3/governance/approval",
        }
        for request in requests:
            assert request.headers["Authorization"] == f"Bearer {API_KEY}"
            assert request.headers[WORKLOAD_HEADER] == "test-workload-token"
            assert request.headers["X-OpenBox-SDK-Version"] == (
                f"openbox-langchain-python-v{SDK_PACKAGE_VERSION}"
            )
            assert workload_key.encode() not in request.content
        assert json.loads(requests[-1].content) == {
            "workflow_id": "workflow-1", "run_id": "run-1", "activity_id": "activity-1",
        }
        http_client = client._sync_client if mode == "sync" else client._async_client
        assert not http_client.is_closed
    finally:
        if mode == "sync":
            middleware.close()
        else:
            await middleware.aclose()
    assert http_client.is_closed


@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.parametrize("status", [401, 403])
async def test_auth_rejection_stops_agent_and_approval_under_fail_open(
    server, workload_key, mode, status,
):
    server.runtime_status = status
    middleware = create_openbox_langchain_middleware(
        api_url=API_URL, api_key=API_KEY, workload_private_key=workload_key,
        validate=False, on_api_error="fail_open",
    )
    try:
        agent = _build_agent(middleware)
        with pytest.raises(OpenBoxAuthError):
            if mode == "sync":
                agent.invoke({"messages": [("user", "hello")]})
            else:
                await agent.ainvoke({"messages": [("user", "hello")]})
        poller = middleware._runtime.adapter._poller
        with pytest.raises(OpenBoxAuthError):
            if mode == "sync":
                poller.wait_for_decision("workflow-1", "run-1", "activity-1")
            else:
                await poller.await_decision("workflow-1", "run-1", "activity-1")
        assert all(
            request.url.path.startswith("/api/v3/") or str(request.url) == TOKEN_ENDPOINT
            for request in server.calls
        )
    finally:
        await middleware.aclose()


@pytest.mark.parametrize(
    ("failure", "error_type"),
    [("bootstrap_status", OpenBoxNetworkError), ("token_status", OpenBoxAuthError)],
)
def test_startup_auth_failure_does_not_downgrade(server, workload_key, failure, error_type):
    setattr(server, failure, 503 if failure == "bootstrap_status" else 400)
    with pytest.raises(error_type):
        create_openbox_langchain_middleware(
            api_url=API_URL, api_key=API_KEY, workload_private_key=workload_key,
        )
    assert server.clients[0]._sync_client is None
    assert all(
        request.url.path == "/api/v3/auth/bootstrap" or str(request.url) == TOKEN_ENDPOINT
        for request in server.calls
    )


@pytest.mark.parametrize("signed", [False, True])
def test_existing_v1_identity_routing_is_preserved(server, signed, agent_private_key):
    kwargs = {"agent_did": AGENT_DID, "agent_private_key": agent_private_key} if signed else {}
    middleware = create_openbox_langchain_middleware(
        api_url=API_URL, api_key=API_KEY, **kwargs,
    )
    try:
        agent = _build_agent(middleware)
        assert _final_text(agent.invoke({"messages": [("user", "hello")]})) == "final answer"
        assert {request.url.path for request in server.calls} == {
            "/api/v1/auth/validate", "/api/v1/governance/evaluate",
        }
        assert all(WORKLOAD_HEADER not in request.headers for request in server.calls)
    finally:
        middleware.close()
