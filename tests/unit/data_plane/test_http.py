from hashlib import sha256

import httpx
import pytest

from portable_batch_execution.contracts import ArtifactRef
from portable_batch_execution.data_plane import (
    HttpPrivateDataPlane,
    PrivateDataPlaneError,
)


def test_http_plane_rejects_non_https_base_url_without_leaking_secrets():
    token = "SENTINEL_TOKEN_DO_NOT_LEAK"
    base_url = "http://plane.example/private-path"
    with pytest.raises(ValueError, match="HTTPS origin URL") as error:
        HttpPrivateDataPlane(base_url, token)
    message = str(error.value)
    assert token not in message
    assert "plane.example" not in message
    assert "http://" not in message


def test_http_plane_resolves_exact_opaque_pair_and_uses_bearer(monkeypatch):
    seen = {}
    def handler(request):
        seen["path"] = request.url.path
        seen["auth"] = request.headers["authorization"]
        return httpx.Response(200, json={"job": {}, "wave": {}, "shards": []})
    monkeypatch.setenv("PBE_PRIVATE_DATA_PLANE_BASE_URL", "https://plane.example")
    monkeypatch.setenv("PBE_PRIVATE_DATA_PLANE_BEARER_TOKEN", "not-for-output")
    plane = HttpPrivateDataPlane.from_environment()
    plane._client = httpx.Client(transport=httpx.MockTransport(handler), headers={"Authorization": "Bearer not-for-output"})
    assert plane.resolve_wave("run-opaque", "wave-opaque")["shards"] == []
    assert seen == {"path": "/v1/runs/run-opaque/waves/wave-opaque", "auth": "Bearer not-for-output"}


def test_http_plane_never_leaks_response_body_or_token_in_error():
    plane = HttpPrivateDataPlane("https://plane.example", "super-secret", client=httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(403, text="private body super-secret"))))
    with pytest.raises(PrivateDataPlaneError) as error:
        plane.resolve_wave("run", "wave")
    assert str(error.value) == "private data plane resolve wave failed with HTTP 403"
    assert "secret" not in str(error.value)
    with pytest.raises(ValueError):
        plane.resolve_wave("../path", "wave")


def test_http_plane_reads_by_object_id_not_artifact_uri():
    data = b"[]"
    ref = ArtifactRef(object_id="opaque-object", uri="pbe://private/opaque-object", sha256="sha256:" + sha256(data).hexdigest(), size_bytes=len(data))
    seen = []
    plane = HttpPrivateDataPlane("https://plane.example", "token", client=httpx.Client(transport=httpx.MockTransport(lambda request: (seen.append(request.url.path), httpx.Response(200, content=data))[1])))
    assert plane.read(ref) == data
    assert seen == ["/v1/artifacts/opaque-object/content"]

def test_http_plane_uses_explicit_finite_timeout_policy():
    plane = HttpPrivateDataPlane("https://plane.example", "token")
    try:
        timeout = plane._client.timeout
        assert isinstance(timeout, httpx.Timeout)
        assert (timeout.connect, timeout.read, timeout.write, timeout.pool) == (
            10.0,
            60.0,
            30.0,
            10.0,
        )
        assert all(
            value is not None and value > 0
            for value in (timeout.connect, timeout.read, timeout.write, timeout.pool)
        )
    finally:
        plane._client.close()


def test_http_plane_timeout_policy_applies_to_every_call():
    seen = []

    def handler(request):
        seen.append(request.extensions.get("timeout"))
        return httpx.Response(200, json={"job": {}, "wave": {}, "shards": []})

    plane = HttpPrivateDataPlane("https://plane.example", "token")
    plane._client._transport = httpx.MockTransport(handler)
    try:
        plane.resolve_wave("run", "wave")
    finally:
        plane._client.close()
    assert seen == [{"connect": 10.0, "read": 60.0, "write": 30.0, "pool": 10.0}]


def test_http_plane_maps_read_timeouts_to_sanitized_error():
    def handler(request):
        raise httpx.ReadTimeout("private body super-secret")

    plane = HttpPrivateDataPlane(
        "https://plane.example",
        "super-secret",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    ref = ArtifactRef(
        object_id="opaque-object",
        uri="pbe://private/opaque-object",
        sha256="sha256:" + sha256(b"").hexdigest(),
    )
    with pytest.raises(PrivateDataPlaneError) as error:
        plane.read(ref)
    assert str(error.value) == "private data plane read artifact failed"


def test_http_plane_exists_and_manifest_keep_error_mapping():
    plane = HttpPrivateDataPlane("https://plane.example", "token")
    responses = {
        "HEAD": httpx.Response(404),
        "GET": httpx.Response(200, json={"logical_run_id": "run", "revision": 0}),
    }
    plane._client = httpx.Client(
        transport=httpx.MockTransport(lambda request: responses[request.method])
    )
    ref = ArtifactRef(
        object_id="opaque-object",
        uri="pbe://private/opaque-object",
        sha256="sha256:" + sha256(b"").hexdigest(),
    )
    assert plane.exists(ref) is False
    with pytest.raises(PrivateDataPlaneError):
        plane.read_manifest("run")

