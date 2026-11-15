"""Exercise gateway failures without model traffic or physical evaluation."""
from contextlib import contextmanager
import errno
from http.client import HTTPConnection, IncompleteRead
from http.server import ThreadingHTTPServer
import io
import json
import socket
import ssl
from threading import Thread
import urllib.error

import pytest

from contest2 import model_gateway
from contest2.budget import BudgetLedger


@contextmanager
def gateway_session(tmp_path, monkeypatch, provider):
    ledger = BudgetLedger(tmp_path / "budget.sqlite", task_id="binary_loss",
                          model_cost_limit_rmb="15", allowed_model="deepseek-flash")
    monkeypatch.setattr(model_gateway.urllib.request, "urlopen", provider)
    monkeypatch.setattr(model_gateway, "peak_during", lambda *args: False)
    server = ThreadingHTTPServer(("127.0.0.1", 0), model_gateway.make_handler(
        ledger, access_token="gateway-secret", provider_key="provider-secret"))
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    def post():
        client = HTTPConnection("127.0.0.1", server.server_port, timeout=5)
        try:
            client.request("POST", "/v1/responses", json.dumps({"model": "deepseek-flash", "input": "test"}),
                           {"Authorization": "Bearer gateway-secret", "Content-Type": "application/json"})
            response = client.getresponse()
            return response.status, json.loads(response.read())
        finally:
            client.close()
    try:
        yield ledger, post
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def response(value):
    result = io.BytesIO(json.dumps(value).encode())
    result.headers = {"Content-Type": "application/json"}
    return result


@pytest.mark.parametrize("cause", [
    ConnectionRefusedError(errno.ECONNREFUSED, "Connection refused"),
    socket.gaierror(socket.EAI_AGAIN, "DNS lookup failed"),
    OSError(errno.ENETUNREACH, "Network unreachable"),
    ssl.SSLCertVerificationError(1, "Certificate verification failed"),
])
def test_connection_failures_release_reservations_and_next_call_succeeds(tmp_path, monkeypatch, cause):
    attempts = []
    usage = {"input_tokens": 100, "input_tokens_details": {"cached_tokens": 20},
             "output_tokens": 30, "total_tokens": 130}
    def provider(*args, **kwargs):
        # Even failed calls must be admitted before trying the provider.
        assert ledger.snapshot()["model_cost_reserved_rmb"] == "2.228224"
        attempts.append(1)
        if len(attempts) <= 7:
            raise urllib.error.URLError(cause)
        return response({"model": "deepseek-flash", "usage": usage})
    with gateway_session(tmp_path, monkeypatch, provider) as (ledger, post):
        for _ in range(7):
            status, value = post()
            assert status == 502 and value["details"]["request_not_sent"]
            state = ledger.snapshot()
            assert state["model_cost_reserved_rmb"] == "0"
            assert state["model_cost_spent_rmb"] == "0"
            assert state["model_dispatch_allowed"]
        assert post()[0] == 200
        state = ledger.snapshot()
        assert state["model_cost_spent_rmb"] == "0.0002004"
        assert state["unresolved_model_calls"] == 0
        assert state["evaluations_accepted"] == 0
        assert len(list((tmp_path / "model-receipts").glob("*.error.json"))) == 7


@pytest.mark.parametrize("error", [
    urllib.error.URLError(TimeoutError("Response timeout")),
    urllib.error.URLError(ConnectionResetError("Connection reset")),
    IncompleteRead(b"partial reply", 100),
    urllib.error.HTTPError("https://api.deepseek.com/responses", 429, "Rate limited", {}, io.BytesIO(b"busy")),
])
def test_ambiguous_failures_keep_reservations_and_original_reason(tmp_path, monkeypatch, error):
    def provider(*args, **kwargs):
        raise error
    with gateway_session(tmp_path, monkeypatch, provider) as (ledger, post):
        status, value = post()
        assert status == 502 and not value["details"]["request_not_sent"]
        state = ledger.snapshot()
        assert state["model_cost_reserved_rmb"] == "2.228224"
        assert state["unresolved_model_calls"] == 1
        evidence = json.loads((tmp_path / "model-receipts" / (value["request_id"] + ".error.json")).read_text())
        assert evidence["error_type"] == type(error).__name__
        assert evidence["message"]


def test_http_error_diagnostics_preserve_status_and_redact_credentials(tmp_path, monkeypatch):
    def provider(*args, **kwargs):
        raise urllib.error.HTTPError("https://api.deepseek.com/responses", 400, "Bad request", {},
            io.BytesIO(b'invalid request provider-secret gateway-secret Bearer token-value sk-another-secret'))
    with gateway_session(tmp_path, monkeypatch, provider) as (ledger, post):
        status, value = post()
        assert status == 502 and value["details"]["http_status"] == 400
        assert "invalid request" in value["details"]["message"]
        log = (tmp_path / "model-receipts" / (value["request_id"] + ".error.json")).read_text()
        for secret in ["provider-secret", "gateway-secret", "token-value", "sk-another-secret"]:
            assert secret not in log and secret not in json.dumps(value)
        assert ledger.snapshot()["unresolved_model_calls"] == 1


def test_missing_usage_is_diagnosable_and_never_refunded(tmp_path, monkeypatch):
    with gateway_session(tmp_path, monkeypatch, lambda *a, **k: response({"model": "deepseek-flash"})) as (ledger, post):
        status, value = post()
        assert status == 502
        assert "Provider usage must be an object" in value["details"]["message"]
        assert ledger.snapshot()["model_cost_reserved_rmb"] == "2.228224"
