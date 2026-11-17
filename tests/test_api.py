import pytest
from pydantic import SecretStr
from fastapi.testclient import TestClient
from app.api.main import create_app
from app.config import get_settings


@pytest.fixture
def client(service):
    return TestClient(create_app(service))


def test_health(client):
    assert client.get("/health").json()["status"] == "ok"


def test_resolve_returns_reply_citations_and_metrics(client):
    r = client.post("/v1/resolve", json={"ticket": "I keep getting error E-4102"})
    body = r.json()
    assert r.status_code == 200 and body["outcome"] == "auto_send"
    assert body["citations"][0]["source"] == "sync_troubleshooting.md"
    assert body["metrics"]["llm_calls"] >= 3


def test_run_trace_is_retrievable(client):
    run_id = client.post("/v1/resolve", json={"ticket": "How do I reset 2FA?"}).json()["run_id"]
    spans = client.get(f"/v1/runs/{run_id}").json()["spans"]
    assert any(sp["name"] == "reflect" for sp in spans)
    assert client.get("/v1/runs/nope").status_code == 404


def test_validation_rejects_bad_input(client):
    assert client.post("/v1/resolve", json={"ticket": ""}).status_code == 422
    assert client.post("/v1/resolve", json={"ticket": "hi", "customer_id": "x; DROP TABLE"}).status_code == 422
    assert client.post("/v1/resolve", json={"ticket": "a" * 5000}).status_code == 422


def test_api_key_is_enforced_when_configured(client, monkeypatch):
    monkeypatch.setattr(get_settings(), "api_key", SecretStr("secret"))
    assert client.post("/v1/resolve", json={"ticket": "hi"}).status_code == 401
    ok = client.post("/v1/resolve", json={"ticket": "How do I reset 2FA?"}, headers={"X-API-Key": "secret"})
    assert ok.status_code == 200


def test_rate_limit(service, monkeypatch):
    monkeypatch.setattr(get_settings(), "rate_limit_per_min", 2)
    c = TestClient(create_app(service))
    codes = [c.post("/v1/resolve", json={"ticket": "How do I reset 2FA?"}).status_code for _ in range(3)]
    assert codes == [200, 200, 429]


def test_metrics_endpoint(client):
    client.post("/v1/resolve", json={"ticket": "How do I reset 2FA?"})
    assert client.get("/v1/metrics").json()["requests"] >= 1


def test_production_startup_path_ingests_and_serves(settings, tmp_path, monkeypatch):
    """No injected service: the app must build the store, seed orders and ingest data/ by itself."""
    monkeypatch.setattr(settings, "chroma_dir", str(tmp_path / "chroma"))
    monkeypatch.setattr(settings, "trace_file", str(tmp_path / "traces.jsonl"))
    with TestClient(create_app()) as c:
        assert c.get("/health").json()["ready"] is True
        r = c.post("/v1/resolve", json={"ticket": "Where is my order ORD-1001?", "customer_id": "CUST-1"})
        assert r.status_code == 200 and "TRK-88213" in r.json()["reply"]


def test_secrets_are_not_printed(settings):
    from pydantic import SecretStr
    settings.gemini_api_key = SecretStr("fake-secret-value")
    assert "fake-secret" not in repr(settings) and "fake-secret" not in str(settings)
