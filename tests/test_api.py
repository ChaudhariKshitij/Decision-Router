import pytest
from fastapi.testclient import TestClient

from app.api import create_app
from tests.conftest import TXN


@pytest.fixture()
def client(router):
    with TestClient(create_app(router)) as c:
        yield c


def test_health_and_docs(client):
    h = client.get("/health").json()
    assert h["status"] == "ok" and h["models_loaded"] and "api_key" not in str(h).lower().replace("api_key_configured", "")
    assert client.get("/openapi.json").status_code == 200


def test_route_ok_and_audit(client):
    r = client.post("/route?policy=risk_aware", json=TXN)
    assert r.status_code == 200
    body = r.json()
    assert body["decision"] in {"FRAUD", "LEGIT", "REVIEW"} and body["selected_model"] in {"fast", "accurate", "deep"}
    assert client.get(f"/audit/{body['request_id']}").json()["request_id"] == body["request_id"]
    assert client.get("/audit/doesnotexist").status_code == 404


@pytest.mark.parametrize("patch", [{"hour": 24}, {"amount": -1}, {"international": 2}, {"avg_amount_30d": 0}, {"device_seen_before": "x"}])
def test_invalid_input_rejected(client, patch):
    assert client.post("/route", json={**TXN, **patch}).status_code == 422


def test_missing_field_and_bad_policy(client):
    bad = dict(TXN); bad.pop("amount")
    assert client.post("/route", json=bad).status_code == 422
    assert client.post("/route?policy=magic", json=TXN).status_code == 422


def test_jev_policy_falls_back_without_key(client):
    b = client.post("/route?policy=jev", json=TXN).json()
    assert b["policy_source"] == "local_fallback" and b["fallback_used"] is True


def test_batch_and_metrics(client):
    r = client.post("/route/batch", json={"transactions": [TXN, TXN]})
    assert r.status_code == 200
    j = r.json()
    assert j["total_transactions"] == 2 and abs(j["fast_pct"] + j["accurate_pct"] + j["deep_pct"] - 100) < 1e-6
    assert client.get("/metrics").json()["transactions_routed"] >= 2
    assert client.post("/route/batch", json={"transactions": []}).status_code == 422


def test_shadow_and_model_info(client):
    s = client.post("/shadow-route", json=TXN).json()
    assert s["agreement"] is None and s["jev_status"] == "disabled"
    mi = client.get("/model-info").json()
    assert {m["name"] for m in mi["models"]} == {"fast", "accurate", "deep"}


def test_no_models_gives_503(tmp_path, monkeypatch):
    monkeypatch.setenv("DR_ARTIFACTS_DIR", str(tmp_path))
    with TestClient(create_app()) as c:
        assert c.get("/health").json()["models_loaded"] is False
        assert c.post("/route", json=TXN).status_code == 503
