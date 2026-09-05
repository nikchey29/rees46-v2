from fastapi.testclient import TestClient

from rees46.serving.api import create_app


def test_health_without_model_is_degraded() -> None:
    client = TestClient(create_app(bundle=None))
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] in {"ok", "degraded"}


def test_recommend_without_model_returns_503_when_no_local_bundle() -> None:
    client = TestClient(create_app(bundle=None))
    response = client.get("/recommend/1")
    if response.status_code != 200:
        assert response.status_code == 503
