import app as app_module

client = app_module.app.test_client()


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.get_json()["status"] == "ok"


def test_home_page():
    response = client.get("/")
    assert response.status_code == 200


def test_invalid_json():
    response = client.post(
        "/predict",
        data="not-json",
        content_type="application/json"
    )
    assert response.status_code == 400


def test_missing_text():
    response = client.post("/predict", json={})
    assert response.status_code == 400


def test_empty_text():
    response = client.post("/predict", json={"text": "   "})
    assert response.status_code == 400


def test_text_too_long():
    response = client.post("/predict", json={"text": "news " * 4001})
    assert response.status_code == 400


def test_valid_prediction():
    response = client.post(
        "/predict",
        json={
            "text": (
                "The Election Commission of India conducts elections "
                "and announces official results through its authorized channels."
            )
        }
    )
    assert response.status_code == 200

    result = response.get_json()
    assert result["label"] in ("REAL", "FAKE")
    assert "confidence_like_score" in result
    assert "score_note" in result
