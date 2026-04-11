"""
test_api.py — Integration tests for the Spam Classifier FastAPI endpoints.

Covers every endpoint defined in app.py:
  POST /predict
  GET  /predictions
  GET  /predictions/stats
  GET  /health
"""

import pytest


# ──────────────────────────────────────────────────────────────────────────────
# /health
# ──────────────────────────────────────────────────────────────────────────────

class TestHealth:
    def test_health_returns_200(self, client):
        r = client.get("/health")
        assert r.status_code == 200

    def test_health_body(self, client):
        data = client.get("/health").json()
        assert "status" in data
        assert data["status"] == "ok"

    def test_model_loaded(self, client):
        data = client.get("/health").json()
        assert data["model_loaded"] is True
        assert data["vectorizer_loaded"] is True


# ──────────────────────────────────────────────────────────────────────────────
# POST /predict
# ──────────────────────────────────────────────────────────────────────────────

class TestPredict:
    def test_spam_message(self, client):
        r = client.post("/predict", json={"text": "WINNER!! You have won a free prize! Call now!"})
        assert r.status_code == 200
        data = r.json()
        assert data["prediction"] == "spam"
        assert 0 < data["confidence"] <= 100
        assert 0.0 <= data["spam_probability"] <= 1.0
        assert 0.0 <= data["ham_probability"] <= 1.0

    def test_ham_message(self, client):
        r = client.post("/predict", json={"text": "Hey, are we still meeting for lunch tomorrow?"})
        assert r.status_code == 200
        data = r.json()
        assert data["prediction"] == "ham"

    def test_response_contains_all_fields(self, client):
        r = client.post("/predict", json={"text": "Test message"})
        assert r.status_code == 200
        data = r.json()
        for field in ("id", "input_text", "prediction", "confidence",
                      "ham_probability", "spam_probability"):
            assert field in data, f"Missing field: {field}"

    def test_probabilities_sum_to_one(self, client):
        r = client.post("/predict", json={"text": "Congratulations! Free iPhone!"})
        data = r.json()
        total = round(data["ham_probability"] + data["spam_probability"], 4)
        assert total == 1.0

    def test_empty_text_returns_400(self, client):
        r = client.post("/predict", json={"text": "   "})
        assert r.status_code == 400

    def test_missing_text_field_returns_422(self, client):
        r = client.post("/predict", json={})
        assert r.status_code == 422

    def test_prediction_saved_to_db(self, client):
        text = "Unique test message for DB check"
        client.post("/predict", json={"text": text})

        history = client.get("/predictions?limit=10").json()
        texts = [r["input_text"] for r in history]
        assert text in texts


# ──────────────────────────────────────────────────────────────────────────────
# GET /predictions
# ──────────────────────────────────────────────────────────────────────────────

class TestPredictions:
    def test_returns_list(self, client):
        r = client.get("/predictions")
        assert r.status_code == 200
        assert isinstance(r.json(), list)

    def test_filter_spam(self, client):
        # Seed a spam prediction
        client.post("/predict", json={"text": "Win a FREE iPhone now!!! Limited offer!!!"})

        r = client.get("/predictions?label=spam")
        assert r.status_code == 200
        records = r.json()
        for record in records:
            assert record["prediction"] == "spam"

    def test_filter_ham(self, client):
        client.post("/predict", json={"text": "I'll see you at the office tomorrow morning"})

        r = client.get("/predictions?label=ham")
        assert r.status_code == 200
        records = r.json()
        for record in records:
            assert record["prediction"] == "ham"

    def test_invalid_label_returns_400(self, client):
        r = client.get("/predictions?label=unknown")
        assert r.status_code == 400

    def test_limit_param(self, client):
        # Seed 3 predictions
        for i in range(3):
            client.post("/predict", json={"text": f"Test message number {i}"})

        r = client.get("/predictions?limit=2")
        assert r.status_code == 200
        assert len(r.json()) <= 2

    def test_record_has_required_fields(self, client):
        client.post("/predict", json={"text": "Field check message"})
        records = client.get("/predictions?limit=1").json()
        assert len(records) >= 1
        for field in ("id", "input_text", "prediction", "confidence",
                      "spam_probability", "ham_probability", "created_at"):
            assert field in records[0], f"Missing field: {field}"


# ──────────────────────────────────────────────────────────────────────────────
# GET /predictions/stats
# ──────────────────────────────────────────────────────────────────────────────

class TestStats:
    def test_returns_200(self, client):
        assert client.get("/predictions/stats").status_code == 200

    def test_stats_structure(self, client):
        data = client.get("/predictions/stats").json()
        for key in ("total_predictions", "spam_count", "ham_count"):
            assert key in data

    def test_counts_are_consistent(self, client):
        data = client.get("/predictions/stats").json()
        assert data["spam_count"] + data["ham_count"] == data["total_predictions"]

    def test_counts_increment_after_predict(self, client):
        before = client.get("/predictions/stats").json()["total_predictions"]
        client.post("/predict", json={"text": "Stats increment check"})
        after = client.get("/predictions/stats").json()["total_predictions"]
        assert after == before + 1
