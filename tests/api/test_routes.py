"""
Tests for the FastAPI application.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
from fastapi.testclient import TestClient

from src.api.app import app
from src.config.config import settings
from src.core import PredictionResult


client = TestClient(app)


@patch("src.api.app.load_inference_artifacts")
def test_root(
    mock_load_artifacts,
):
    """
    Root endpoint should return application metadata.
    """

    response = client.get("/api/v1/")

    assert response.status_code == 200

    body = response.json()

    assert body["name"] == "Customer Churn Prediction API"
    assert body["version"] == settings.artifacts.version
    assert body["model"] == settings.training.model.name


@patch("src.api.app.load_inference_artifacts")
def test_health(
    mock_load_artifacts,
):
    """
    Health endpoint should report healthy.
    """

    response = client.get("/api/v1/health")

    assert response.status_code == 200

    assert response.json() == {
        "status": "healthy",
    }


@patch("src.api.routes.run_inference_pipeline")
@patch("src.api.routes.get_inference_artifacts")
@patch("src.api.app.load_inference_artifacts")
def test_predict(
    mock_load_artifacts,
    mock_get_artifacts,
    mock_run_pipeline,
    mock_inference_artifacts,
):
    """
    Prediction endpoint should return prediction result.
    """

    mock_get_artifacts.return_value = (
        mock_inference_artifacts
    )

    mock_run_pipeline.return_value = PredictionResult(
        predictions=np.array([1]),
        probabilities=np.array([0.87]),
    )

    payload = {
        "customer": {
            "Gender": "Male",
            "Senior Citizen": 0,
            "Partner": "Yes",
            "Dependents": "No",
            "Tenure Months": 12,
            "Phone Service": "Yes",
            "Multiple Lines": "No",
            "Internet Service": "Fiber optic",
            "Online Security": "No",
            "Online Backup": "Yes",
            "Device Protection": "No",
            "Tech Support": "No",
            "Streaming TV": "Yes",
            "Streaming Movies": "Yes",
            "Contract": "Month-to-month",
            "Paperless Billing": "Yes",
            "Payment Method": "Electronic check",
            "Monthly Charges": 79.5,
            "Total Charges": 980.0,
            "CLTV": 4500,
        }
    }

    response = client.post(
        "/api/v1/predict",
        json=payload,
    )

    assert response.status_code == 200

    body = response.json()

    assert body == {
        "prediction": 1,
        "probability": 0.87,
        "threshold": settings.evaluation.threshold,
    }

    mock_get_artifacts.assert_called_once()

    mock_run_pipeline.assert_called_once()


@patch("src.api.app.load_inference_artifacts")
def test_predict_validation_error(
    mock_load_artifacts,
):
    """
    Invalid request should return HTTP 422.
    """

    response = client.post(
        "/api/v1/predict",
        json={},
    )

    assert response.status_code == 422

