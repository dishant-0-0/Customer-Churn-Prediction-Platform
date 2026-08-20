"""
Pydantic schemas for Customer Churn Prediction API.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from src.config.config import settings


class PredictionRequest(BaseModel):
    """
    Request body for churn prediction.
    """

    customer: dict[str, Any] = Field(
        ...,
        description="Raw customer features.",
        examples=[
            {
                "Gender": "Male",
                "Senior Citizen": "No",
                "Partner": "Yes",
                "Dependents": "No",
                "Tenure Months": 24,
                "Phone Service": "Yes",
                "Multiple Lines": "No",
                "Internet Service": "Fiber Optic",
                "Online Security": "No",
                "Online Backup": "Yes",
                "Device Protection": "No",
                "Tech Support": "No",
                "Streaming TV": "Yes",
                "Streaming Movies": "Yes",
                "Contract": "Month-to-month",
                "Paperless Billing": "Yes",
                "Payment Method": "Electronic check",
                "Monthly Charges": 79.85,
                "Total Charges": 1856.40,
                "CLTV": 3520,
                "City": "Los Angeles",
            }
        ],
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "customer": {
                    "Gender": "Male",
                    "Senior Citizen": "No",
                    "Partner": "Yes",
                    "Dependents": "No",
                    "Tenure Months": 24,
                    "Phone Service": "Yes",
                    "Multiple Lines": "No",
                    "Internet Service": "Fiber Optic",
                    "Online Security": "No",
                    "Online Backup": "Yes",
                    "Device Protection": "No",
                    "Tech Support": "No",
                    "Streaming TV": "Yes",
                    "Streaming Movies": "Yes",
                    "Contract": "Month-to-month",
                    "Paperless Billing": "Yes",
                    "Payment Method": "Electronic check",
                    "Monthly Charges": 79.85,
                    "Total Charges": 1856.40,
                    "CLTV": 3520,
                    "City": "Los Angeles",
                }
            }
        }
    )


class PredictionResponse(BaseModel):
    """
    Response returned by prediction endpoints.
    """

    prediction: int = Field(..., description="Prediction churn class.")

    probability: float = Field(
        ..., ge=0.0, le=1.0, description="Prediction probability of churn."
    )

    threshold: float = Field(
        default=settings.evaluation.threshold,
        description="Decision threshold used for prediciton.",
    )


class HealthResponse(BaseModel):
    """
    API health status.
    """

    status: str = "healthy"


class RootResponse(BaseModel):
    """
    Root endpoint response.
    """

    name: str
    version: str
    model: str


class ErrorResponse(BaseModel):
    """
    Standard API error response.
    """

    error: str = Field(..., description="Error type.")

    message: str = Field(..., description="Detailed error message")

    status_code: int = Field(..., description="HHTP status code.")
