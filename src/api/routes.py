"""

"""

from __future__ import annotations
import pandas as pd
from fastapi import APIRouter
from src.api.dependencies import get_inference_artifacts
from src.api.schemas import (
    HealthResponse,
    PredictionRequest,
    PredictionResponse,
    RootResponse
)
from src.config.config import settings
from src.pipelines.inference_pipeline import run_inference_pipeline
from src.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(
    prefix= "/api/v1",
    tags= ["Customer Churn"]
)

@router.get( "/", response_model=RootResponse)
async def root() -> RootResponse:
    """
    Root endpoint.
    """

    return RootResponse(
        name = "Customer Churn Prediction API",
        version= settings.artifacts.version,
        model= settings.training.model.name
    )

@router.get( "/health", response_model= HealthResponse)
async def health() -> HealthResponse:
    """
    Health check endpoint.
    """

    return HealthResponse()

@router.post("/predict", response_model= PredictionResponse)
async def predict(
    request: PredictionRequest
) -> PredictionResponse:
    """
    Predict customer churn.
    """

    logger.info("Received prediction request.")

    artifacts = get_inference_artifacts()

    customer = pd.DataFrame(
        [request.customer]
    )

    result = run_inference_pipeline(
        X = customer,
        artifacts= artifacts
    )

    logger.info("Prediction completed successfully.")

    return PredictionResponse(
        prediction= int(result.predictions[0]),
        probability= float(result.probabilities[0]),
        threshold= settings.evaluation.threshold
    )