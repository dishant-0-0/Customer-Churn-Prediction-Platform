"""
FastAPI application.
"""

from __future__ import annotations
from contextlib import asynccontextmanager
from fastapi import FastAPI
from src.api.dependencies import load_inference_artifacts
from src.api.routes import router
from src.utils.logger import get_logger
from src.config.config import settings
from src.api.exceptions import (
    register_exception_handlers,
)

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage application startup and shutdown.
    """

    logger.info("Starting Customer Churn API.")

    load_inference_artifacts()

    logger.info("Customer Churn API started successfully.")

    yield

    logger.info("Shutting down Customer Churn API.")

app = FastAPI(
    title= "Customer Churn Prediction API",
    description=(
        "REST API for predicting customer churn using the "
        "trained machine learning model."
    ),
    version= settings.artifacts.version,
    lifespan= lifespan
)
register_exception_handlers(app)
app.include_router(router)