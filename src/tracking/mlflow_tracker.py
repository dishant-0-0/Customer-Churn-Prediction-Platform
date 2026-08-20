"""
MLflow experiment tracking.
"""

from __future__ import annotations
from pathlib import Path
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from xgboost import XGBClassifier
from src.config.config import settings
from src.core import TrainingResult
from src.utils.logger import get_logger
from src.config.paths import PROJECT_ROOT

logger = get_logger(__name__)


def _log_parameters(
    training_result: TrainingResult
) -> None:
    """
    Log experiment parameters.
    """

    logger.info("Logging parameters.")

    params = {
        "threshold": settings.evaluation.threshold,
        "random_state": settings.data.random_state,
        "test_size": settings.data.test_size,
        "target": settings.data.target,
    }

    params.update(training_result.model.get_params())

    mlflow.log_params(params)


def _log_metrics(
    training_result: TrainingResult
) -> None:
    """
    Log evaluation metrics.
    """

    logger.info("Logging metrics.")

    evaluation = training_result.evaluation

    mlflow.log_metrics(
        {
            "accuracy": evaluation.accuracy,
            "precision": evaluation.precision,
            "recall": evaluation.recall,
            "f1": evaluation.f1,
            "roc_auc": evaluation.roc_auc,
            "average_precision": evaluation.average_precision
        }
    )


def _log_model(
    training_result: TrainingResult,
) -> None:
    """
    Log trained model.
    """

    logger.info("Logging model.")

    if isinstance(training_result.model, XGBClassifier):
        mlflow.xgboost.log_model(
            xgb_model=training_result.model,
            artifact_path="model",
        )
        return

    mlflow.sklearn.log_model(
        sk_model= training_result.model,
        name= "model"
    )


def _log_artifacts(
    training_result: TrainingResult
) -> None:
    """
    Log experiment artifacts.
    """

    logger.info("Logging artifacts.")

    artifact_root = training_result.artifacts_path

    for directory in (
        "metrics",
        "figures",
        "reports"
    ):
        mlflow.log_artifact(
            artifact_root / directory,
            artifact_path= directory,
        )


def _get_tracking_uri() -> str:
    """
    Resolve the configured MLflow tracking URI.
    """

    uri = settings.tracking.tracking_uri

    if uri.startswith("sqlite:///"):
        database = uri.replace("sqlite:///", "")
        database_path = PROJECT_ROOT / database
        return f"sqlite:///{database_path.as_posix()}"

    return uri


def _set_tags() -> None:
    """
    Set MLflow run tags.
    """

    logger.info("Setting MLflow tags.")

    mlflow.set_tags(
        {
            "project": "Customer Chrun Prediction",
            "model": settings.training.model.name,
            "version": settings.artifacts.version,
            "author": "Dishant Patel"
        }
    )


def log_experiment(
    training_result: TrainingResult
) -> None:
    """
    Log a training run to MLflow.
    """

    if not settings.tracking.enabled:
        logger.info("MLflow tracking is disabled.")
        return

    tracking_uri = _get_tracking_uri()

    logger.info("Starting MLflow experiment logging.")

    mlflow.set_tracking_uri(tracking_uri)

    mlflow.set_experiment(
        settings.tracking.experiment_name
    )

    logger.info(
        f"MLflow experiment: '{settings.tracking.experiment_name}'"
    )
    logger.info(
        f"Tracking URI: {tracking_uri}"
    )

    with mlflow.start_run(
        run_name=f"{settings.training.model.name}_{settings.artifacts.version}"
    ):

        _log_parameters(training_result)

        _log_metrics(training_result)

        _log_model(training_result)

        _log_artifacts(training_result)

    logger.info("MLflow logging completed successfully.")