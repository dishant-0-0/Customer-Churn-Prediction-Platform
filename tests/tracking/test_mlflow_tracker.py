"""
Tests for MLflow experiment tracking.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from xgboost import XGBClassifier

from src.core import TrainingResult
from src.tracking import mlflow_tracker


def test_get_tracking_uri_sqlite():
    """
    SQLite URI should be converted to an absolute path.
    """

    uri = mlflow_tracker._get_tracking_uri()

    expected = (
        f"sqlite:///{(mlflow_tracker.PROJECT_ROOT / 'mlflow.db').as_posix()}"
    )

    assert uri == expected


@patch("src.tracking.mlflow_tracker.mlflow.log_params")
def test_log_parameters(
    mock_log_params,
    mock_training_result,
):
    """
    Parameters should be logged.
    """

    mlflow_tracker._log_parameters(
        mock_training_result,
    )

    mock_log_params.assert_called_once()

    params = mock_log_params.call_args.args[0]

    assert params["threshold"] == (
        mlflow_tracker.settings.evaluation.threshold
    )

    assert params["target"] == (mlflow_tracker.settings.data.target)


@patch("src.tracking.mlflow_tracker.mlflow.log_metrics")
def test_log_metrics(
    mock_log_metrics,
    mock_training_result,
):
    """
    Metrics should be logged.
    """

    mlflow_tracker._log_metrics(
        mock_training_result,
    )

    mock_log_metrics.assert_called_once()

    metrics = mock_log_metrics.call_args.args[0]

    assert metrics["accuracy"] == mock_training_result.evaluation.accuracy

    assert metrics["roc_auc"] == mock_training_result.evaluation.roc_auc


@patch("src.tracking.mlflow_tracker.mlflow.sklearn.log_model")
def test_log_model_sklearn(
    mock_log_model,
    mock_training_result,
):
    """
    Sklearn models should use sklearn flavor.
    """

    mlflow_tracker._log_model(
        mock_training_result,
    )

    mock_log_model.assert_called_once_with(
        sk_model=mock_training_result.model,
        name="model",
    )


@patch("src.tracking.mlflow_tracker.mlflow.xgboost.log_model")
def test_log_model_xgboost(
    mock_log_model,
    mock_evaluation_result,
    mock_inference_artifacts,
):
    """
    XGBoost models should use xgboost flavor.
    """

    model = XGBClassifier(
        n_estimators=1,
        max_depth=1,
    )

    training_result = TrainingResult(
        model=model,
        evaluation=mock_evaluation_result,
        artifacts=mock_inference_artifacts,
        artifacts_path=Path("artifacts"),
    )

    mlflow_tracker._log_model(
        training_result,
    )

    mock_log_model.assert_called_once_with(
        xgb_model=model,
        artifact_path="model",
    )


@patch("src.tracking.mlflow_tracker.mlflow.log_artifact")
def test_log_artifacts(
    mock_log_artifact,
    mock_training_result,
):
    """
    Experiment artifacts should be logged.
    """

    mlflow_tracker._log_artifacts(
        mock_training_result,
    )

    assert mock_log_artifact.call_count == 3

    calls = mock_log_artifact.call_args_list

    expected = [
        ("metrics", "metrics"),
        ("figures", "figures"),
        ("reports", "reports"),
    ]

    for call_args, (directory, artifact_path) in zip(
        calls,
        expected,
        strict=True,
    ):
        assert call_args.kwargs["artifact_path"] == artifact_path

        logged_path = Path(call_args.args[0])

        assert logged_path.name == directory

        assert logged_path.parent == mock_training_result.artifacts_path


@patch("src.tracking.mlflow_tracker.mlflow.set_tags")
def test_set_tags(
    mock_set_tags,
):
    """
    MLflow tags should be configured.
    """

    mlflow_tracker._set_tags()

    mock_set_tags.assert_called_once()

    tags = mock_set_tags.call_args.args[0]

    assert tags["project"] == "Customer Chrun Prediction"
    assert tags["author"] == "Dishant Patel"


@patch("src.tracking.mlflow_tracker.mlflow.start_run")
@patch("src.tracking.mlflow_tracker.mlflow.set_experiment")
@patch("src.tracking.mlflow_tracker.mlflow.set_tracking_uri")
@patch("src.tracking.mlflow_tracker._log_artifacts")
@patch("src.tracking.mlflow_tracker._log_model")
@patch("src.tracking.mlflow_tracker._log_metrics")
@patch("src.tracking.mlflow_tracker._log_parameters")
@patch("src.tracking.mlflow_tracker._get_tracking_uri")
def test_log_experiment(
    mock_get_uri,
    mock_log_parameters,
    mock_log_metrics,
    mock_log_model,
    mock_log_artifacts,
    mock_set_tracking_uri,
    mock_set_experiment,
    mock_start_run,
    mock_training_result,
):
    """
    Experiment logging should orchestrate MLflow.
    """

    mock_get_uri.return_value = "sqlite:///mlflow.db"

    mock_start_run.return_value.__enter__.return_value = None
    mock_start_run.return_value.__exit__.return_value = None

    mlflow_tracker.log_experiment(
        mock_training_result,
    )

    mock_set_tracking_uri.assert_called_once_with(
        "sqlite:///mlflow.db",
    )

    mock_set_experiment.assert_called_once_with(
        mlflow_tracker.settings.tracking.experiment_name,
    )

    mock_log_parameters.assert_called_once_with(
        mock_training_result,
    )

    mock_log_metrics.assert_called_once_with(
        mock_training_result,
    )

    mock_log_model.assert_called_once_with(
        mock_training_result,
    )

    mock_log_artifacts.assert_called_once_with(
        mock_training_result,
    )
