"""
Shared pytest fixtures.
"""

from __future__ import annotations
import pandas as pd
import numpy as np
import pytest
import json
import joblib
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from src.persistence import save, load
from unittest.mock import Mock, MagicMock
from src.core import (
    InferenceArtifacts, 
    EvaluationResult, 
    TrainingResult,
    ProcessedData
)


SAMPLE_CUSTOMER = {
    "CustomerID": ["0001", "0002", "0003", "0004"],
    "Gender": ["Male", "Female", "Male", "Female"],
    "Senior Citizen": [0, 1, 0, 1],
    "Partner": ["Yes", "No", "Yes", "No"],
    "Dependents": ["No", "Yes", "No", "Yes"],
    "Tenure Months": [12, 36, 18, 48],
    "Phone Service": ["Yes", "Yes", "Yes", "Yes"],
    "Multiple Lines": ["No", "Yes", "Yes", "No"],
    "Internet Service": [
        "Fiber optic",
        "DSL",
        "DSL",
        "Fiber optic",
    ],
    "Online Security": ["No", "Yes", "Yes", "No"],
    "Online Backup": ["Yes", "No", "Yes", "No"],
    "Device Protection": ["Yes", "Yes", "No", "Yes"],
    "Tech Support": ["No", "Yes", "No", "Yes"],
    "Streaming TV": ["Yes", "No", "Yes", "No"],
    "Streaming Movies": ["Yes", "Yes", "No", "Yes"],
    "Contract": [
        "Month-to-month",
        "Two year",
        "One year",
        "Month-to-month",
    ],
    "Paperless Billing": ["Yes", "No", "Yes", "No"],
    "Payment Method": [
        "Electronic check",
        "Credit card (automatic)",
        "Bank transfer (automatic)",
        "Mailed check",
    ],
    "Monthly Charges": [80.5, 55.0, 70.0, 95.0],
    "Total Charges": [966.0, 1980.0, 1260.0, 4560.0],
    "CLTV": [4200, 6200, 5000, 7100],
    "Latitude": [40.1, 41.2, 40.8, 41.5],
    "Longitude": [-75.2, -74.3, -75.0, -74.1],
    "Zip Code": [10001, 10002, 10003, 10004],
    "Churn Value": [1, 0, 1, 0],
}

@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    """
    Return a sample customer dataframe.
    """

    return pd.DataFrame(SAMPLE_CUSTOMER).copy()


@pytest.fixture
def sample_prediction_dataframe(
    sample_dataframe: pd.DataFrame
) -> pd.DataFrame:
    """
    Return sample data without the target column.
    """

    return sample_dataframe.drop(
        columns=["Churn Value"],
    )


@pytest.fixture
def high_value_threshold(
    sample_dataframe: pd.DataFrame
) -> float:
    """
    Return the CLTV threshold.
    """

    return sample_dataframe["CLTV"].median()


@pytest.fixture
def mock_preprocessor() -> Mock:
    """
    Return a fitted mock preprocessor.
    """

    preprocessor = Mock()

    preprocessor.transform.return_value = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ]
    )

    return preprocessor


@pytest.fixture
def mock_model():
    model = MagicMock()

    model.predict.return_value = np.array([1, 0])

    model.predict_proba.return_value = np.array(
        [
            [0.2, 0.8],
            [0.7, 0.3],
        ]
    )

    model.get_params.return_value = {
        "n_estimators": 100,
        "max_depth": 6,
    }

    return model


@pytest.fixture
def serializable_model() -> DummyClassifier:
    """
    Return a fitted sklearn model that can be serialized.
    """

    model = DummyClassifier(
        strategy="most_frequent",
    )

    model.fit(
        [[0], [1]],
        [0, 1],
    )

    return model


@pytest.fixture
def mock_processed_data():
    """
    Return a fully populated ProcessedData instance.
    """

    preprocessor = MagicMock()

    preprocessor.get_feature_names_out.return_value = np.array(
        [
            "A",
            "B",
        ]
    )

    return ProcessedData(
        X_train=pd.DataFrame(
            {
                "A": [1, 2],
                "B": [3, 4],
            }
        ),
        X_test=pd.DataFrame(
            {
                "A": [5, 6],
                "B": [7, 8],
            }
        ),
        X_train_processed=np.array(
            [
                [1.0, 3.0],
                [2.0, 4.0],
            ]
        ),
        X_test_processed=np.array(
            [
                [5.0, 7.0],
                [6.0, 8.0],
            ]
        ),
        y_train=pd.Series([0, 1]),
        y_test=pd.Series([1, 0]),
        preprocessor=preprocessor,
        feature_names=[
            "A",
            "B",
        ],
        numerical_columns=[
            "A",
        ],
        categorical_columns=[
            "B",
        ],
        high_value_threshold=5000.0,
    )


@pytest.fixture
def sample_feature_names() -> list[str]:
    """
    Return sample feature names.
    """

    return [
        "Monthly Charges",
        "Tenure Months",
        "Total Services",
        "Avg Monthly Spend",
    ]


@pytest.fixture
def mock_inference_artifacts(
    mock_model,
    mock_processed_data,
) -> InferenceArtifacts:
    """
    Return inference artifacts for testing.
    """

    return InferenceArtifacts(
        model=mock_model,
        preprocessor=mock_processed_data.preprocessor,
        feature_names=mock_processed_data.feature_names,
        high_value_threshold=mock_processed_data.high_value_threshold,
    )


@pytest.fixture
def serializable_inference_artifacts(
    serializable_model,
):
    """
    Return fully serializable inference artifacts.
    """

    preprocessor = StandardScaler()
    preprocessor.fit(
        [
            [0.0, 0.0],
            [1.0, 1.0],
        ]
    )

    return InferenceArtifacts(
        model=serializable_model,
        preprocessor=preprocessor,
        feature_names=["A", "B"],
        high_value_threshold=5000.0,
    )


@pytest.fixture
def mock_evaluation_result() -> EvaluationResult:
    """
    Return a mock evaluation result.
    """

    return EvaluationResult(
        y_pred=np.array([1, 0, 1, 0]),
        y_prob=np.array([0.91, 0.12, 0.81, 0.22]),
        accuracy=0.95,
        precision=0.94,
        recall=0.96,
        f1=0.95,
        roc_auc=0.98,
        confusion_matrix=np.array(
            [
                [2, 0],
                [0, 2],
            ]
        ),
        fpr=np.array(
            [
                0.0,
                0.1,
                1.0,
            ]
        ),
        tpr=np.array(
            [
                0.0,
                0.9,
                1.0,
            ]
        ),
        roc_thresholds=np.array(
            [
                np.inf,
                0.9,
                0.1,
            ]
        ),
        precision_curve=np.array(
            [
                1.0,
                0.95,
                0.90,
            ]
        ),
        recall_curve=np.array(
            [
                0.0,
                0.75,
                1.0,
            ]
        ),
        average_precision=0.97,
        pr_thresholds=np.array(
            [
                0.2,
                0.5,
            ]
        ),
    )


@pytest.fixture
def patched_artifact_dirs(
    tmp_path,
    monkeypatch,
):
    """
    Patch persistence directories to use temporary folders.
    """

    artifacts_dir = tmp_path / "artifacts"
    config_dir = tmp_path / "config"

    monkeypatch.setattr(
        save,
        "ARTIFACTS_DIR",
        artifacts_dir,
    )

    monkeypatch.setattr(
        load,
        "ARTIFACTS_DIR",
        artifacts_dir,
    )

    monkeypatch.setattr(
        save,
        "CONFIG_DIR",
        config_dir,
    )

    config_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    config_file = (
        config_dir /
        save.settings.files.config_file
    )

    config_file.write_text(
        "dummy-config",
        encoding="utf-8",
    )

    return artifacts_dir


@pytest.fixture
def create_saved_artifacts(
    patched_artifact_dirs,
):
    """
    Factory fixture that creates a complete experiment directory
    containing all required inference artifacts.
    """

    def _create(
        experiment_name: str,
        feature_names: list[str] | None = None,
        high_value_threshold: float = 5000.0,
    ) -> tuple[str, Path]:

        if feature_names is None:
            feature_names = [
                "A",
                "B",
            ]

        experiment_dir = (
            patched_artifact_dirs /
            experiment_name
        )

        model_dir = (
            experiment_dir /
            "model"
        )

        model_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        model = DummyClassifier(
            strategy="most_frequent",
        )

        model.fit(
            [[0], [1]],
            [0, 1],
        )

        preprocessor = StandardScaler()

        preprocessor.fit(
            [[0], [1]],
        )

        joblib.dump(
            model,
            model_dir / "model.joblib",
        )

        joblib.dump(
            preprocessor,
            model_dir / "preprocessor.joblib",
        )

        with (
            model_dir /
            "feature_names.json"
        ).open(
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                feature_names,
                file,
            )

        with (
            model_dir /
            "metadata.json"
        ).open(
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                {
                    "high_value_threshold":
                        high_value_threshold,
                },
                file,
            )

        return (
            experiment_name,
            experiment_dir,
        )

    return _create


@pytest.fixture
def mock_training_result(
    mock_evaluation_result,
    mock_inference_artifacts,
) -> TrainingResult:
    """
    Return a mock TrainingResult.
    """

    return TrainingResult(
        model=mock_inference_artifacts.model,
        evaluation=mock_evaluation_result,
        artifacts=mock_inference_artifacts,
        artifacts_path=Path("artifacts/xgboost_v1"),
    )