"""
Tests for the training pipeline.
"""

from __future__ import annotations
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
import pandas as pd
from src.core import (
    InferenceArtifacts,
    ProcessedData,
    TrainingResult,
)
from src.pipelines.training_pipeline import (
    prepare_training_data,
    run_training_pipeline,
)



@patch("src.pipelines.training_pipeline.clean_feature_names")
@patch("src.pipelines.training_pipeline.build_preprocessor")
@patch("src.pipelines.training_pipeline.get_feature_types")
@patch("src.pipelines.training_pipeline.split_data")
@patch("src.pipelines.training_pipeline.feature_engineering_pipeline")
@patch("src.pipelines.training_pipeline.load_data")
def test_prepare_training_data(
    mock_load_data,
    mock_feature_engineering,
    mock_split_data,
    mock_get_feature_types,
    mock_build_preprocessor,
    mock_clean_feature_names,
    sample_dataframe,
):
    """
    Training data should be prepared successfully.
    """

    df = sample_dataframe.copy()

    df["CLTV"] = np.linspace(
        1000,
        5000,
        len(df),
    )

    mock_load_data.return_value = df
    mock_feature_engineering.return_value = df

    X_train = pd.DataFrame(
        {
            "A": [1],
            "B": [2],
        }
    )

    X_test = pd.DataFrame(
        {
            "A": [3],
            "B": [4],
        }
    )

    y_train = pd.Series([0])
    y_test = pd.Series([1])

    mock_split_data.return_value = (
        X_train,
        X_test,
        y_train,
        y_test,
    )

    mock_get_feature_types.return_value = (
        ["A"],
        ["B"],
    )

    preprocessor = MagicMock()

    preprocessor.fit_transform.return_value = np.array([[1.0, 2.0]])
    preprocessor.transform.return_value = np.array([[3.0, 4.0]])
    preprocessor.get_feature_names_out.return_value = np.array(
        [
            "A",
            "B",
        ]
    )

    mock_build_preprocessor.return_value = preprocessor

    mock_clean_feature_names.return_value = [
        "A",
        "B",
    ]

    result = prepare_training_data()

    assert isinstance(
        result,
        ProcessedData,
    )

    mock_load_data.assert_called_once()
    mock_feature_engineering.assert_called_once()
    mock_split_data.assert_called_once()
    mock_get_feature_types.assert_called_once_with(
        X_train,
    )
    mock_build_preprocessor.assert_called_once_with(
        numerical_cols=["A"],
        categorical_cols=["B"],
    )
    mock_clean_feature_names.assert_called_once()

    assert result.feature_names == [
        "A",
        "B",
    ]


@patch("src.pipelines.training_pipeline.log_experiment")
@patch("src.pipelines.training_pipeline.generate_html_report")
@patch("src.pipelines.training_pipeline.save_figures")
@patch("src.pipelines.training_pipeline.generate_visualization")
@patch("src.pipelines.training_pipeline.generate_explanation")
@patch("src.pipelines.training_pipeline.save_metrics")
@patch("src.pipelines.training_pipeline.save_artifacts")
@patch("src.pipelines.training_pipeline.evaluate_model")
@patch("src.pipelines.training_pipeline.train_model")
@patch("src.pipelines.training_pipeline.get_model")
@patch("src.pipelines.training_pipeline.prepare_training_data")
def test_run_training_pipeline(
    mock_prepare_training_data,
    mock_get_model,
    mock_train_model,
    mock_evaluate_model,
    mock_save_artifacts,
    mock_save_metrics,
    mock_generate_explanation,
    mock_generate_visualization,
    mock_save_figures,
    mock_generate_html_report,
    mock_log_experiment,
    mock_processed_data,
    mock_evaluation_result,
):
    """
    Training pipeline should orchestrate every stage.
    """

    model = MagicMock()

    mock_prepare_training_data.return_value = (
        mock_processed_data
    )

    mock_get_model.return_value = model
    mock_train_model.return_value = model
    mock_evaluate_model.return_value = (
        mock_evaluation_result
    )

    mock_save_artifacts.return_value = Path(
        "artifacts/xgboost_v1"
    )

    mock_generate_explanation.return_value = MagicMock()

    mock_generate_visualization.return_value = {
        "roc_curve": MagicMock(),
    }

    result = run_training_pipeline(
        force=True,
    )

    assert isinstance(
        result,
        TrainingResult,
    )

    assert isinstance(
        result.artifacts,
        InferenceArtifacts,
    )

    mock_prepare_training_data.assert_called_once()

    mock_get_model.assert_called_once()

    mock_train_model.assert_called_once_with(
        mock_processed_data,
        model,
    )

    mock_evaluate_model.assert_called_once_with(
        mock_processed_data,
        model,
    )

    mock_save_artifacts.assert_called_once()

    mock_save_metrics.assert_called_once_with(
        mock_evaluation_result,
    )

    mock_generate_explanation.assert_called_once()

    mock_generate_visualization.assert_called_once()

    mock_save_figures.assert_called_once()

    mock_generate_html_report.assert_called_once()

    mock_log_experiment.assert_called_once_with(
        training_result=result,
    )