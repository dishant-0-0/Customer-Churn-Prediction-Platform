"""
Tests for preprocessing utilities.
"""

from __future__ import annotations

import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer

from src.data.preprocessing import (
    build_preprocessor,
    get_feature_types,
    split_data,
)


def test_split_data_empty_dataframe():
    """
    Empty dataframe should raise ValueError.
    """

    with pytest.raises(ValueError):
        split_data(
            df=pd.DataFrame(),
            target="Churn Value",
        )


def test_split_data_missing_target(
    sample_prediction_dataframe,
):
    """
    Missing target should raise ValueError.
    """

    with pytest.raises(ValueError):
        split_data(
            df=sample_prediction_dataframe,
            target="Churn Value",
        )


def test_split_data_success(
    sample_dataframe,
):
    """
    Dataset should be split correctly.
    """

    X_train, X_test, y_train, y_test = split_data(
        df=sample_dataframe,
        target="Churn Value",
        test_size=0.5,
        random_state=42,
    )

    assert len(X_train) == 2
    assert len(X_test) == 2
    assert len(y_train) == 2
    assert len(y_test) == 2
    assert "Churn Value" not in X_train.columns
    assert "Churn Value" not in X_test.columns
    assert y_train.name == "Churn Value"
    assert y_test.name == "Churn Value"


def test_get_feature_types_empty_dataframe():
    """
    Empty dataframe should raise ValueError.
    """

    with pytest.raises(ValueError):
        get_feature_types(
            pd.DataFrame(),
        )


def test_get_feature_types(
    sample_prediction_dataframe,
):
    """
    Numerical and categorical features should be detected.
    """

    numerical, categorical = get_feature_types(
        sample_prediction_dataframe,
    )

    assert "Monthly Charges" in numerical
    assert "Total Charges" in numerical
    assert "Gender" in categorical
    assert "Contract" in categorical


def test_build_preprocessor(
    sample_prediction_dataframe,
):
    """
    Preprocessor should be created.
    """

    numerical, categorical = get_feature_types(
        sample_prediction_dataframe,
    )

    preprocessor = build_preprocessor(
        numerical,
        categorical,
    )

    assert isinstance(
        preprocessor,
        ColumnTransformer,
    )


def test_build_preprocessor_fit_transform(
    sample_prediction_dataframe,
):
    """
    Preprocessor should fit and transform data.
    """

    numerical, categorical = get_feature_types(
        sample_prediction_dataframe,
    )

    preprocessor = build_preprocessor(
        numerical,
        categorical,
    )

    transformed = preprocessor.fit_transform(
        sample_prediction_dataframe,
    )

    assert transformed.shape[0] == len(
        sample_prediction_dataframe,
    )


def test_build_preprocessor_unknown_category(
    sample_prediction_dataframe,
):
    """
    Unknown categories should be ignored.
    """

    numerical, categorical = get_feature_types(
        sample_prediction_dataframe,
    )

    preprocessor = build_preprocessor(
        numerical,
        categorical,
    )

    preprocessor.fit(
        sample_prediction_dataframe,
    )

    new_data = sample_prediction_dataframe.copy()

    new_data.loc[
        0,
        "Gender",
    ] = "Unknown"

    transformed = preprocessor.transform(
        new_data,
    )

    assert transformed.shape[0] == len(
        new_data,
    )
