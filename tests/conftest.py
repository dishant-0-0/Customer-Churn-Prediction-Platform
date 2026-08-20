"""
Shared pytest fixtures.
"""

from __future__ import annotations
import pandas as pd
import pytest
from unittest.mock import Mock
import numpy as np


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
def mock_model() -> Mock:
    """
    Return a mock classifier.
    """

    model = Mock()

    model.predict_proba.return_value = np.array(
        [
            [0.3, 0.7],
            [0.8, 0.2],
        ]
    )

    return model


@pytest.fixture
def mock_processed_data() -> Mock:
    """
    Return processed test data.
    """

    processed = Mock()

    processed.X_test_processed = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ]
    )

    processed.y_test = np.array(
        [
            1,
            0,
        ]
    )

    return processed