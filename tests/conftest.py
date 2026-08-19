"""
Shared pytest fixtures.
"""

from __future__ import annotations
import pandas as pd
import pytest


SAMPLE_CUSTOMER = {
    "CustomerID": ["0001", "0002"],
    "Gender": ["Male", "Female"],
    "Senior Citizen": [0, 1],
    "Partner": ["Yes", "No"],
    "Dependents": ["No", "Yes"],
    "Tenure Months": [12, 36],
    "Phone Service": ["Yes", "Yes"],
    "Multiple Lines": ["No", "Yes"],
    "Internet Service": ["Fiber optic", "DSL"],
    "Online Security": ["No", "Yes"],
    "Online Backup": ["Yes", "No"],
    "Device Protection": ["Yes", "Yes"],
    "Tech Support": ["No", "Yes"],
    "Streaming TV": ["Yes", "No"],
    "Streaming Movies": ["Yes", "Yes"],
    "Contract": ["Month-to-month", "Two year"],
    "Paperless Billing": ["Yes", "No"],
    "Payment Method": [
        "Electronic check",
        "Credit card (automatic)",
    ],
    "Monthly Charges": [80.5, 55.0],
    "Total Charges": [966.0, 1980.0],
    "CLTV": [4200, 6200],
    "Latitude": [40.1, 41.2],
    "Longitude": [-75.2, -74.3],
    "Zip Code": [10001, 10002],
    "Churn Value": [1, 0],
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
def high_value_threshold() -> float:
    """
    Return the CLTV threshold.
    """

    return sample_dataframe["CLTV"].median()