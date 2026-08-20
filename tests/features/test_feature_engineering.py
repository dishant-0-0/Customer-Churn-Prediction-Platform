"""
Tests for feature engineering.
"""

from __future__ import annotations
import pandas as pd
import pytest
from src.features.feature_engineering import (
    validate_dataframe,
    drop_identifier_columns,
    create_tenure_group,
    create_total_services,
    create_avg_monthly_spend,
    create_high_value_customer,
    create_monthly_contract,
    drop_location_columns,
    feature_engineering_pipeline,
)


def test_validate_dataframe_empty():
    """
    Empty dataframe should raise ValueError.
    """

    with pytest.raises(ValueError):
        validate_dataframe(
            pd.DataFrame()
        )


def test_validate_dataframe_missing_columns():
    """
    Missing required columns should raise ValueError.
    """

    df = pd.DataFrame(
        {
            "CustomerID": ["0001"]
        }
    )

    with pytest.raises(ValueError):
        validate_dataframe(df)


def test_validate_dataframe_success(
    sample_dataframe,
):
    """
    Valid dataframe should pass validation.
    """

    result = validate_dataframe(
        sample_dataframe
    )

    assert result.equals(
        sample_dataframe
    )


def test_drop_identifier_columns(
    sample_dataframe,
):
    """
    Identifier columns should be removed.
    """

    result = drop_identifier_columns(
        sample_dataframe
    )

    assert "CustomerID" not in result.columns


def test_create_total_services(
    sample_dataframe,
):
    """
    Total services should be created.
    """

    result = create_total_services(
        sample_dataframe
    )

    assert "Total Services" in result.columns
    assert result["Total Services"].dtype == int
    assert result.loc[0, "Total Services"] == 5
    assert result.loc[1, "Total Services"] == 6


def test_create_monthly_contract(
    sample_dataframe,
):
    """
    Monthly contract indicator should be created.
    """

    result = create_monthly_contract(
        sample_dataframe
    )

    assert "Monthly Contract" in result.columns
    assert result.loc[0, "Monthly Contract"] == 1
    assert result.loc[1, "Monthly Contract"] == 0
    assert result["Monthly Contract"].isin(
        [0, 1]
    ).all()


def test_create_avg_monthly_spend(
    sample_dataframe,
):
    """
    Average monthly spend should be calculated correctly.
    """

    result = create_avg_monthly_spend(
        sample_dataframe
    )

    assert "Avg Monthly Spend" in result.columns
    assert result.loc[0, "Avg Monthly Spend"] == pytest.approx(
        966.0 / 13
    )
    assert result.loc[1, "Avg Monthly Spend"] == pytest.approx(
        1980.0 / 37
    )


def test_create_tenure_group(
    sample_dataframe,
):
    """
    Tenure group should be created.
    """

    result = create_tenure_group(
        sample_dataframe,
    )

    assert "Tenure Group" in result.columns

    assert result["Tenure Group"].notna().all()


def test_drop_location_columns(
    sample_dataframe,
):
    """
    Location columns should be removed.
    """

    result = drop_location_columns(
        sample_dataframe,
    )

    assert "Latitude" not in result.columns
    assert "Longitude" not in result.columns
    assert "Zip Code" not in result.columns

    
def test_create_high_value_customer(
    sample_dataframe,
    high_value_threshold,
):
    """
    High value customer indicator should be binary.
    """

    result = create_high_value_customer(
        df=sample_dataframe,
        threshold=high_value_threshold,
    )

    assert "High Value Customer" in result.columns
    assert result.loc[0, "High Value Customer"] == 0
    assert result.loc[1, "High Value Customer"] == 1
    assert result["High Value Customer"].isin(
        [0, 1]
    ).all()


def test_feature_engineering_pipeline(
    sample_dataframe,
    high_value_threshold,
):
    """
    Feature engineering pipeline should add all engineered features.
    """

    result = feature_engineering_pipeline(
        df=sample_dataframe,
        high_value_threshold=high_value_threshold,
    )

    expected_columns = {
        "Tenure Group",
        "Total Services",
        "Avg Monthly Spend",
        "High Value Customer",
        "Monthly Contract",
    }

    assert len(result) == len(sample_dataframe)
    assert result.isnull().sum().sum() == 0
    assert expected_columns.issubset(
        result.columns
    )