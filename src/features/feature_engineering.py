"""
Feature Engineering Pipeline
"""

from __future__ import annotations
import pandas as pd
from src.config.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

def validate_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate the input dataframe before feature engineering.
    """

    if df.empty:
        raise ValueError("Input dataframe is empty.")

    required_columns = settings.feature_engineering.required_columns

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    logger.info("Dataframe validation successful.")

    return df

def drop_identifier_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove identifier and target leakage columns.
    """

    logger.info("Dropping identifier columns.")

    return df.drop(
        columns= settings.feature_engineering.drop_columns,
        errors= "ignore"
    )

def create_tenure_group(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create tenure groups from tenure months.
    """

    logger.info("Creating 'Tenure Group' feature.")

    df = df.copy()

    df["Tenure Group"] = pd.cut(
        df["Tenure Months"],
        bins = settings.feature_engineering.tenure_bins,
        labels = settings.feature_engineering.tenure_labels,
        include_lowest = True
    )

    return df

def create_avg_monthly_spend(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create average monthly spend feature.
    """

    logger.info("Creating 'Avg Monthly Spend' feature.")

    df = df.copy()

    df["Avg Monthly Spend"] = (
        df["Total Charges"]
        / (df["Tenure Months"] + 1)
    )

    return df

def create_high_value_customer(
        df: pd.DataFrame,
        threshold: float
    ) -> pd.DataFrame:
    """
    Create high-value customer indicator.
    """

    logger.info("Creating 'High Value Customer' feature.")

    df = df.copy()

    df["High Value Customer"] = (
        df["CLTV"] > threshold
    ).astype(int)

    return df

def drop_location_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop location-related columns.
    """

    logger.info("Dropping location columns.")

    return df.drop(
        columns= settings.feature_engineering.drop_location_columns,
        errors= "ignore"
    )

def create_total_services(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create total subscribed services feature.
    """

    logger.info("Creating 'Total Services' feature.")

    df = df.copy()

    service_columns = [
        "Phone Service",
        "Multiple Lines",
        "Online Security",
        "Online Backup",
        "Device Protection",
        "Tech Support",
        "Streaming TV",
        "Streaming Movies",
    ]

    df["Total Services"] = (
        df[service_columns]
        .eq("Yes")
        .sum(axis=1)
    )

    return df


def create_monthly_contract(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create monthly contract indicator.
    """

    logger.info("Creating 'Monthly Contract' feature.")

    df = df.copy()

    df["Monthly Contract"] = (
        df["Contract"] == "Month-to-month"
    ).astype(int)

    return df


def feature_engineering_pipeline(
        df: pd.DataFrame,
        high_value_threshold: float,
    ) -> pd.DataFrame:
    """
    Apply all feature engineering transformations.
    """

    logger.info("Starting feature engineering pipeline.")

    df = validate_dataframe(df)
    df = drop_identifier_columns(df)
    df = create_tenure_group(df)
    df = create_total_services(df)
    df = create_avg_monthly_spend(df)
    df = create_high_value_customer(df, high_value_threshold)
    df = create_monthly_contract(df)
    df = drop_location_columns(df)

    logger.info("Feature engineering completed.")

    return df