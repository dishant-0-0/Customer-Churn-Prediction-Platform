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

    Parameters
    ----------
    df: pd.DataFrame (Input Dataframe)

    Returns
    -------
    pd.DataFrame (Validated Dataframe)

    Raises
    ------
    ValueError (If the dataframe is empty or missing required columns)
    """

    if df.empty:
        raise ValueError("Input dataframe is empty.")

    required_columns = settings.feature_engineering.required_columns

    missing = set(required_columns) - set(df.columns)

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

def create_tenure_groups(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create tenure groups from tenure months.

    Parameters
    ----------
    df: pd.DataFrame (Input Dataframe)

    Returns
    -------
    pd.DataFrame (Dataframe with the 'Tenure Group' feature)
    """

    logger.info("Creating tenure groups.")

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

    logger.info("Creating average monthly spend feature.")

    df = df.copy()

    df["Avg Monthly Spend"] = (
        df["Total Charges"]
        / (df["Tenure Months"] + 1)
    )

    return df

def create_high_value_customer(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create high-value customer indicator.
    """

    logger.info("Creating high-value customer feature.")

    df = df.copy()

    strategy = settings.feature_engineering.high_value_strategy
    if strategy == "median":
        threshold = df["CLTV"].meadian()
    else:
        raise ValueError(
            f"Unsupported strategy: {strategy}"
        )

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

def feature_engineering_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all feature engineering transformations.
    """

    logger.info("Starting feature engineering pipeline.")

    df = validate_dataframe(df)
    df = drop_identifier_columns(df)
    df = create_tenure_groups(df)
    df = create_avg_monthly_spend(df)
    df = create_high_value_customer(df)
    df = drop_location_columns(df)

    logger.info("Feature engineering completed.")

    return df