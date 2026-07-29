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