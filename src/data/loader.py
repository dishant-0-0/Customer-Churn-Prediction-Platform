"""
Data Loading Utilities

This module provides reusable functions for loading datasets
from the data directory.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pandas as pd

from src.config.paths import INTERIM_DATA_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)

DATA_STAGE_MAP: dict[str, Path] = {
    "raw": RAW_DATA_DIR,
    "interim": INTERIM_DATA_DIR,
    "processed": PROCESSED_DATA_DIR,
}


def load_data(stage: str, filename: str, **kwargs) -> pd.DataFrame:
    """
    Load a dataset from the specified data stage.
    """

    if stage not in DATA_STAGE_MAP:
        raise ValueError(
            f"Invalid stage: {stage}.Expected stages: {list(DATA_STAGE_MAP)}."
        )

    filepath = DATA_STAGE_MAP[stage] / filename

    if not filepath.exists():
        raise FileNotFoundError(f"Dataset not found: {filepath}")

    logger.info("Loading data from '%s'.", filepath)

    df = cast(
        pd.DataFrame,
        pd.read_csv(filepath, **kwargs),
    )

    logger.info("Loaded dataset with shape %s", df.shape)

    return df


def save_data(df: pd.DataFrame, stage: str, filename: str, **kwargs) -> None:
    """
    Save a dataframe to the specified data stage.
    """

    if stage not in DATA_STAGE_MAP:
        raise ValueError(
            f"Invalid stage: {stage}.Expected stages: {list(DATA_STAGE_MAP)}."
        )

    filepath = DATA_STAGE_MAP[stage] / filename

    logger.info("Saving dataset to '%s'.", filepath)

    filepath.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(filepath, index=False, **kwargs)

    logger.info("Successfully saved dataset with shape %s.", df.shape)
