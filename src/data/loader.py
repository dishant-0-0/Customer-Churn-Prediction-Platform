"""
Data Loading Utilities

This module provides reusable functions for loading datasets 
from the data directory.
"""

from pathlib import Path
import logging
import pandas as pd
from src.config.paths import (
    RAW_DATA_DIR,
    INTERIM_DATA_DIR,
    PROCESSED_DATA_DIR
)

logger = logging.getLogger(__name__)


DATA_STAGE_MAP = {
    "raw" : RAW_DATA_DIR,
    "interim" : INTERIM_DATA_DIR,
    "processed" : PROCESSED_DATA_DIR
}

def load_data(stage: str, filename: str) -> pd.DataFrame:
    """
    Load a dataset from the specified data stage.
    
    Parameters
    ----------
    stage : str
        One of:
        - raw
        - interim
        - processed

    filename : str
        CSV filename

    Returns
    -------
    pd.DataFrame
        Loaded dataset.
    """

    filepath = DATA_STAGE_MAP[stage] / filename

    if not filepath.exists():
        raise FileNotFoundError(
            f"Dataset not found: {filepath}"
        )

    logger.info("Loading dataset from %s", filepath)

    return pd.read_csv(filepath)