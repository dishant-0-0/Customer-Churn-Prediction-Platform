"""
Data Loading Utilities

This module provides reusable functions for loading datasets 
from the data directory.
"""

from pathlib import Path
import logging

import pandas as pd


logger = logging.getLogger(__name__)


# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"

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

    filepath = DATA_DIR / stage / filename

    if not filepath.exists():
        raise FileNotFoundError(
            f"Dataset not found: {filepath}"
        )

    logger.info("Loading dataset from %s", filepath)

    return pd.read_csv(filepath)