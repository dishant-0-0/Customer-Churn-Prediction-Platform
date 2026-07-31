"""
Data preprocessing utilities
"""

from typing import Tuple
from src.utils.logger import get_logger
import pandas as pd
from src.core.entities import ProcessedData
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from src.config.config import settings

logger = get_logger(__name__)

def split_data(
    df: pd.DataFrame,
    target: str,
    test_size: float | None = None,
    random_state: int | None = None  
) -> Tuple:
    """
    Split dataframe into train and test sets
    """

    logger.info("Splitting dataset into train and test sets.")

    if test_size is None:
        test_size = settings.data.test_size
    if random_state is None:
        random_state = settings.data.random_state

    X = df.drop(columns = [target])
    y = df[target]

    return train_test_split(
        X,
        y,
        test_size = test_size,
        stratify = y,
        random_state = random_state
    )


def get_feature_types(
    X: pd.DataFrame,
):
    """
    Return numerical and categorical feature lists.
    """

    numerical_cols = X.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    categorical_cols = X.select_dtypes(
        include=["object","category","bool"]
    ).columns.tolist()

    logger.info(
        "Detected %d numerical and %d catergorical features.",
        len(numerical_cols),
        len(categorical_cols)
    )

    return numerical_cols, categorical_cols


def build_preprocessor(
    numerical_cols,
    categorical_cols
):
    """
    Create preprocessing pipeline
    """

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median")
            ),
            (
                "scaler",
                StandardScaler()
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent")
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numerical",
                numeric_pipeline,
                numerical_cols,
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_cols,
            ),
        ]
    )

    return preprocessor