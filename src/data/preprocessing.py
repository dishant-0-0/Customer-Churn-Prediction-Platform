"""
Data preprocessing utilities
"""

from typing import Tuple

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler


def split_data(
    df: pd.DataFrame,
    target: str,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple:
    """
    Split dataframe into train and test sets
    """

    X = df.drop(columns = [target])
    y = df[target]

    return train_test_split(
        X,
        y,
        test_size = test_size,
        stratify = y,
        random_state = random_state
    )


def get_features_types(
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

    return numerical_cols, categorical_cols