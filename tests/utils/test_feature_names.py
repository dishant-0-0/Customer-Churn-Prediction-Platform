"""
Tests for feature name formatting utilities.
"""

from __future__ import annotations

from src.utils.feature_names import (
    clean_feature_name,
    clean_feature_names,
)


def test_clean_feature_name_categorical():
    """
    Categorical feature names should be formatted correctly.
    """

    result = clean_feature_name(
        "categorical__Contract_Month-to-month",
    )

    assert result == "Contract : Month-to-month"


def test_clean_feature_name_categorical_multiple_underscores():
    """
    Categorical values containing underscores should be preserved.
    """

    result = clean_feature_name(
        "categorical__Payment Method_Bank_transfer_(automatic)",
    )

    assert result == "Payment Method : Bank_transfer_(automatic)"


def test_clean_feature_name_categorical_without_value():
    """
    Malformed categorical names without an encoded value should
    return the feature name unchanged.
    """

    result = clean_feature_name(
        "categorical__Contract",
    )

    assert result == "Contract"


def test_clean_feature_name_numerical():
    """
    Numerical prefixes should be removed.
    """

    result = clean_feature_name(
        "numerical__Monthly Charges",
    )

    assert result == "Monthly Charges"


def test_clean_feature_name_plain():
    """
    Plain feature names should remain unchanged.
    """

    result = clean_feature_name(
        "Total Services",
    )

    assert result == "Total Services"


def test_clean_feature_names():
    """
    Multiple feature names should all be cleaned.
    """

    result = clean_feature_names(
        [
            "numerical__Monthly Charges",
            "categorical__Contract_Month-to-month",
            "Total Services",
        ]
    )

    assert result == [
        "Monthly Charges",
        "Contract : Month-to-month",
        "Total Services",
    ]


def test_clean_feature_names_empty():
    """
    Empty input should return an empty list.
    """

    assert clean_feature_names([]) == []
