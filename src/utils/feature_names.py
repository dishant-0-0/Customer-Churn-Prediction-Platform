"""
Utilities for formatting feature names for presentation.
"""

from __future__ import annotations


def clean_feature_name(name: str) -> str:
    """
    Convert internal pipeline feature names into human-readable labels.
    """
    if name.startswith("categorical__"):
        name = name.removeprefix("categorical__")

        parts = name.split("_")

        if len(parts) >= 2:
            feature = parts[0]
            value = "_".join(parts[1:])
            return f"{feature} : {value}"

        return name

    if name.startswith("numerical__"):
        return name.removeprefix("numerical__")

    return name


def clean_feature_names(
    feature_names: list[str],
) -> list[str]:
    """
    Clean multiple feature names.
    """

    return [clean_feature_name(name) for name in feature_names]
