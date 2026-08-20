"""
Tests for data loader utilities.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.loader import (
    DATA_STAGE_MAP,
    load_data,
    save_data,
)


def test_load_data_invalid_stage():
    """
    Invalid stage should raise ValueError.
    """

    with pytest.raises(ValueError):
        load_data(
            stage="invalid",
            filename="customers.csv",
        )


def test_load_data_missing_file(
    tmp_path,
    monkeypatch,
):
    """
    Missing dataset should raise FileNotFoundError.
    """

    monkeypatch.setitem(
        DATA_STAGE_MAP,
        "raw",
        tmp_path,
    )

    with pytest.raises(FileNotFoundError):
        load_data(
            stage="raw",
            filename="missing.csv",
        )


def test_save_data(
    sample_dataframe,
    tmp_path,
    monkeypatch,
):
    """
    DataFrame should be saved successfully and create the parent directory.
    """

    nested_dir = tmp_path / "nested"

    monkeypatch.setitem(
        DATA_STAGE_MAP,
        "raw",
        nested_dir,
    )

    save_data(
        df=sample_dataframe,
        stage="raw",
        filename="customers.csv",
    )

    assert nested_dir.exists()
    assert nested_dir.is_dir()

    saved_file = nested_dir / "customers.csv"

    assert saved_file.exists()


def test_load_data(
    sample_dataframe,
    tmp_path,
    monkeypatch,
):
    """
    Dataset should be loaded successfully.
    """

    monkeypatch.setitem(
        DATA_STAGE_MAP,
        "raw",
        tmp_path,
    )

    sample_dataframe.to_csv(
        tmp_path / "customers.csv",
        index=False,
    )

    result = load_data(
        stage="raw",
        filename="customers.csv",
        dtype={"CustomerID": str},
    )

    pd.testing.assert_frame_equal(
        result,
        sample_dataframe,
    )


def test_save_data_invalid_stage(
    sample_dataframe,
):
    """
    Invalid stage should raise ValueError.
    """

    with pytest.raises(ValueError):
        save_data(
            df=sample_dataframe,
            stage="invalid",
            filename="customers.csv",
        )


def test_save_and_load_round_trip(
    sample_dataframe,
    tmp_path,
    monkeypatch,
):
    """
    Saved data should be loaded without modification.
    """

    monkeypatch.setitem(
        DATA_STAGE_MAP,
        "raw",
        tmp_path,
    )

    save_data(
        df=sample_dataframe,
        stage="raw",
        filename="customers.csv",
    )

    loaded = load_data(
        stage="raw",
        filename="customers.csv",
        dtype={"CustomerID": str},
    )

    pd.testing.assert_frame_equal(
        loaded,
        sample_dataframe,
    )


def test_load_data_passes_kwargs(
    sample_dataframe,
    tmp_path,
    monkeypatch,
):
    """
    load_data should forward keyword arguments to pandas.read_csv().
    """

    monkeypatch.setitem(
        DATA_STAGE_MAP,
        "raw",
        tmp_path,
    )

    sample_dataframe.to_csv(
        tmp_path / "customers.csv",
        index=False,
    )

    result = load_data(
        stage="raw",
        filename="customers.csv",
        dtype={"CustomerID": str},
    )

    assert result["CustomerID"].dtype == object
