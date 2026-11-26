"""
tests/test_prepare_data.py
==========================
Unit tests for Task 2 (data preparation) artifacts.

Run from the project root::

    .venv/Scripts/python.exe -m pytest tests/test_prepare_data.py -v
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT       = Path(__file__).resolve().parents[1]
MODEL_DIR  = ROOT / "outputs" / "model"

# ---------------------------------------------------------------------------
# Fixtures — load artifacts once per session
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def district_order():
    path = MODEL_DIR / "district_order.json"
    assert path.exists(), f"district_order.json not found; run prepare_data.py first."
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)["districts"]


@pytest.fixture(scope="session")
def feature_matrix():
    path = MODEL_DIR / "feature_matrix.npy"
    assert path.exists(), "feature_matrix.npy not found."
    return np.load(path)


@pytest.fixture(scope="session")
def case_matrix():
    path = MODEL_DIR / "case_matrix.npy"
    assert path.exists(), "case_matrix.npy not found."
    return np.load(path)


@pytest.fixture(scope="session")
def date_index():
    path = MODEL_DIR / "date_index.csv"
    assert path.exists(), "date_index.csv not found."
    return pd.read_csv(path)


@pytest.fixture(scope="session")
def split_indices():
    path = MODEL_DIR / "split_indices.json"
    assert path.exists(), "split_indices.json not found."
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


@pytest.fixture(scope="session")
def feature_names():
    path = MODEL_DIR / "feature_names.json"
    assert path.exists(), "feature_names.json not found."
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)["features"]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestDistrictOrder:
    def test_exactly_25_districts(self, district_order):
        assert len(district_order) == 25, (
            f"Expected 25 districts, got {len(district_order)}"
        )

    def test_alphabetical_order(self, district_order):
        assert district_order == sorted(district_order), (
            "District order is not alphabetical."
        )

    def test_no_duplicates(self, district_order):
        assert len(set(district_order)) == len(district_order), (
            "Duplicate district names found."
        )

    def test_known_districts_present(self, district_order):
        expected = {
            "Colombo", "Kandy", "Galle", "Jaffna", "Ampara",
            "Trincomalee", "Batticaloa", "Anuradhapura", "Kurunegala",
        }
        for d in expected:
            assert d in district_order, f"{d} not found in district_order."


class TestFeatureMatrix:
    def test_shape(self, feature_matrix, district_order):
        n_months = 204   # 2008-01 through 2024-12
        n_dist   = len(district_order)
        n_feat   = 5     # Cases, Precip, TempMin, TempMax, SoilMoisture
        assert feature_matrix.shape == (n_months, n_dist, n_feat), (
            f"feature_matrix shape {feature_matrix.shape} ≠ "
            f"({n_months}, {n_dist}, {n_feat})"
        )

    def test_dtype_float32(self, feature_matrix):
        assert feature_matrix.dtype == np.float32, (
            f"Expected float32, got {feature_matrix.dtype}"
        )

    def test_no_nans(self, feature_matrix):
        n_nan = int(np.isnan(feature_matrix).sum())
        assert n_nan == 0, f"feature_matrix contains {n_nan} NaN values."

    def test_no_negative_cases(self, feature_matrix):
        cases = feature_matrix[:, :, 0]
        assert (cases < 0).sum() == 0, "Negative case counts found."

    def test_feature_names_count(self, feature_names):
        assert len(feature_names) == 5, (
            f"Expected 5 feature names, got {len(feature_names)}"
        )

    def test_cases_is_first_feature(self, feature_names):
        assert feature_names[0] == "Cases", (
            f"First feature should be 'Cases', got '{feature_names[0]}'"
        )


class TestCaseMatrix:
    def test_shape(self, case_matrix, district_order):
        n_months = 204
        n_dist   = len(district_order)
        assert case_matrix.shape == (n_months, n_dist), (
            f"case_matrix shape {case_matrix.shape} ≠ ({n_months}, {n_dist})"
        )

    def test_matches_feature_matrix_cases(self, case_matrix, feature_matrix):
        np.testing.assert_array_equal(
            case_matrix, feature_matrix[:, :, 0],
            err_msg="case_matrix does not match feature_matrix[:,:,0]"
        )


class TestDateIndex:
    def test_row_count(self, date_index):
        assert len(date_index) == 204, (
            f"Expected 204 date rows, got {len(date_index)}"
        )

    def test_first_month(self, date_index):
        assert date_index.iloc[0]["date"] == "2008-01-01", (
            f"First date should be 2008-01-01, got {date_index.iloc[0]['date']}"
        )

    def test_last_month(self, date_index):
        assert date_index.iloc[-1]["date"] == "2024-12-01", (
            f"Last date should be 2024-12-01, got {date_index.iloc[-1]['date']}"
        )

    def test_month_indices_sequential(self, date_index):
        expected = list(range(204))
        actual   = date_index["month_idx"].tolist()
        assert actual == expected, "month_idx values are not sequential 0–203."


class TestSplitIndices:
    def test_expected_phases(self, split_indices):
        for phase in ("train", "val", "test"):
            assert phase in split_indices, f"Phase '{phase}' missing from split_indices."

    def test_no_temporal_overlap(self, split_indices):
        train_end = split_indices["train"][1]
        val_start = split_indices["val"][0]
        val_end   = split_indices["val"][1]
        test_start = split_indices["test"][0]
        assert train_end < val_start,  "Train and val overlap."
        assert val_end   < test_start, "Val and test overlap."

    def test_train_period(self, split_indices, date_index):
        s, e = split_indices["train"]
        assert date_index.iloc[s]["year_month"] == "2008-01", (
            f"Train should start 2008-01, got {date_index.iloc[s]['year_month']}"
        )
        assert date_index.iloc[e]["year_month"] == "2021-12", (
            f"Train should end 2021-12, got {date_index.iloc[e]['year_month']}"
        )

    def test_val_period(self, split_indices, date_index):
        s, e = split_indices["val"]
        assert date_index.iloc[s]["year_month"] == "2022-01"
        assert date_index.iloc[e]["year_month"] == "2023-12"

    def test_test_period(self, split_indices, date_index):
        s, e = split_indices["test"]
        assert date_index.iloc[s]["year_month"] == "2024-01"
        assert date_index.iloc[e]["year_month"] == "2024-12"

    def test_test_covers_12_months(self, split_indices):
        s, e = split_indices["test"]
        assert e - s + 1 == 12, (
            f"Test period should be 12 months, got {e - s + 1}"
        )
