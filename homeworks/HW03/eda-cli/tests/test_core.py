from __future__ import annotations

import pandas as pd
import numpy as np
import pytest

from eda_cli.core import (
    compute_quality_flags,
    correlation_matrix,
    flatten_summary_for_print,
    missing_table,
    summarize_dataset,
    top_categories,
)


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [10, 20, 30, None],
            "height": [140, 150, 160, 170],
            "city": ["A", "B", "A", None],
        }
    )


def test_summarize_dataset_basic():
    df = _sample_df()
    summary = summarize_dataset(df)

    assert summary.n_rows == 4
    assert summary.n_cols == 3
    assert any(c.name == "age" for c in summary.columns)
    assert any(c.name == "city" for c in summary.columns)

    summary_df = flatten_summary_for_print(summary)
    assert "name" in summary_df.columns
    assert "missing_share" in summary_df.columns


def test_missing_table_and_quality_flags():
    df = _sample_df()
    missing_df = missing_table(df)

    assert "missing_count" in missing_df.columns
    assert missing_df.loc["age", "missing_count"] == 1

    summary = summarize_dataset(df)
    flags = compute_quality_flags(df, summary, missing_df) 
    assert 0.0 <= flags["quality_score"] <= 1.0


def test_correlation_and_top_categories():
    df = _sample_df()
    corr = correlation_matrix(df)
    assert "age" in corr.columns or corr.empty is False

    top_cats = top_categories(df, max_columns=5, top_k=2)
    assert "city" in top_cats
    city_table = top_cats["city"]
    assert "value" in city_table.columns
    assert len(city_table) <= 2


@pytest.mark.parametrize("test_case", [
    "duplicates", 
    "no_duplicates", 
    "constant_columns"
])
def test_compute_quality_flags_new_logic(test_case):
    
    if test_case == "duplicates":
        data = {
            'user_id': [1, 2, 1, 3, 2],
            'value': [10, 20, 10, 30, 20],
            'const': [42, 42, 42, 42, 42]
        }
        
    elif test_case == "no_duplicates":
        data = {
            'user_id': list(range(1, 101)), 
            'value': list(range(10, 110))
        }
    
    else:
        data = {
            'id': [1, 2, 3],
            'constant1': ['A', 'A', 'A'],
            'constant2': [0, 0, 0]
        }
    
    df = pd.DataFrame(data)
    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    
    flags = compute_quality_flags(df, summary, missing_df)

    expected_dup_rows = 4 if test_case == "duplicates" else 0
    assert flags["num_duplicate_rows"] == expected_dup_rows
    assert flags["has_duplicate_rows"] == (expected_dup_rows > 0)
    
    summary_df = flatten_summary_for_print(summary)
    num_constants = int((summary_df["unique"] <= 1).sum())
    assert flags["no_constant_columns"] == (num_constants == 0)
    assert flags["some_constant_columns"] == (0 < num_constants < 10)
    
    if test_case == "duplicates":
        assert flags["quality_score"] < 0.9 
    elif test_case == "no_duplicates":
        assert flags["quality_score"] == 1.0 
    else: 
        assert flags["quality_score"] < 0.95
