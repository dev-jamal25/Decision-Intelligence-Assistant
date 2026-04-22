"""Focused unit tests for the base preprocessing functions.

These tests use small in-memory DataFrames so they run fast and do not touch
``data/raw/twcs.csv``.
"""

from __future__ import annotations

import pandas as pd

from scripts.preprocess.prepare_data import (
    RELEVANT_COLUMNS,
    basic_clean_text,
    build_customer_ticket_base,
    select_relevant_columns,
)


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "tweet_id": [1, 2, 3, 4, 5],
            "author_id": ["a", "b", "c", "d", "e"],
            "inbound": [True, False, True, True, False],
            "created_at": ["t0", "t1", "t2", "t3", "t4"],
            "text": ["hello", "  ", "world  ", None, "support reply"],
            "response_tweet_id": ["2", None, None, None, None],
            "in_response_to_tweet_id": [None, 1.0, None, None, None],
            "extra_noise_col": [0, 0, 0, 0, 0],
        }
    )


def test_select_relevant_columns_keeps_only_expected():
    df = select_relevant_columns(_sample_df())
    assert list(df.columns) == RELEVANT_COLUMNS


def test_basic_clean_text_drops_null_and_empty_and_strips():
    df = basic_clean_text(select_relevant_columns(_sample_df()))
    # Null text (id=4) and whitespace-only (id=2) are dropped
    assert set(df["tweet_id"]) == {1, 3, 5}
    # Trailing whitespace is stripped
    assert df.loc[df["tweet_id"] == 3, "text"].iloc[0] == "world"


def test_basic_clean_text_inbound_is_real_bool():
    df = basic_clean_text(select_relevant_columns(_sample_df()))
    assert df["inbound"].dtype == bool


def test_build_customer_ticket_base_only_inbound_true():
    cleaned = basic_clean_text(select_relevant_columns(_sample_df()))
    customer = build_customer_ticket_base(cleaned)
    assert customer["inbound"].all()
    assert set(customer["tweet_id"]) == {1, 3}
