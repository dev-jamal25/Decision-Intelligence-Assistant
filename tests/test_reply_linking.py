"""Focused tests for reply linking in the preprocessing pipeline."""

from __future__ import annotations

import pandas as pd

from scripts.preprocess.prepare_data import (
    attach_first_support_reply,
    basic_clean_text,
    build_rag_case_base,
    build_support_reply_lookup,
    parse_response_ids,
    select_relevant_columns,
)


def _frame() -> pd.DataFrame:
    """Small mixed frame with customer + support tweets and varied response_tweet_id."""
    return pd.DataFrame(
        {
            "tweet_id": [10, 20, 21, 30, 40, 41, 50],
            "author_id": ["cust_a", "sup_x", "sup_x", "cust_b", "sup_y", "sup_y", "cust_c"],
            "inbound": [True, False, False, True, False, False, True],
            "created_at": ["t10", "t20", "t21", "t30", "t40", "t41", "t50"],
            "text": [
                "customer a asks",
                "support x first",
                "support x second",
                "customer b asks",
                "support y first",
                "support y second",
                "customer c asks (no reply)",
            ],
            # customer a -> "20,21" (comma separated, pick first that is inbound=False)
            # support x 20 -> "30" (not relevant for customer-linking)
            # support x 21 -> nan
            # customer b -> "40"  (single id)
            # support y 40 -> nan
            # support y 41 -> nan
            # customer c -> nan   (no replies)
            "response_tweet_id": ["20,21", "30", None, "40", None, None, None],
            "in_response_to_tweet_id": [None, 10.0, 10.0, None, 30.0, 30.0, None],
        }
    )


# ---------- parse_response_ids ----------

def test_parse_response_ids_null_returns_empty():
    assert parse_response_ids(None) == []
    assert parse_response_ids(float("nan")) == []
    assert parse_response_ids("") == []
    assert parse_response_ids("   ") == []


def test_parse_response_ids_single_id():
    assert parse_response_ids("42") == ["42"]
    assert parse_response_ids(42) == ["42"]


def test_parse_response_ids_comma_separated_preserves_order():
    assert parse_response_ids("20,21") == ["20", "21"]
    assert parse_response_ids(" 20 , 21 ,22 ") == ["20", "21", "22"]


# ---------- build_support_reply_lookup ----------

def test_support_reply_lookup_only_has_inbound_false_rows():
    df = basic_clean_text(select_relevant_columns(_frame()))
    lookup = build_support_reply_lookup(df)
    # Keys are string tweet_ids, values are support-side rows only
    assert set(lookup.keys()) == {"20", "21", "40", "41"}
    for row in lookup.values():
        assert not row["inbound"]


# ---------- attach_first_support_reply ----------

def test_attach_first_support_reply_picks_first_inbound_false_match():
    df = basic_clean_text(select_relevant_columns(_frame()))
    customer = df[df["inbound"]].copy()
    lookup = build_support_reply_lookup(df)
    joined = attach_first_support_reply(customer, lookup)

    # customer a: "20,21" -> first match is support tweet 20
    a = joined[joined["tweet_id"] == 10].iloc[0]
    assert a["support_reply_tweet_id"] == 20
    assert a["support_reply_text"] == "support x first"

    # customer b: single id "40" -> support tweet 40
    b = joined[joined["tweet_id"] == 30].iloc[0]
    assert b["support_reply_tweet_id"] == 40
    assert b["support_reply_text"] == "support y first"

    # customer c: no response ids -> reply fields are NA
    c = joined[joined["tweet_id"] == 50].iloc[0]
    assert pd.isna(c["support_reply_tweet_id"])
    assert pd.isna(c["support_reply_text"])


def test_attach_first_support_reply_never_returns_inbound_true_as_reply():
    """If response_tweet_id points only to inbound=True ids, we should attach nothing."""
    df = pd.DataFrame(
        {
            "tweet_id": [1, 2],
            "author_id": ["cust_a", "cust_b"],
            "inbound": [True, True],
            "created_at": ["t1", "t2"],
            "text": ["a", "b"],
            "response_tweet_id": ["2", "1"],  # customer <-> customer ids only
            "in_response_to_tweet_id": [None, None],
        }
    )
    df = basic_clean_text(select_relevant_columns(df))
    lookup = build_support_reply_lookup(df)  # empty, no inbound=False rows
    customer = df[df["inbound"]].copy()
    joined = attach_first_support_reply(customer, lookup)
    assert joined["support_reply_tweet_id"].isna().all()


# ---------- build_rag_case_base ----------

def test_build_rag_case_base_has_expected_columns_and_shape():
    df = basic_clean_text(select_relevant_columns(_frame()))
    rag = build_rag_case_base(df)
    assert list(rag.columns) == [
        "customer_tweet_id",
        "customer_author_id",
        "customer_created_at",
        "customer_text",
        "support_reply_tweet_id",
        "support_reply_author_id",
        "support_reply_created_at",
        "support_reply_text",
    ]
    # One row per customer ticket (3 customers in the fixture)
    assert len(rag) == 3
    # Two of them have replies, one does not
    assert rag["support_reply_tweet_id"].notna().sum() == 2
    assert rag["support_reply_tweet_id"].isna().sum() == 1
