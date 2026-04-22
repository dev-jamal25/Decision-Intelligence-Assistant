"""Base preprocessing for the Twitter Customer Support dataset.

Scope (so far):
    1. Load ``data/raw/twcs.csv``
    2. Select a fixed set of relevant columns
    3. Apply safe base cleaning (non-null text, strip, drop empty)
    4. Build a customer-ticket base dataset: ``inbound = True`` only
    5. Build a RAG case base by linking each customer tweet to its first
       ``inbound = False`` support reply (when available)
    6. Save both artefacts to ``data/interim/``

Intentionally NOT implemented here:
    - narrative embedding text field
    - weak labeling
    - engineered features
    - multi-hop thread reconstruction
    - embeddings / vector DB / API logic

Dataset interpretation:
    inbound = True  -> customer / user tweet sent to support
    inbound = False -> support / company-side tweet or reply
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_CSV = PROJECT_ROOT / "data" / "raw" / "twcs.csv"
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"

RELEVANT_COLUMNS = [
    "tweet_id",
    "author_id",
    "inbound",
    "created_at",
    "text",
    "response_tweet_id",
    "in_response_to_tweet_id",
]

CUSTOMER_TICKETS_FILENAME = "customer_tickets_base.parquet"
RAG_CASE_BASE_FILENAME = "rag_case_base.parquet"


def load_data(path: Path = RAW_CSV, nrows: int | None = None) -> pd.DataFrame:
    """Load the raw twcs CSV. ``nrows`` is useful for smoke tests."""
    return pd.read_csv(path, nrows=nrows)


def select_relevant_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df[RELEVANT_COLUMNS].copy()


def basic_clean_text(df: pd.DataFrame) -> pd.DataFrame:
    """Keep non-null text, cast to string, strip whitespace, drop empties.

    Also coerces ``inbound`` to a real boolean dtype when the source is
    already boolean-like (True/False strings or bools).
    """
    out = df.dropna(subset=["text"]).copy()
    out["text"] = out["text"].astype(str).str.strip()
    out = out[out["text"] != ""].copy()

    if out["inbound"].dtype != bool:
        out["inbound"] = out["inbound"].map(
            {True: True, False: False, "True": True, "False": False}
        )
        out = out.dropna(subset=["inbound"]).copy()
        out["inbound"] = out["inbound"].astype(bool)

    return out


def build_customer_ticket_base(df: pd.DataFrame) -> pd.DataFrame:
    """Customer-side tickets only (``inbound == True``)."""
    return df[df["inbound"]].copy()


def parse_response_ids(value: object) -> list[str]:
    """Parse the ``response_tweet_id`` field into a clean list of string ids.

    The raw column can be null, a single id, or a comma-separated list.
    Ids are returned as strings in their original order so downstream joins
    remain dtype-independent.
    """
    if value is None:
        return []
    if isinstance(value, float) and pd.isna(value):
        return []
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return []
    return [part.strip() for part in text.split(",") if part.strip()]


def build_support_reply_lookup(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Index ``inbound = False`` rows by their (string) ``tweet_id``."""
    support = df[~df["inbound"]]
    lookup: dict[str, pd.Series] = {}
    for _, row in support.iterrows():
        lookup[str(row["tweet_id"])] = row
    return lookup


def attach_first_support_reply(
    customer: pd.DataFrame,
    support_lookup: dict[str, pd.Series],
) -> pd.DataFrame:
    """Return a new dataframe: customer rows with first linked support reply.

    For each customer ticket, walks its ``response_tweet_id`` list in order
    and picks the first id that resolves to a support-side tweet in the
    lookup. Customer tickets without a linked support reply are still kept
    (reply fields left as ``pd.NA``).
    """
    reply_tweet_ids: list[object] = []
    reply_author_ids: list[object] = []
    reply_created_ats: list[object] = []
    reply_texts: list[object] = []

    for _, row in customer.iterrows():
        candidates = parse_response_ids(row.get("response_tweet_id"))
        chosen = None
        for cid in candidates:
            if cid in support_lookup:
                chosen = support_lookup[cid]
                break

        if chosen is None:
            reply_tweet_ids.append(pd.NA)
            reply_author_ids.append(pd.NA)
            reply_created_ats.append(pd.NA)
            reply_texts.append(pd.NA)
        else:
            reply_tweet_ids.append(chosen["tweet_id"])
            reply_author_ids.append(chosen["author_id"])
            reply_created_ats.append(chosen["created_at"])
            reply_texts.append(chosen["text"])

    out = customer.copy()
    out["support_reply_tweet_id"] = reply_tweet_ids
    out["support_reply_author_id"] = reply_author_ids
    out["support_reply_created_at"] = reply_created_ats
    out["support_reply_text"] = reply_texts
    return out


def build_rag_case_base(df: pd.DataFrame) -> pd.DataFrame:
    """Build the RAG case base dataframe from a cleaned tweets frame.

    One row per customer ticket, with the first linked support reply attached
    if one exists. Columns are renamed to make the customer/reply split
    explicit downstream.
    """
    customer = build_customer_ticket_base(df)
    lookup = build_support_reply_lookup(df)
    joined = attach_first_support_reply(customer, lookup)

    return joined.rename(
        columns={
            "tweet_id": "customer_tweet_id",
            "author_id": "customer_author_id",
            "created_at": "customer_created_at",
            "text": "customer_text",
        }
    )[
        [
            "customer_tweet_id",
            "customer_author_id",
            "customer_created_at",
            "customer_text",
            "support_reply_tweet_id",
            "support_reply_author_id",
            "support_reply_created_at",
            "support_reply_text",
        ]
    ]


def save_outputs(
    customer_df: pd.DataFrame,
    rag_df: pd.DataFrame,
    interim_dir: Path = INTERIM_DIR,
) -> dict[str, Path]:
    interim_dir.mkdir(parents=True, exist_ok=True)
    customer_path = interim_dir / CUSTOMER_TICKETS_FILENAME
    rag_path = interim_dir / RAG_CASE_BASE_FILENAME
    customer_df.to_parquet(customer_path, index=False)
    rag_df.to_parquet(rag_path, index=False)
    return {"customer_tickets": customer_path, "rag_case_base": rag_path}


def run(nrows: int | None = None) -> dict:
    """End-to-end base preprocessing + reply linking. Returns a summary dict."""
    raw = load_data(nrows=nrows)
    total_rows = len(raw)

    selected = select_relevant_columns(raw)
    cleaned = basic_clean_text(selected)
    rows_after_cleaning = len(cleaned)

    customer = build_customer_ticket_base(cleaned)
    rows_customer = len(customer)

    rag_base = build_rag_case_base(cleaned)
    linked_replies = int(rag_base["support_reply_tweet_id"].notna().sum())

    paths = save_outputs(customer, rag_base)

    summary = {
        "total_rows": total_rows,
        "rows_after_cleaning": rows_after_cleaning,
        "rows_customer_tickets": rows_customer,
        "rag_case_rows": len(rag_base),
        "linked_replies": linked_replies,
        "customer_tickets_path": str(paths["customer_tickets"]),
        "rag_case_base_path": str(paths["rag_case_base"]),
    }
    logger.info("total rows loaded:          %d", summary["total_rows"])
    logger.info("rows after base cleaning:   %d", summary["rows_after_cleaning"])
    logger.info("rows in customer-ticket ds: %d", summary["rows_customer_tickets"])
    logger.info("rag case rows:              %d", summary["rag_case_rows"])
    logger.info("linked support replies:     %d", summary["linked_replies"])
    logger.info("wrote: %s", summary["customer_tickets_path"])
    logger.info("wrote: %s", summary["rag_case_base_path"])
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Base preprocessing for twcs.csv")
    parser.add_argument(
        "--nrows",
        type=int,
        default=None,
        help="Optional row cap for smoke-testing on a subset.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run(nrows=args.nrows)


if __name__ == "__main__":
    main()
