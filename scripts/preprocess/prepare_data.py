"""Base preprocessing for the Twitter Customer Support dataset.

Scope (this slice only):
    1. Load ``data/raw/twcs.csv``
    2. Select a fixed set of relevant columns
    3. Apply safe base cleaning (non-null text, strip, drop empty)
    4. Build a customer-ticket base dataset: ``inbound = True`` only
    5. Save the cleaned customer-ticket dataset to ``data/interim/``

Intentionally NOT implemented here:
    - weak labeling
    - engineered features
    - reply linking
    - RAG corpus building
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


def save_outputs(df: pd.DataFrame, interim_dir: Path = INTERIM_DIR) -> Path:
    interim_dir.mkdir(parents=True, exist_ok=True)
    out_path = interim_dir / CUSTOMER_TICKETS_FILENAME
    df.to_parquet(out_path, index=False)
    return out_path


def run(nrows: int | None = None) -> dict:
    """End-to-end base preprocessing. Returns a small summary dict."""
    raw = load_data(nrows=nrows)
    total_rows = len(raw)

    selected = select_relevant_columns(raw)
    cleaned = basic_clean_text(selected)
    rows_after_cleaning = len(cleaned)

    customer = build_customer_ticket_base(cleaned)
    rows_customer = len(customer)

    out_path = save_outputs(customer)

    summary = {
        "total_rows": total_rows,
        "rows_after_cleaning": rows_after_cleaning,
        "rows_customer_tickets": rows_customer,
        "output_path": str(out_path),
    }
    logger.info("total rows loaded:          %d", summary["total_rows"])
    logger.info("rows after base cleaning:   %d", summary["rows_after_cleaning"])
    logger.info("rows in customer-ticket ds: %d", summary["rows_customer_tickets"])
    logger.info("wrote: %s", summary["output_path"])
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
