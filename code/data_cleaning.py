# code/data_cleaning.py
from __future__ import annotations

import re
from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_PATH = DATA_DIR / "used_cars.csv"
PROCESSED_DIR = DATA_DIR / "processed"
PROCESSED_PATH = PROCESSED_DIR / "used_cars_clean.csv"


def _to_int_from_milage(x: str) -> float:
    """Convert '51,000 mi.' -> 51000. Returns NaN if cannot parse."""
    if pd.isna(x):
        return float("nan")
    s = str(x)
    s = s.replace("mi.", "").replace("mi", "")
    s = s.replace(",", "").strip()
    return float(s) if s and s.isdigit() else float("nan")


def _to_float_from_price(x: str) -> float:
    """Convert '$10,300' -> 10300.0. Returns NaN if cannot parse."""
    if pd.isna(x):
        return float("nan")
    s = str(x).replace("$", "").replace(",", "").strip()
    # allow "10300" or "10300.50"
    return float(s) if re.match(r"^\d+(\.\d+)?$", s) else float("nan")


def clean_used_cars(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Standardize column names (optional, but nice)
    df.columns = [c.strip().lower() for c in df.columns]

    # Parse numeric fields
    df["milage"] = df["milage"].apply(_to_int_from_milage)
    df["price"] = df["price"].apply(_to_float_from_price)

    # Clean title: treat NaN as "No" (or Unknown). We'll use No here.
    df["clean_title"] = df["clean_title"].fillna("No")

    # Basic cleanup of strings
    for c in ["brand", "model", "fuel_type", "engine", "transmission", "ext_col", "int_col", "accident", "clean_title"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()

    # Drop rows without target
    df = df.dropna(subset=["price"])

    # Optional: drop rows with missing milage/year (or keep and impute later)
    # For now, keep milage missing; model pipeline can handle imputation.

    return df


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    df_raw = pd.read_csv(RAW_PATH)
    df_clean = clean_used_cars(df_raw)

    df_clean.to_csv(PROCESSED_PATH, index=False)
    print(f"Saved cleaned data: {PROCESSED_PATH}")
    print(df_clean.head())


if __name__ == "__main__":
    main()
