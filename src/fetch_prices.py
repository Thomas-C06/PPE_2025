"""Standalone script to test and preview market data downloads for GeoQuant AI.

Run directly from the src/ directory:
    python fetch_prices.py

This script is separate from loader.py (the main class) and serves as a
quick verification tool to confirm that all tickers download correctly.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running as a standalone script from the src/ directory
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import END_DATE, PRIMARY_TICKER, START_DATE, TICKER_NAMES, TICKERS
from loader import PriceLoader


def fetch_and_preview() -> None:
    """Download prices for all configured tickers and display a summary table."""
    base_dir = Path(__file__).resolve().parents[1]

    loader = PriceLoader(
        tickers=TICKERS,
        start_date=START_DATE,
        end_date=END_DATE,
        base_dir=base_dir,
    )

    print(f"GeoQuant AI -- Price Download Preview")
    print(f"Period : {START_DATE}  ->  {END_DATE}")
    print(f"Tickers: {len(TICKERS)}")
    print("-" * 60)

    raw_data = loader.fetch_data()
    processed_data = loader.compute_returns(raw_data)

    # Affiche un résumé par ticker
    print(f"{'Name':<12} {'Ticker':<14} {'Rows':>5}  {'Start':<12} {'End':<12}  {'Last Close':>10}")
    print("-" * 75)
    for ticker, df in processed_data.items():
        name = TICKER_NAMES.get(ticker, ticker)
        price_col = "Adj Close" if "Adj Close" in df.columns else "Close"
        # Gestion MultiIndex yfinance (si besoin)
        if hasattr(df.columns, "levels"):
            df.columns = df.columns.get_level_values(0)
        first_date = df.index.min().date()
        last_date  = df.index.max().date()
        try:
            last_close = df[price_col].iloc[-1]
            close_str  = f"{last_close:>10.2f}"
        except Exception:
            close_str  = f"{'N/A':>10}"
        star = " <- PRIMARY" if ticker == PRIMARY_TICKER else ""
        print(f"{name:<12} {ticker:<14} {len(df):>5}  {str(first_date):<12} {str(last_date):<12}  {close_str}{star}")

    print("-" * 75)
    print(f"All {len(TICKERS)} tickers downloaded successfully.")


if __name__ == "__main__":
    fetch_and_preview()
