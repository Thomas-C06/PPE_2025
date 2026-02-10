"""Entry point for GeoQuant AI market data pipeline."""

from __future__ import annotations

from pathlib import Path

from config import END_DATE, START_DATE, TICKERS
from loader import PriceLoader


def main() -> None:
    """Orchestrate data download, processing, and persistence."""
    base_dir = Path(__file__).resolve().parents[1]

    loader = PriceLoader(
        tickers=TICKERS,
        start_date=START_DATE,
        end_date=END_DATE,
        base_dir=base_dir,
    )

    raw_data = loader.fetch_data()
    processed_data = loader.compute_returns(raw_data)
    loader.save_data(raw_data, processed_data)

    for ticker, df in processed_data.items():
        print(f"\nPreview for {ticker}:")
        print(df.head())


if __name__ == "__main__":
    main()
