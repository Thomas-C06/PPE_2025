"""Entry point for GeoQuant AI -- unified historical data pipeline."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from config import END_DATE, PRIMARY_TICKER, START_DATE, TICKER_NAMES, TICKERS
from loader import PriceLoader
from merger import DataMerger
from news_loader import NewsLoader
from visualizer import DataVisualizer


def main() -> None:
    """
    Build the historical market dataset from one coherent news source.

    Order:
        1. Load and aggregate the processed Kaggle news file.
        2. Restrict the price window to the overlap between market config and
           available news coverage.
        3. Merge prices + historical news, then compute technical features.
        4. Save charts used by the dashboard.
    """
    base_dir = Path(__file__).resolve().parents[1]

    print("=" * 60)
    print("STEP 1 -- Loading unified historical news source")
    print("=" * 60)

    news_loader = NewsLoader(base_dir=base_dir)
    news_agg = news_loader.run()

    news_start = pd.Timestamp(news_agg["date"].min()).normalize()
    news_end = pd.Timestamp(news_agg["date"].max()).normalize()

    effective_start = max(pd.Timestamp(START_DATE), news_start)
    effective_end = min(pd.Timestamp(END_DATE), news_end)

    print()
    print("=" * 60)
    print("STEP 2 -- Downloading market prices on overlapping period")
    print("=" * 60)
    print(
        f"  Historical overlap used for Bloc 1: "
        f"{effective_start.date()} -> {effective_end.date()}"
    )

    loader = PriceLoader(
        tickers=TICKERS,
        start_date=str(effective_start.date()),
        end_date=str((effective_end + pd.Timedelta(days=1)).date()),
        base_dir=base_dir,
    )

    raw_data = loader.fetch_data()
    processed_data = loader.compute_returns(raw_data)
    loader.save_data(raw_data, processed_data)

    print(f"  Downloaded {len(processed_data)} tickers.")

    print()
    print("=" * 60)
    print(f"STEP 3 -- Merging prices with unified news ({PRIMARY_TICKER})")
    print("=" * 60)

    primary_df = processed_data[PRIMARY_TICKER]
    merger = DataMerger(base_dir=base_dir, primary_ticker=PRIMARY_TICKER)
    dataset_final = merger.run(primary_df, news_agg)

    print()
    print("=" * 60)
    print("STEP 4 -- Generating visualisations")
    print("=" * 60)

    ticker_name = TICKER_NAMES.get(PRIMARY_TICKER, PRIMARY_TICKER)
    viz = DataVisualizer(base_dir=base_dir, ticker_name=ticker_name)
    viz.run(dataset_final)

    print()
    print("=" * 60)
    print("PIPELINE SUMMARY -- BLOC 1 DATA ENGINEERING")
    print("=" * 60)

    start_dt = dataset_final["date"].min()
    end_dt = dataset_final["date"].max()
    days_with_news = int((dataset_final["nb_articles"] > 0).sum())
    max_drawdown = float(dataset_final["Drawdown"].min())
    total_rows = len(dataset_final)
    output_path = base_dir / "data" / "processed" / "dataset_final.csv"
    charts_dir = base_dir / "data" / "processed" / "graphiques"

    print(f"  Ticker         : {ticker_name} ({PRIMARY_TICKER})")
    print(f"  Period         : {start_dt.date()}  ->  {end_dt.date()}")
    print(f"  Total rows     : {total_rows:>6} trading days")
    print(f"  Days with news : {days_with_news:>6} ({days_with_news / total_rows:.0%})")
    print(f"  Max Drawdown   : {max_drawdown:.2%}")
    print(f"  Dataset saved  : {output_path}")
    print(f"  Charts saved   : {charts_dir}")
    print("=" * 60)
    print("Bloc 1 complete. Ready for Bloc 2 -- NLP (FinBERT).")


if __name__ == "__main__":
    main()
