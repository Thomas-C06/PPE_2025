"""Entry point for GeoQuant AI -- Bloc 1 Data Engineering pipeline."""

from __future__ import annotations

from pathlib import Path

from config import END_DATE, PRIMARY_TICKER, START_DATE, TICKER_NAMES, TICKERS
from loader import PriceLoader
from merger import DataMerger
from news_loader import NewsLoader
from visualizer import DataVisualizer


def main() -> None:
    """Orchestrate the full Bloc 1 pipeline: prices -> news -> merge -> visualise."""
    base_dir = Path(__file__).resolve().parents[1]

    # ── STEP 1 : Téléchargement des prix ──────────────────────────────────────
    print("=" * 60)
    print("STEP 1 -- Downloading market prices")
    print("=" * 60)

    loader = PriceLoader(
        tickers=TICKERS,
        start_date=START_DATE,
        end_date=END_DATE,
        base_dir=base_dir,
    )

    raw_data       = loader.fetch_data()
    processed_data = loader.compute_returns(raw_data)
    loader.save_data(raw_data, processed_data)

    print(f"  Downloaded {len(processed_data)} tickers.")

    # ── STEP 2 : Chargement et agrégation des news ────────────────────────────
    print()
    print("=" * 60)
    print("STEP 2 -- Loading and aggregating news")
    print("=" * 60)

    news_loader = NewsLoader(base_dir=base_dir)
    news_agg    = news_loader.run()

    # ── STEP 3 : Fusion prix + news + features techniques ────────────────────
    print()
    print("=" * 60)
    print(f"STEP 3 -- Merging prices with news ({PRIMARY_TICKER})")
    print("=" * 60)

    primary_df   = processed_data[PRIMARY_TICKER]
    merger       = DataMerger(base_dir=base_dir, primary_ticker=PRIMARY_TICKER)
    dataset_final = merger.run(primary_df, news_agg)

    # ── STEP 4 : Visualisations ───────────────────────────────────────────────
    print()
    print("=" * 60)
    print("STEP 4 -- Generating visualisations")
    print("=" * 60)

    ticker_name = TICKER_NAMES.get(PRIMARY_TICKER, PRIMARY_TICKER)
    viz         = DataVisualizer(base_dir=base_dir, ticker_name=ticker_name)
    viz.run(dataset_final)

    # ── RÉSUMÉ FINAL ──────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("PIPELINE SUMMARY -- BLOC 1 DATA ENGINEERING")
    print("=" * 60)

    start_dt       = dataset_final["date"].min()
    end_dt         = dataset_final["date"].max()
    days_with_news = int((dataset_final["nb_articles"] > 0).sum())
    max_drawdown   = float(dataset_final["Drawdown"].min())
    total_rows     = len(dataset_final)
    output_path    = base_dir / "data" / "processed" / "dataset_final.csv"
    charts_dir     = base_dir / "data" / "processed" / "graphiques"

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
