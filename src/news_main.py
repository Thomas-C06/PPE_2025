"""Entry point for GeoQuant AI news data pipeline."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from google_news_historical_loader import GoogleNewsHistoricalLoader
from news_config import NEWS_END_DATE, NEWS_START_DATE
from kaggle_news_loader import NewsLoader


def main() -> None:
    """Orchestrate news download, processing, and persistence."""
    base_dir = Path(__file__).resolve().parents[1]

    # ── Dataset principal 2022-2024 ──────────────────────────────────────────
    # Kaggle "S&P 500 with Financial News Headlines (2008-2024)"
    # URL : https://www.kaggle.com/datasets/dyutidasmahaptra/s-and-p-500-with-financial-news-headlines-20082024
    #
    # Instructions :
    #   1. Télécharger le ZIP sur Kaggle
    #   2. Extraire et placer le CSV dans data/raw/
    #   3. Vérifier que CSV_FILE correspond au nom exact du fichier extrait

    DATASET  = "sp500_news_2024"
    CSV_FILE = "sp500_news_headlines.csv"                # nom du fichier après extraction

    csv_path = base_dir / "data" / "raw" / CSV_FILE

    loader = NewsLoader(
        csv_path=csv_path,
        dataset=DATASET,
        start_date=NEWS_START_DATE,
        end_date=NEWS_END_DATE,
        base_dir=base_dir,
    )

    raw_df = loader.load_raw()
    processed_df = loader.process(raw_df)
    processed_df["source"] = processed_df["source"].fillna("Kaggle S&P 500 dataset")

    reliable_cutoff = pd.Timestamp(processed_df["date"].max()).normalize()
    extension_start = reliable_cutoff + pd.Timedelta(days=1)
    extension_end = pd.Timestamp(NEWS_END_DATE).normalize()

    combined_df = processed_df.copy()
    if extension_start <= extension_end:
        extender = GoogleNewsHistoricalLoader(
            base_dir=base_dir,
            start_date=str(extension_start.date()),
            end_date=str(extension_end.date()),
        )
        extension_df = extender.run()
        if not extension_df.empty:
            if "url" in extension_df.columns and "url" not in combined_df.columns:
                combined_df["url"] = pd.NA
            if "published_at" in extension_df.columns and "published_at" not in combined_df.columns:
                combined_df["published_at"] = pd.NaT

            extension_append = extension_df.copy()
            for col in combined_df.columns:
                if col not in extension_append.columns:
                    extension_append[col] = pd.NA
            combined_df = pd.concat(
                [combined_df, extension_append[combined_df.columns]],
                ignore_index=True,
            )
            combined_df = (
                combined_df
                .drop_duplicates(subset=["date", "headline", "source"])
                .sort_values(["date", "headline"])
                .reset_index(drop=True)
            )

    print("\n[news_main] Historical source mix:")
    print(
        "           - Kaggle S&P 500 dataset up to "
        f"{reliable_cutoff.date()}"
    )
    if len(combined_df) > len(processed_df):
        print(
            "           - Google News RSS historical extension with real pubDate from "
            f"{extension_start.date()} to {combined_df['date'].max().date()}"
        )
    else:
        print("           - No reliable extension retained after the Kaggle cutoff.")

    loader.save_data(raw_df, combined_df)

    print("\nPreview of processed news data:")
    print(combined_df[["date", "headline", "source"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
